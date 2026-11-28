from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.stats import spearmanr
from wfl.configset import ConfigSet

from mlipflow.data.io import setup_logging

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from mace.calculators import MACECalculator

setup_logging()
logger = logging.getLogger(__name__)

DEFAULT_LAMBDA_GRID: tuple[float, ...] = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
DEFAULT_CALIBRATION_DELTAS: tuple[float, ...] = (0.50, 0.20, 0.10)


def _iter_configs(configs: str | Path | ConfigSet | Iterable[Atoms]) -> Iterable[Atoms]:
    """Yield ``Atoms`` from a file path, ``ConfigSet``, or iterable of ``Atoms``."""
    if isinstance(configs, (str, Path)):
        return ConfigSet(str(configs))
    return configs


class PerElementLLPR:
    """Per-element Last-Layer Prediction Rigidity (LLPR) with split-conformal calibration.

    Operates on frozen MACE descriptors to provide calibrated per-atom and
    worst-atom structure force-error bounds in physical units (eV/A):

    1. Streams over training configurations to accumulate per-element ``float64``
       Gram matrices :math:`G_Z = F_Z^T F_Z` and lower-triangular Cholesky factors
       :math:`L_Z` of :math:`A_Z = G_Z + \\sigma_Z^2 I`.
    2. Evaluates deterministic raw leverage :math:`\\sqrt{f_i^T A_Z^{-1} f_i}` via
       ``torch.linalg.solve_triangular`` (never explicit inversion).
    3. Fits per-element split-conformal multipliers ``q_atom_by_z[Z]`` and
       structure-level correction factors ``q_struct`` at multiple coverage levels
       (default ``deltas=(0.50, 0.20, 0.10)``) in a single pass over held-out
       ``dft_test`` configurations.

    Parameters
    ----------
    lambda_reg : float, default=1e-4
        Dimensionless ridge regularization scaled by ``tr(G_Z) / D`` per element.
    invariants_only : bool, default=True
        Whether to request invariant-only descriptors from ``MACECalculator``.
    device : str, default="cpu"
        PyTorch device for Gram matrices and Cholesky triangular solves.
    default_delta : float, default=0.20
        Default miscoverage rate used by ``score_atoms`` / ``score_configs`` when
        ``calibrated=True`` and ``delta`` is omitted (0.20 corresponds to 80% coverage).
    eps : float, default=1e-12
        Numerical stabilizer added to raw denominators in conformal ratios.
    """

    def __init__(
        self,
        lambda_reg: float = 1e-4,
        invariants_only: bool = True,
        device: str = "cpu",
        default_delta: float = 0.20,
        eps: float = 1e-12,
    ) -> None:
        self.lambda_reg = float(lambda_reg)
        self.invariants_only = bool(invariants_only)
        self.device = str(device)
        self.default_delta = float(default_delta)
        self.eps = float(eps)

        # Fitted state
        self.gram_matrices: dict[int, torch.Tensor] = {}
        self.atom_counts: dict[int, int] = {}
        self.cholesky_factors: dict[int, torch.Tensor] = {}
        self.feature_dim: int | None = None

        # Multi-level split-conformal calibration keyed by miscoverage delta
        self.calibrations: dict[float, dict[str, Any]] = {}
        self.element_metrics: dict[int, dict[str, Any]] = {}

    @property
    def q_atom(self) -> float:
        """Global fallback multiplier at ``default_delta`` (1.0 if uncalibrated)."""
        cal = self.calibrations.get(self.default_delta)
        return float(cal["q_atom"]) if cal is not None else 1.0

    @property
    def q_atom_by_z(self) -> dict[int, float]:
        """Per-element multipliers at ``default_delta`` (empty if uncalibrated)."""
        cal = self.calibrations.get(self.default_delta)
        return dict(cal["q_atom_by_z"]) if cal is not None else {}

    @property
    def q_struct(self) -> float:
        """Structure-level conformal correction at ``default_delta`` (1.0 if uncalibrated)."""
        cal = self.calibrations.get(self.default_delta)
        return float(cal["q_struct"]) if cal is not None else 1.0

    def get_calibration(self, delta: float | None = None) -> dict[str, Any]:
        """Return calibration multipliers for the requested miscoverage level ``delta``.

        Parameters
        ----------
        delta : float or None, optional
            Miscoverage rate (e.g. ``0.50``, ``0.20``, ``0.10``). If ``None``,
            uses ``self.default_delta`` (returning neutral multipliers if ``calibrate``
            has not been called yet).

        Returns
        -------
        dict of str to Any
            Dictionary with keys ``"q_atom"``, ``"q_atom_by_z"``, and ``"q_struct"``.

        Raises
        ------
        KeyError
            If an explicit ``delta`` was requested that is not present in
            ``self.calibrations``.
        """
        if delta is None:
            if not self.calibrations:
                return {"q_atom": 1.0, "q_atom_by_z": {}, "q_struct": 1.0}
            target_delta = self.default_delta
        else:
            target_delta = float(delta)

        if target_delta in self.calibrations:
            return self.calibrations[target_delta]

        for key, value in self.calibrations.items():
            if math.isclose(key, target_delta, rel_tol=0.0, abs_tol=1e-9):
                return value

        available = sorted(self.calibrations.keys())
        raise KeyError(
            f"Requested calibration delta={target_delta!r} is not available. "
            f"Calibrated deltas: {available}."
        )

    @staticmethod
    def _get_active_indices(
        atoms: Atoms,
        atom_indices: slice | Sequence[int] | np.ndarray | None = None,
        exclude_fixed: bool = True,
    ) -> np.ndarray:
        """Return active atom indices, optionally excluding ASE ``FixAtoms`` constraints."""
        n_atoms = len(atoms)
        if atom_indices is not None:
            if isinstance(atom_indices, slice):
                idx = np.arange(*atom_indices.indices(n_atoms), dtype=int)
            else:
                idx = np.asarray(atom_indices, dtype=int)
        else:
            idx = np.arange(n_atoms, dtype=int)

        if exclude_fixed and atoms.constraints:
            fixed_mask = np.zeros(n_atoms, dtype=bool)
            for constraint in atoms.constraints:
                if hasattr(constraint, "index"):
                    fixed_mask[constraint.index] = True
                elif hasattr(constraint, "get_indices"):
                    fixed_mask[constraint.get_indices()] = True
            active_idx = idx[~fixed_mask[idx]]
            if len(active_idx) > 0:
                return active_idx
        return idx

    def _extract_descriptors(
        self, atoms: Atoms, mace_calc: MACECalculator | Any
    ) -> np.ndarray:
        """Extract per-atom descriptors from ``mace_calc`` as a ``float64`` array."""
        desc = mace_calc.get_descriptors(atoms, invariants_only=self.invariants_only)
        return np.asarray(desc, dtype=np.float64)

    @staticmethod
    def _compute_sigma2(gram: torch.Tensor, lambda_reg: float) -> float:
        """Compute trace-scaled prior variance ``sigma_Z^2`` for a Gram matrix."""
        dim = gram.shape[0]
        trace_scale = torch.trace(gram).item() / dim
        return max(float(lambda_reg) * trace_scale, 1e-12)

    @staticmethod
    def _factorize_gram(gram: torch.Tensor, sigma2: float) -> torch.Tensor:
        """Compute ``float64`` Cholesky factor ``L`` of ``gram + sigma2 * I``."""
        dim = gram.shape[0]
        eye = torch.eye(dim, dtype=torch.float64, device=gram.device)
        reg_mat = gram + float(sigma2) * eye
        for attempt in range(5):
            try:
                return torch.linalg.cholesky(reg_mat)
            except torch.linalg.LinAlgError:
                reg_mat = reg_mat + (10.0**attempt) * float(sigma2) * eye
        raise RuntimeError("Cholesky factorization failed for regularized Gram matrix.")

    def _factorize(self, lambda_reg: float | None = None) -> None:
        """Compute ``float64`` Cholesky factors ``L_Z`` of ``G_Z + sigma_Z^2 I``."""
        if lambda_reg is not None:
            self.lambda_reg = float(lambda_reg)

        self.cholesky_factors = {}
        for z, gram in self.gram_matrices.items():
            sigma2 = self._compute_sigma2(gram, self.lambda_reg)
            try:
                self.cholesky_factors[z] = self._factorize_gram(gram, sigma2)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Cholesky factorization failed for element Z={z}."
                ) from exc

    def fit(
        self,
        train_configs: str | Path | ConfigSet | Iterable[Atoms],
        mace_calc: MACECalculator | Any,
        atom_indices: slice | Sequence[int] | np.ndarray | None = None,
        exclude_fixed: bool = True,
    ) -> PerElementLLPR:
        """Stream over training configurations and accumulate per-element Gram matrices.

        Parameters
        ----------
        train_configs : str or Path or ConfigSet or Iterable of Atoms
            Training configurations used to build per-element Gram matrices.
        mace_calc : MACECalculator
            Frozen MACE calculator exposing ``get_descriptors(atoms, invariants_only=...)``.
        atom_indices : slice or Sequence of int or np.ndarray, optional
            Optional subset of atom indices to include.
        exclude_fixed : bool, default=True
            If True, excludes atoms constrained by ASE ``FixAtoms`` constraints.

        Returns
        -------
        PerElementLLPR
            Fitted instance (``self``).

        Raises
        ------
        ValueError
            If ``train_configs`` yields no active atoms.
        """
        self.gram_matrices = {}
        self.atom_counts = {}
        self.cholesky_factors = {}
        self.feature_dim = None

        n_configs = 0
        for at in _iter_configs(train_configs):
            n_configs += 1
            idx = self._get_active_indices(at, atom_indices, exclude_fixed)
            if len(idx) == 0:
                continue
            desc = self._extract_descriptors(at, mace_calc)[idx]
            numbers = at.get_atomic_numbers()[idx]

            if self.feature_dim is None:
                self.feature_dim = int(desc.shape[1])

            feats = torch.as_tensor(desc, dtype=torch.float64, device=self.device)
            for z in np.unique(numbers):
                z_int = int(z)
                fz = feats[numbers == z]
                if z_int not in self.gram_matrices:
                    self.gram_matrices[z_int] = torch.zeros(
                        (self.feature_dim, self.feature_dim),
                        dtype=torch.float64,
                        device=self.device,
                    )
                    self.atom_counts[z_int] = 0
                self.gram_matrices[z_int] += fz.T @ fz
                self.atom_counts[z_int] += int(fz.shape[0])

        if not self.gram_matrices:
            raise ValueError("train_configs yielded no active atoms to fit.")

        self._factorize(self.lambda_reg)
        logger.info(
            "Fitted PerElementLLPR on %d configs (D=%d, atom_counts=%s).",
            n_configs,
            self.feature_dim,
            self.atom_counts,
        )
        return self

    @staticmethod
    def _score_feature_tensor(
        feats: torch.Tensor,
        numbers: np.ndarray,
        cholesky_factors: dict[int, torch.Tensor],
    ) -> np.ndarray:
        """Compute deterministic raw per-atom LLPR std from a ``float64`` tensor."""
        scores = np.full(len(numbers), np.inf, dtype=np.float64)
        with torch.no_grad():
            for z in np.unique(numbers):
                z_int = int(z)
                if z_int not in cholesky_factors:
                    continue
                mask_np = numbers == z
                mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=feats.device)
                fz = feats[mask_t]  # (n_z, D)
                chol = cholesky_factors[z_int]  # (D, D)
                z_vec = torch.linalg.solve_triangular(chol, fz.T, upper=False)
                quad = torch.sum(z_vec**2, dim=0)
                scores[mask_np] = torch.sqrt(torch.clamp(quad, min=0.0)).cpu().numpy()
        return scores

    def _score_feature_array(
        self, feats_np: np.ndarray, numbers: np.ndarray
    ) -> np.ndarray:
        """Compute deterministic raw per-atom LLPR std ``sqrt(f^T A_Z^{-1} f)``."""
        if not self.cholesky_factors:
            raise RuntimeError(
                "PerElementLLPR must be fitted via fit() before scoring."
            )

        feats = torch.as_tensor(feats_np, dtype=torch.float64, device=self.device)
        return self._score_feature_tensor(feats, numbers, self.cholesky_factors)

    def select_sequential_batch(
        self,
        candidate_descs: Sequence[np.ndarray],
        candidate_numbers: Sequence[np.ndarray],
        force_tol_eVA: float = 0.10,
        delta: float = 0.50,
        top_k: int | None = None,
    ) -> list[int]:
        """Select non-redundant structures via greedy per-element Gram/Cholesky updates.

        Works strictly on in-memory copies of ``self.gram_matrices`` and
        ``self.cholesky_factors`` while holding training prior variances
        :math:`\\sigma_Z^2` and split-conformal multipliers fixed so calibrated
        scores decrease monotonically as environments are selected.

        Parameters
        ----------
        candidate_descs : Sequence of np.ndarray
            Active-atom descriptor arrays of shape ``(n_active, D)`` per candidate.
        candidate_numbers : Sequence of np.ndarray
            Active-atom atomic numbers of shape ``(n_active,)`` per candidate.
        force_tol_eVA : float, default=0.10
            Physical worst-atom force-error tolerance in eV/A. Candidates whose
            updated score drops to ``<= force_tol_eVA`` are pruned automatically
            when ``top_k is None``.
        delta : float, default=0.50
            Miscoverage level for calibrated bounds (``0.50`` = expected error).
        top_k : int or None, optional
            If provided, selects up to ``top_k`` candidates in greedy Gram-update
            order without applying the ``force_tol_eVA`` pruning gate (used for
            diversity-preserving fallback selection).

        Returns
        -------
        list of int
            Indices into ``candidate_descs`` in greedy selection order.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        ValueError
            If ``candidate_descs`` and ``candidate_numbers`` lengths differ.
        """
        if not self.gram_matrices or not self.cholesky_factors:
            raise RuntimeError(
                "PerElementLLPR must be fitted via fit() before batch selection."
            )
        if len(candidate_descs) != len(candidate_numbers):
            raise ValueError(
                "candidate_descs and candidate_numbers must have identical lengths."
            )
        n_candidates = len(candidate_descs)
        if n_candidates == 0:
            return []
        if top_k is not None and top_k <= 0:
            return []

        cal = self.get_calibration(delta)
        q_atom = float(cal["q_atom"])
        q_atom_by_z: dict[int, float] = dict(cal["q_atom_by_z"])
        q_struct = float(cal["q_struct"])

        # Clone working Gram/Cholesky state and freeze training prior variances sigma_Z^2
        work_grams: dict[int, torch.Tensor] = {
            z: g.detach().clone() for z, g in self.gram_matrices.items()
        }
        work_chols: dict[int, torch.Tensor] = {
            z: chol.detach().clone() for z, chol in self.cholesky_factors.items()
        }
        sigma2_by_z: dict[int, float] = {
            z: self._compute_sigma2(g, self.lambda_reg)
            for z, g in self.gram_matrices.items()
        }

        cand_feats = [
            torch.as_tensor(
                np.asarray(d, dtype=np.float64), dtype=torch.float64, device=self.device
            )
            for d in candidate_descs
        ]
        cand_nums = [np.asarray(n, dtype=int) for n in candidate_numbers]
        cand_elem_sets = [{int(z) for z in np.unique(n)} for n in cand_nums]

        def _score_candidate(idx: int) -> float:
            nums = cand_nums[idx]
            if len(nums) == 0:
                return 0.0
            raw = self._score_feature_tensor(cand_feats[idx], nums, work_chols)
            cal_atom = self._apply_atom_calibration(raw, nums, q_atom, q_atom_by_z)
            return q_struct * float(np.max(cal_atom))

        scores = [_score_candidate(i) for i in range(n_candidates)]

        if top_k is not None:
            remaining = list(range(n_candidates))
            max_picks: int | None = min(int(top_k), n_candidates)
        else:
            remaining = [i for i, s in enumerate(scores) if s > force_tol_eVA]
            max_picks = None

        selected: list[int] = []
        while remaining:
            if max_picks is not None and len(selected) >= max_picks:
                break

            best_idx = max(remaining, key=lambda idx: (scores[idx], -idx))
            selected.append(best_idx)
            remaining.remove(best_idx)

            if not remaining or (max_picks is not None and len(selected) >= max_picks):
                break

            feats_best = cand_feats[best_idx]
            nums_best = cand_nums[best_idx]
            updated_z: set[int] = set()
            for z in cand_elem_sets[best_idx]:
                mask_t = torch.as_tensor(
                    nums_best == z, dtype=torch.bool, device=self.device
                )
                fz = feats_best[mask_t]
                if fz.shape[0] == 0:
                    continue
                gram_update = fz.T @ fz
                if z not in work_grams:
                    work_grams[z] = gram_update
                    sigma2_by_z[z] = self._compute_sigma2(
                        work_grams[z], self.lambda_reg
                    )
                else:
                    work_grams[z] = work_grams[z] + gram_update
                work_chols[z] = self._factorize_gram(work_grams[z], sigma2_by_z[z])
                updated_z.add(z)

            if updated_z:
                for idx in remaining:
                    if cand_elem_sets[idx] & updated_z:
                        scores[idx] = _score_candidate(idx)

            if max_picks is None:
                remaining = [idx for idx in remaining if scores[idx] > force_tol_eVA]

        return selected

    @staticmethod
    def _apply_atom_calibration(
        raw_scores: np.ndarray,
        numbers: np.ndarray,
        q_atom: float,
        q_atom_by_z: dict[int, float],
    ) -> np.ndarray:
        """Scale raw per-atom LLPR scores using element-specific multipliers."""
        cal_scores = np.empty_like(raw_scores, dtype=np.float64)
        for z in np.unique(numbers):
            mask = numbers == z
            q_z = q_atom_by_z.get(int(z), q_atom)
            cal_scores[mask] = q_z * raw_scores[mask]
        return cal_scores

    def score_atoms(
        self,
        atoms: Atoms,
        mace_calc: MACECalculator | Any,
        atom_indices: slice | Sequence[int] | np.ndarray | None = None,
        exclude_fixed: bool = True,
        calibrated: bool = True,
        delta: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Score a single ``Atoms`` configuration.

        Parameters
        ----------
        atoms : Atoms
            Atomic configuration to score.
        mace_calc : MACECalculator
            Frozen MACE calculator exposing ``get_descriptors``.
        atom_indices : slice or Sequence of int or np.ndarray, optional
            Optional subset of active atom indices.
        exclude_fixed : bool, default=True
            If True, excludes fixed atoms from scoring (assigned ``0.0``).
        calibrated : bool, default=True
            If True, applies per-element multipliers prior to max-pooling and
            scales the structure maximum by ``q_struct``.
        delta : float or None, optional
            Miscoverage level to use when ``calibrated=True`` (e.g. ``0.50``,
            ``0.20``, ``0.10``). Defaults to ``self.default_delta``.

        Returns
        -------
        atom_scores : np.ndarray of shape (len(atoms),)
            Per-atom LLPR score (``0.0`` on inactive/fixed atoms; ``inf`` on unseen Z).
        struct_score : float
            Worst active-atom score (calibrated in eV/A if ``calibrated=True``).
        """
        idx = self._get_active_indices(atoms, atom_indices, exclude_fixed)
        desc = self._extract_descriptors(atoms, mace_calc)[idx]
        numbers = atoms.get_atomic_numbers()[idx]

        raw_active = self._score_feature_array(desc, numbers)
        full_scores = np.zeros(len(atoms), dtype=np.float64)

        if calibrated:
            cal = self.get_calibration(delta)
            cal_active = self._apply_atom_calibration(
                raw_active,
                numbers,
                q_atom=float(cal["q_atom"]),
                q_atom_by_z=cal["q_atom_by_z"],
            )
            full_scores[idx] = cal_active
            struct_score = float(cal["q_struct"]) * float(np.max(cal_active))
            return full_scores, struct_score

        full_scores[idx] = raw_active
        return full_scores, float(np.max(raw_active))

    def score_atoms_multi_delta(
        self,
        atoms: Atoms,
        mace_calc: MACECalculator | Any,
        deltas: Sequence[float] = (0.50, 0.10),
        atom_indices: slice | Sequence[int] | np.ndarray | None = None,
        exclude_fixed: bool = True,
    ) -> dict[float, tuple[np.ndarray, float] | None]:
        """Score a single ``Atoms`` configuration at multiple ``delta`` levels in one descriptor pass.

        Extracts active-atom descriptors and evaluates raw per-atom LLPR leverage
        once, then applies the split-conformal calibration multipliers for each
        requested ``delta``. Any ``delta`` not present in ``self.calibrations``
        yields ``None`` instead of raising ``KeyError``.

        Parameters
        ----------
        atoms : Atoms
            Atomic configuration to score.
        mace_calc : MACECalculator
            Frozen MACE calculator exposing ``get_descriptors``.
        deltas : Sequence of float, default=(0.50, 0.10)
            Miscoverage levels to evaluate.
        atom_indices : slice or Sequence of int or np.ndarray, optional
            Optional subset of active atom indices.
        exclude_fixed : bool, default=True
            If True, excludes fixed atoms from scoring (assigned ``0.0``).

        Returns
        -------
        dict of float to (tuple of (np.ndarray, float) or None)
            Mapping ``float(delta) -> (atom_scores, struct_score)`` for every
            calibrated delta, or ``None`` if that delta was not calibrated.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self.cholesky_factors:
            raise RuntimeError(
                "PerElementLLPR must be fitted via fit() before scoring."
            )

        idx = self._get_active_indices(atoms, atom_indices, exclude_fixed)
        n_atoms = len(atoms)
        if len(idx) > 0:
            desc = self._extract_descriptors(atoms, mace_calc)[idx]
            numbers = atoms.get_atomic_numbers()[idx]
            raw_active = self._score_feature_array(desc, numbers)
        else:
            numbers = np.empty(0, dtype=int)
            raw_active = np.empty(0, dtype=np.float64)

        results: dict[float, tuple[np.ndarray, float] | None] = {}
        for delta in deltas:
            delta_f = float(delta)
            try:
                cal = self.get_calibration(delta_f)
            except KeyError:
                results[delta_f] = None
                continue

            full_scores = np.zeros(n_atoms, dtype=np.float64)
            if len(idx) > 0:
                cal_active = self._apply_atom_calibration(
                    raw_active,
                    numbers,
                    q_atom=float(cal["q_atom"]),
                    q_atom_by_z=cal["q_atom_by_z"],
                )
                full_scores[idx] = cal_active
                struct_score = float(cal["q_struct"]) * float(np.max(cal_active))
            else:
                struct_score = 0.0
            results[delta_f] = (full_scores, struct_score)

        return results

    def score_configs(
        self,
        configs: str | Path | ConfigSet | Iterable[Atoms],
        mace_calc: MACECalculator | Any,
        atom_indices: slice | Sequence[int] | np.ndarray | None = None,
        exclude_fixed: bool = True,
        calibrated: bool = True,
        delta: float | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Score a collection of ``Atoms`` configurations.

        Parameters
        ----------
        configs : str or Path or ConfigSet or Iterable of Atoms
            Configurations to score.
        mace_calc : MACECalculator
            Frozen MACE calculator exposing ``get_descriptors``.
        atom_indices : slice or Sequence of int or np.ndarray, optional
            Optional subset of active atom indices.
        exclude_fixed : bool, default=True
            If True, excludes fixed atoms from scoring.
        calibrated : bool, default=True
            If True, returns calibrated eV/A bounds at ``delta``.
        delta : float or None, optional
            Miscoverage level when ``calibrated=True``. Defaults to ``self.default_delta``.

        Returns
        -------
        all_atom_scores : list of np.ndarray
            Per-atom score array for each configuration.
        struct_scores : np.ndarray
            Worst active-atom score per configuration.
        """
        all_atom_scores: list[np.ndarray] = []
        struct_scores_list: list[float] = []
        for at in _iter_configs(configs):
            a_sc, s_sc = self.score_atoms(
                at,
                mace_calc,
                atom_indices=atom_indices,
                exclude_fixed=exclude_fixed,
                calibrated=calibrated,
                delta=delta,
            )
            all_atom_scores.append(a_sc)
            struct_scores_list.append(s_sc)
        return all_atom_scores, np.asarray(struct_scores_list, dtype=np.float64)

    @staticmethod
    def _conformal_quantile(ratios: np.ndarray, delta: float) -> float:
        """Compute finite-sample split-conformal quantile at miscoverage ``delta``."""
        n = len(ratios)
        level = min(1.0, math.ceil((n + 1) * (1.0 - delta)) / n)
        return float(np.quantile(ratios, level, method="higher"))

    def calibrate(
        self,
        val_configs: str | Path | ConfigSet | Iterable[Atoms],
        mace_calc: MACECalculator | Any,
        force_calc: Calculator | None = None,
        ref_forces_key: str = "DFT_forces",
        pred_forces_key: str | None = "MACE_forces",
        atom_indices: slice | Sequence[int] | np.ndarray | None = None,
        exclude_fixed: bool = True,
        deltas: Sequence[float] = DEFAULT_CALIBRATION_DELTAS,
        tune_lambdas: Sequence[float] | None = DEFAULT_LAMBDA_GRID,
    ) -> dict[str, Any]:
        """Tune ``lambda_reg`` and fit multi-level split-conformal bounds in one pass.

        Parameters
        ----------
        val_configs : str or Path or ConfigSet or Iterable of Atoms
            Held-out validation configurations (``dft_test``) containing ``ref_forces_key``.
        mace_calc : MACECalculator
            Frozen MACE calculator used for descriptor extraction (and force evaluation
            if ``pred_forces_key`` is absent/``None`` and ``force_calc`` is ``None``).
        force_calc : Calculator or None, optional
            Optional calculator (e.g., MACE + D3 ``SumCalculator``) used for force
            prediction when ``pred_forces_key`` is ``None`` or absent on ``val_configs``.
        ref_forces_key : str, default="DFT_forces"
            ``atoms.arrays`` key for reference DFT forces.
        pred_forces_key : str or None, default="MACE_forces"
            ``atoms.arrays`` key for pre-stored MLIP forces. Pass ``None`` to ignore
            any pre-stored forces and force fresh evaluation with ``force_calc`` / ``mace_calc``.
        atom_indices : slice or Sequence of int or np.ndarray, optional
            Optional subset of active atom indices.
        exclude_fixed : bool, default=True
            If True, excludes fixed atoms during calibration.
        deltas : Sequence of float, default=(0.50, 0.20, 0.10)
            Miscoverage levels to fit simultaneously in one pass.
        tune_lambdas : Sequence of float or None, default=(1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
            Grid of ``lambda_reg`` values to sweep by atom-count-weighted mean
            per-element Spearman rank correlation. Pass ``None`` to keep current ``lambda_reg``.

        Returns
        -------
        dict of str to Any
            Structured diagnostics containing ``best_lambda``, ``per_element``
            (MAE, p90, MaxErr, Spearman rho, per-delta ``q_atom``), ``spearman_by_z``,
            ``calibrations``, ``coverage``, and validation error/bound arrays.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        ValueError
            If ``val_configs`` is empty or any ``delta`` is outside ``(0, 1)``.
        KeyError
            If ``ref_forces_key`` is missing on a validation configuration.
        """
        if not self.gram_matrices:
            raise RuntimeError(
                "PerElementLLPR must be fitted via fit() before calibrating."
            )

        delta_list = [float(d) for d in deltas]
        if not delta_list or any(not (0.0 < d < 1.0) for d in delta_list):
            raise ValueError(
                f"deltas must be non-empty values strictly in (0, 1), got {deltas!r}."
            )

        eval_calc = force_calc if force_calc is not None else mace_calc

        cached_descs: list[np.ndarray] = []
        cached_nums: list[np.ndarray] = []
        cached_atom_errs: list[np.ndarray] = []
        struct_max_errs_list: list[float] = []

        for at in _iter_configs(val_configs):
            if ref_forces_key not in at.arrays:
                raise KeyError(
                    f"Reference forces key {ref_forces_key!r} not found in atoms.arrays."
                )
            idx = self._get_active_indices(at, atom_indices, exclude_fixed)
            if len(idx) == 0:
                continue

            f_ref = np.asarray(at.arrays[ref_forces_key][idx], dtype=np.float64)
            if pred_forces_key is not None and pred_forces_key in at.arrays:
                f_pred = np.asarray(at.arrays[pred_forces_key][idx], dtype=np.float64)
            else:
                at_copy = at.copy()
                at_copy.calc = eval_calc
                f_pred = np.asarray(at_copy.get_forces()[idx], dtype=np.float64)

            atom_err = np.linalg.norm(f_ref - f_pred, axis=1)
            cached_descs.append(self._extract_descriptors(at, mace_calc)[idx])
            cached_nums.append(at.get_atomic_numbers()[idx])
            cached_atom_errs.append(atom_err)
            struct_max_errs_list.append(float(np.max(atom_err)))

        if not cached_descs:
            raise ValueError("val_configs yielded no active atoms for calibration.")

        all_atom_errs = np.concatenate(cached_atom_errs)
        all_nums = np.concatenate(cached_nums)
        unique_z = np.unique(all_nums)
        struct_max_errs = np.asarray(struct_max_errs_list, dtype=np.float64)

        # Tune shared lambda_reg by atom-count-weighted mean per-element Spearman rho
        best_lambda = self.lambda_reg
        best_rho = -np.inf
        lambda_grid = (
            list(tune_lambdas) if tune_lambdas is not None else [self.lambda_reg]
        )

        for lam in lambda_grid:
            self._factorize(lam)
            raw_flat = np.concatenate(
                [
                    self._score_feature_array(d, n)
                    for d, n in zip(cached_descs, cached_nums)
                ]
            )
            rho_sum = 0.0
            w_sum = 0
            for z in unique_z:
                mask = all_nums == z
                n_z = int(np.sum(mask))
                if n_z < 3:
                    continue
                r_z = float(spearmanr(raw_flat[mask], all_atom_errs[mask]).statistic)
                if np.isfinite(r_z):
                    rho_sum += n_z * r_z
                    w_sum += n_z
            score_rho = (
                rho_sum / w_sum
                if w_sum > 0
                else float(spearmanr(raw_flat, all_atom_errs).statistic)
            )
            if np.isfinite(score_rho) and score_rho > best_rho:
                best_rho = score_rho
                best_lambda = float(lam)

        # Finalize Cholesky factors at best_lambda and cache raw scores once
        self._factorize(best_lambda)
        raw_per_config = [
            self._score_feature_array(d, n) for d, n in zip(cached_descs, cached_nums)
        ]
        raw_atom_flat = np.concatenate(raw_per_config)

        spearman_by_z: dict[int, float] = {}
        for z in unique_z:
            z_int = int(z)
            mask = all_nums == z
            if int(np.sum(mask)) >= 3:
                spearman_by_z[z_int] = float(
                    spearmanr(raw_atom_flat[mask], all_atom_errs[mask]).statistic
                )

        # Fit split-conformal multipliers at every requested delta in one pass
        global_ratios = all_atom_errs / (raw_atom_flat + self.eps)
        self.calibrations = {}
        cal_diagnostics: dict[float, dict[str, Any]] = {}
        coverage_summary: dict[float, dict[str, float]] = {}
        val_atom_bounds_by_delta: dict[float, np.ndarray] = {}
        val_struct_bounds_by_delta: dict[float, np.ndarray] = {}

        for delta in delta_list:
            q_atom = self._conformal_quantile(global_ratios, delta)
            q_atom_by_z: dict[int, float] = {}
            for z in unique_z:
                z_int = int(z)
                mask = all_nums == z
                z_ratios = all_atom_errs[mask] / (raw_atom_flat[mask] + self.eps)
                q_atom_by_z[z_int] = self._conformal_quantile(z_ratios, delta)

            # Apply element-specific calibration BEFORE max-pooling across active atoms
            cal_per_config = [
                self._apply_atom_calibration(r, n, q_atom, q_atom_by_z)
                for r, n in zip(raw_per_config, cached_nums)
            ]
            val_atom_bounds = np.concatenate(cal_per_config)
            pre_struct_max = np.array(
                [float(np.max(c)) for c in cal_per_config], dtype=np.float64
            )

            struct_ratios = struct_max_errs / (pre_struct_max + self.eps)
            q_struct = self._conformal_quantile(struct_ratios, delta)
            val_struct_bounds = q_struct * pre_struct_max

            atom_cov = float(np.mean(all_atom_errs <= val_atom_bounds))
            struct_cov = float(np.mean(struct_max_errs <= val_struct_bounds))

            self.calibrations[delta] = {
                "q_atom": float(q_atom),
                "q_atom_by_z": dict(q_atom_by_z),
                "q_struct": float(q_struct),
            }
            cal_diagnostics[delta] = {
                "q_atom": float(q_atom),
                "q_atom_by_z": dict(q_atom_by_z),
                "q_struct": float(q_struct),
                "spearman_atom": float(
                    spearmanr(val_atom_bounds, all_atom_errs).statistic
                ),
                "spearman_struct": float(
                    spearmanr(val_struct_bounds, struct_max_errs).statistic
                ),
                "atom_coverage": atom_cov,
                "struct_coverage": struct_cov,
            }
            coverage_summary[delta] = {"atom": atom_cov, "struct": struct_cov}
            val_atom_bounds_by_delta[delta] = val_atom_bounds
            val_struct_bounds_by_delta[delta] = val_struct_bounds

        if self.default_delta not in self.calibrations:
            self.default_delta = delta_list[0]

        per_element: dict[int, dict[str, Any]] = {}
        for z in sorted(int(val) for val in unique_z):
            mask = all_nums == z
            e_z = all_atom_errs[mask]
            per_element[z] = {
                "n_atoms": len(e_z),
                "mae": float(np.mean(e_z)),
                "p90": float(np.quantile(e_z, 0.90)),
                "max_err": float(np.max(e_z)),
                "spearman": spearman_by_z.get(z),
                "q_atom": {
                    delta: float(self.calibrations[delta]["q_atom_by_z"][z])
                    for delta in delta_list
                },
            }
        self.element_metrics = per_element

        logger.info(
            "Calibrated PerElementLLPR on %d configs | best_lambda=%.1e | deltas=%s | coverage=%s",
            len(cached_descs),
            best_lambda,
            delta_list,
            {
                d: (round(v["atom"], 3), round(v["struct"], 3))
                for d, v in coverage_summary.items()
            },
        )

        return {
            "best_lambda": best_lambda,
            "per_element": per_element,
            "spearman_by_z": spearman_by_z,
            "calibrations": cal_diagnostics,
            "coverage": coverage_summary,
            "q_atom": self.q_atom,
            "q_atom_by_z": self.q_atom_by_z,
            "q_struct": self.q_struct,
            "val_atom_errors": all_atom_errs,
            "val_struct_errors": struct_max_errs,
            "val_atom_bounds": val_atom_bounds_by_delta,
            "val_struct_bounds": val_struct_bounds_by_delta,
        }

    def save(self, path: str | Path) -> None:
        """Save UQ state as pure tensors and Python primitives (``weights_only=True`` safe).

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"model_v0_llpr.pt"``).
        """
        pure_calibrations = {
            float(delta): {
                "q_atom": float(entry["q_atom"]),
                "q_atom_by_z": {
                    int(z): float(v) for z, v in entry["q_atom_by_z"].items()
                },
                "q_struct": float(entry["q_struct"]),
            }
            for delta, entry in self.calibrations.items()
        }
        pure_element_metrics = {
            int(z): {
                "n_atoms": int(stats["n_atoms"]),
                "mae": float(stats["mae"]),
                "p90": float(stats["p90"]),
                "max_err": float(stats["max_err"]),
                "spearman": (
                    None if stats.get("spearman") is None else float(stats["spearman"])
                ),
                "q_atom": {
                    float(d): float(q) for d, q in stats.get("q_atom", {}).items()
                },
            }
            for z, stats in self.element_metrics.items()
        }
        payload = {
            "lambda_reg": float(self.lambda_reg),
            "invariants_only": bool(self.invariants_only),
            "feature_dim": None if self.feature_dim is None else int(self.feature_dim),
            "default_delta": float(self.default_delta),
            "eps": float(self.eps),
            "atom_counts": {int(z): int(c) for z, c in self.atom_counts.items()},
            "calibrations": pure_calibrations,
            "element_metrics": pure_element_metrics,
            "gram_matrices": {
                int(z): g.detach().to(device="cpu", dtype=torch.float64)
                for z, g in self.gram_matrices.items()
            },
            "cholesky_factors": {
                int(z): chol.detach().to(device="cpu", dtype=torch.float64)
                for z, chol in self.cholesky_factors.items()
            },
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> PerElementLLPR:
        """Load a serialized ``PerElementLLPR`` using ``torch.load(..., weights_only=True)``.

        Parameters
        ----------
        path : str or Path
            Path to the saved ``.pt`` artifact.
        device : str, default="cpu"
            Target PyTorch device for restored Gram matrices and Cholesky factors.

        Returns
        -------
        PerElementLLPR
            Restored instance ready for deterministic scoring.
        """
        payload = torch.load(str(path), map_location=device, weights_only=True)
        obj = cls(
            lambda_reg=float(payload["lambda_reg"]),
            invariants_only=bool(payload["invariants_only"]),
            device=device,
            default_delta=float(payload.get("default_delta", 0.20)),
            eps=float(payload.get("eps", 1e-12)),
        )
        obj.feature_dim = payload["feature_dim"]
        obj.atom_counts = {int(z): int(c) for z, c in payload["atom_counts"].items()}
        obj.calibrations = {
            float(delta): {
                "q_atom": float(entry["q_atom"]),
                "q_atom_by_z": {
                    int(z): float(v) for z, v in entry["q_atom_by_z"].items()
                },
                "q_struct": float(entry["q_struct"]),
            }
            for delta, entry in payload.get("calibrations", {}).items()
        }
        obj.element_metrics = {
            int(z): {
                "n_atoms": int(stats["n_atoms"]),
                "mae": float(stats["mae"]),
                "p90": float(stats["p90"]),
                "max_err": float(stats["max_err"]),
                "spearman": (
                    None if stats.get("spearman") is None else float(stats["spearman"])
                ),
                "q_atom": {
                    float(d): float(q) for d, q in stats.get("q_atom", {}).items()
                },
            }
            for z, stats in payload.get("element_metrics", {}).items()
        }
        obj.gram_matrices = {
            int(z): g.to(device=device, dtype=torch.float64)
            for z, g in payload["gram_matrices"].items()
        }
        obj.cholesky_factors = {
            int(z): chol.to(device=device, dtype=torch.float64)
            for z, chol in payload["cholesky_factors"].items()
        }
        return obj
