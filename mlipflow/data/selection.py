from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ase.io import write
from mace.calculators import MACECalculator
from wfl.configset import ConfigSet, OutputSpec

from mlipflow.data import setup_logging
from mlipflow.data.llpr import PerElementLLPR
from mlipflow.data.processing import check_maxforce_and_cleanarrays

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

    from mlipflow.strategies.mlip import MACEModel

setup_logging()
logger = logging.getLogger(__name__)

_VALID_SPLIT_MODES = ("stratified_random", "random")
_VALID_ERROR_METRICS = ("mae", "p90", "max_err")


def _random_split_indices(
    n_total: int, test_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint ``(train_idx, test_idx)`` arrays via unbiased random permutation."""
    if n_total == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty
    if n_total == 1:
        return np.array([0], dtype=int), np.empty(0, dtype=int)

    n_test = max(1, min(n_total - 1, round(n_total * test_fraction)))
    perm = rng.permutation(n_total)
    return perm[n_test:], perm[:n_test]


def _stratified_random_split_indices(
    configs: list[Atoms],
    test_fraction: float,
    rng: np.random.Generator,
    stratify_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint ``(train_idx, test_idx)`` arrays stratified by ``stratify_key``."""
    n_total = len(configs)
    if n_total <= 1:
        return _random_split_indices(n_total, test_fraction, rng)

    groups: dict[str | None, list[int]] = {}
    for idx, at in enumerate(configs):
        raw_val = at.info.get(stratify_key)
        key = None if raw_val is None else str(raw_val)
        groups.setdefault(key, []).append(idx)

    # Fall back to unstratified random split when stratification is trivial
    if len(groups) <= 1 or all(len(indices) <= 1 for indices in groups.values()):
        logger.debug(
            "Stratification key '%s' has <= 1 multi-frame group; falling back to random split.",
            stratify_key,
        )
        return _random_split_indices(n_total, test_fraction, rng)

    sorted_keys = sorted(
        groups.keys(), key=lambda k: (k is None, "" if k is None else k)
    )
    target_test = max(1, min(n_total - 1, round(n_total * test_fraction)))

    alloc: dict[str | None, int] = {}
    remainders: list[tuple[float, str | None]] = []
    for k in sorted_keys:
        n_g = len(groups[k])
        if n_g <= 1:
            alloc[k] = 0
            continue
        exact = n_g * test_fraction
        base = min(int(np.floor(exact)), n_g - 1)
        alloc[k] = base
        if base < n_g - 1:
            remainders.append((exact - base, k))

    deficit = target_test - sum(alloc.values())
    if deficit > 0:
        remainders.sort(key=lambda item: -item[0])
        for _, k in remainders:
            if deficit <= 0:
                break
            if alloc[k] < len(groups[k]) - 1:
                alloc[k] += 1
                deficit -= 1
    elif deficit < 0:
        for k in sorted_keys:
            if deficit >= 0:
                break
            if alloc[k] > 0:
                alloc[k] -= 1
                deficit += 1

    train_idx: list[int] = []
    test_idx: list[int] = []
    for k in sorted_keys:
        perm = rng.permutation(groups[k])
        n_g_test = alloc[k]
        test_idx.extend(perm[:n_g_test].tolist())
        train_idx.extend(perm[n_g_test:].tolist())

    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def split_configset(
    in_file: str | list[Atoms] | ConfigSet,
    out_file: str | dict[str, str],
    main_suffix: str = "train",
    side_suffix: str = "test",
    split_mode: Literal["stratified_random", "random"] = "stratified_random",
    test_fraction: float = 0.2,
    seed: int | None = 42,
    stratify_key: str = "species",
    clean_data: bool = True,
    max_force: float = 12.0,
    mlip_prefix: str = "MACE",
    calc: str = "opt",
) -> None:
    """Split a ConfigSet into unbiased training and held-out test sets.

    Supports species-stratified random splitting (default) and unstratified
    random splitting with a deterministic random seed, preserving the full
    force-error distribution in the held-out test set for unbiased accuracy
    evaluation and split-conformal UQ calibration.

    Parameters
    ----------
    in_file : str or list of Atoms or ConfigSet
        Input file or atomic configurations.
    out_file : str or dict of str to str
        Base filename for output files (written as ``f"{suffix}_{out_file}"``)
        or a mapping ``{main_suffix: train_path, side_suffix: test_path}``.
    main_suffix : str, default="train"
        Key/prefix for the training split output file.
    side_suffix : str, default="test"
        Key/prefix for the held-out test split output file.
    split_mode : {"stratified_random", "random"}, default="stratified_random"
        Splitting strategy. ``"stratified_random"`` stratifies by ``stratify_key``
        and falls back cleanly to ``"random"`` if the key is absent.
    test_fraction : float, default=0.2
        Fraction of configurations assigned to the held-out test split; must lie
        strictly in ``(0.0, 1.0)``.
    seed : int or None, default=42
        Random seed for deterministic splitting.
    stratify_key : str, default="species"
        ``atoms.info`` key used for stratification when ``split_mode="stratified_random"``.
    clean_data : bool, default=True
        If True, runs ``check_maxforce_and_cleanarrays`` prior to splitting.
    max_force : float, default=12.0
        Maximum force threshold for filtering configurations when ``clean_data`` is True.
    mlip_prefix : str, default="MACE"
        MLIP prefix used for cleaning arrays when ``clean_data`` is True.
    calc : str, default="opt"
        Calculation context used for key cleaning when ``clean_data`` is True.

    Raises
    ------
    ValueError
        If ``split_mode`` is not in ``("stratified_random", "random")`` or
        ``test_fraction`` is not strictly between 0 and 1.
    """
    if split_mode not in _VALID_SPLIT_MODES:
        raise ValueError(
            f"Invalid split_mode={split_mode!r}. Expected one of {_VALID_SPLIT_MODES}."
        )
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(
            f"test_fraction must be strictly between 0 and 1, got {test_fraction!r}."
        )

    if clean_data:
        configs = check_maxforce_and_cleanarrays(
            in_file=in_file,
            out_file=None,
            mlip_prefix=mlip_prefix,
            calc=calc,
            max_force=max_force,
        )
    else:
        configs = list(ConfigSet(in_file))

    rng = np.random.default_rng(seed)
    if split_mode == "stratified_random":
        main_chunk, side_chunk = _stratified_random_split_indices(
            configs=configs,
            test_fraction=test_fraction,
            rng=rng,
            stratify_key=stratify_key,
        )
    else:
        main_chunk, side_chunk = _random_split_indices(
            n_total=len(configs), test_fraction=test_fraction, rng=rng
        )

    main_at = [configs[idx] for idx in main_chunk]
    side_at = [configs[idx] for idx in side_chunk]

    for atoms, suffix in zip([main_at, side_at], [main_suffix, side_suffix]):
        if isinstance(out_file, dict):
            write(out_file[suffix], atoms)
        else:
            write(f"{suffix}_{out_file}", atoms)


def _element_exceeds_target(
    element_metrics: dict[int, dict[str, Any]] | None,
    target: float | dict[int, float],
    metric: str = "p90",
) -> bool:
    """Return True if at least one element in ``element_metrics`` exceeds ``target``."""
    if metric not in _VALID_ERROR_METRICS:
        raise ValueError(
            f"Invalid element_error_metric={metric!r}. Expected one of {_VALID_ERROR_METRICS}."
        )
    if not element_metrics:
        return False

    for z, stats in element_metrics.items():
        if not isinstance(stats, dict):
            continue
        val = stats.get(metric)
        if val is None or not math.isfinite(float(val)):
            continue
        if isinstance(target, dict):
            t_z = target.get(z, target.get(int(z)))
            if t_z is None:
                continue
            threshold = float(t_z)
        else:
            threshold = float(target)
        if float(val) > threshold:
            return True
    return False


def _load_pool_configs(
    pool_file: str | Path | Sequence[str | Path] | ConfigSet | Sequence[Atoms],
) -> list[Atoms]:
    """Load pool configurations as fresh ``Atoms`` copies from files, ConfigSet, or sequence."""
    if isinstance(pool_file, ConfigSet):
        return [at.copy() for at in pool_file]
    if isinstance(pool_file, (str, Path)):
        return [at.copy() for at in ConfigSet(str(pool_file))]
    pool_list = list(pool_file)
    if not pool_list:
        return []
    if isinstance(pool_list[0], (str, Path)):
        return [at.copy() for at in ConfigSet([str(p) for p in pool_list])]
    return [at.copy() for at in pool_list]


def _can_use_inline_uq(
    pool_configs: Sequence[Atoms],
    max_uq_key: str,
    uq_delta_key: str,
    uq_model_key: str,
    delta: float,
    resolved_uq_name: str,
) -> bool:
    """Validate whether pre-populated inline UQ scores can safely pre-filter the pool."""
    if not all(max_uq_key in at.info for at in pool_configs):
        return False

    for at in pool_configs:
        if uq_delta_key in at.info:
            stored_delta = float(at.info[uq_delta_key])
            if not math.isclose(stored_delta, float(delta), rel_tol=0.0, abs_tol=1e-9):
                logger.warning(
                    "Inline UQ provenance mismatch on %s (expected delta=%s, found %s); "
                    "falling back to full-pool scoring.",
                    uq_delta_key,
                    delta,
                    stored_delta,
                )
                return False
        if uq_model_key in at.info:
            stored_model = str(at.info[uq_model_key])
            if stored_model != resolved_uq_name:
                logger.warning(
                    "Inline UQ provenance mismatch on %s (expected %r, found %r); "
                    "falling back to full-pool scoring.",
                    uq_model_key,
                    resolved_uq_name,
                    stored_model,
                )
                return False
    return True


def _write_selected_configs(
    out_file: str | Path, selected_configs: list[Atoms]
) -> None:
    """Write selected configurations (or an empty 0-byte file if none were selected)."""
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_configs:
        OutputSpec(str(out_path), overwrite=True).write(ConfigSet(selected_configs))
    else:
        out_path.write_text("", encoding="utf-8")


def select_by_uncertainty(
    pool_file: str | Path | Sequence[str | Path] | ConfigSet | Sequence[Atoms],
    out_file: str | Path,
    mlip_strategy: MACEModel,
    uq_file: str | Path | None = None,
    force_tol_eVA: float = 0.10,
    min_select_count: int = 0,
    atom_indices: slice | Sequence[int] | np.ndarray | None = None,
    exclude_fixed: bool = True,
    delta: float = 0.50,
    device: str = "cpu",
    dtype: str = "float32",
    element_metrics: dict[int, dict[str, Any]] | None = None,
    element_error_target: float | dict[int, float] | None = None,
    element_error_metric: str = "p90",
    use_inline_uq: bool = False,
) -> list[Atoms]:
    """Select non-redundant uncertain structures via Sequential Batch LLPR.

    Loads a calibrated :class:`PerElementLLPR` artifact, identifies candidate
    structures whose calibrated worst-atom force-error bound exceeds
    ``force_tol_eVA`` (at miscoverage ``delta``, default ``0.50``), and applies
    greedy per-element Gram/Cholesky updates on an in-memory copy of the Gram
    matrices to suppress near-duplicate local atomic environments without an
    arbitrary per-iteration count cap.

    Parameters
    ----------
    pool_file : str or Path or Sequence or ConfigSet
        Pool configurations (file path, list of paths, ``ConfigSet``, or ``Atoms``).
    out_file : str or Path
        Output XYZ file path for selected configurations.
    mlip_strategy : MACEModel
        MLIP strategy providing ``model_file``, ``mlip_prefix``, and default ``uq_file``.
    uq_file : str or Path or None, optional
        Path to the serialized ``PerElementLLPR`` ``.pt`` artifact. Defaults to
        ``mlip_strategy.uq_file``.
    force_tol_eVA : float, default=0.10
        Physical worst-atom force-error tolerance in eV/A.
    min_select_count : int, default=0
        Optional greedy top-K fallback count when zero structures exceed
        ``force_tol_eVA``, activated only if at least one element in
        ``element_metrics`` exceeds ``element_error_target``.
    atom_indices : slice or Sequence of int or np.ndarray, optional
        Optional subset of active atom indices to score.
    exclude_fixed : bool, default=True
        If True, excludes ASE ``FixAtoms`` constrained atoms from scoring.
    delta : float, default=0.50
        Split-conformal miscoverage level for calibrated bounds (``0.50`` = expected error).
    device : str, default="cpu"
        PyTorch device for MACE descriptor extraction and LLPR solves.
    dtype : str, default="float32"
        Default floating-point precision for ``MACECalculator``.
    element_metrics : dict of int to dict, optional
        Per-element validation metrics (``mae``, ``p90``, ``max_err``) keyed by
        atomic number ``Z``. If ``None``, falls back to ``llpr.element_metrics``
        stored in the ``.pt`` artifact.
    element_error_target : float or dict of int to float or None, optional
        Target element error in eV/A gating the ``min_select_count`` fallback.
        Defaults to ``force_tol_eVA`` when ``None``.
    element_error_metric : {"mae", "p90", "max_err"}, default="p90"
        Per-element metric key compared against ``element_error_target``.
    use_inline_uq : bool, default=False
        If True and ``{prefix}max_uq`` is already populated on every pool frame
        (with matching provenance tags if present), pre-filter frames below
        ``force_tol_eVA`` before extracting MACE descriptors only for passing
        structures.

    Returns
    -------
    list of Atoms
        Selected non-redundant configurations (also written to ``out_file``).

    Raises
    ------
    FileNotFoundError
        If the resolved ``uq_file`` does not exist on disk.
    ValueError
        If ``element_error_metric`` is invalid.
    """
    if element_error_metric not in _VALID_ERROR_METRICS:
        raise ValueError(
            f"Invalid element_error_metric={element_error_metric!r}. "
            f"Expected one of {_VALID_ERROR_METRICS}."
        )

    resolved_uq = (
        uq_file if uq_file is not None else getattr(mlip_strategy, "uq_file", None)
    )
    if resolved_uq is None or not Path(resolved_uq).exists():
        raise FileNotFoundError(
            f"Calibrated LLPR artifact not found: {resolved_uq!r}. "
            "Run post-fit LLPR calibration before uncertainty selection."
        )

    llpr = PerElementLLPR.load(resolved_uq, device=device)
    effective_metrics = (
        element_metrics
        if element_metrics is not None
        else (llpr.element_metrics or None)
    )

    pool_configs = _load_pool_configs(pool_file)
    if not pool_configs:
        logger.info("Pool is empty; writing empty selection output to %s.", out_file)
        _write_selected_configs(out_file, [])
        return []

    prefix = getattr(mlip_strategy, "mlip_prefix", "MACE_")
    if not isinstance(prefix, str):
        prefix = "MACE_"
    atom_uq_key = f"{prefix}atom_uq"
    max_uq_key = f"{prefix}max_uq"
    uq_delta_key = f"{prefix}uq_delta"
    uq_model_key = f"{prefix}uq_model"
    resolved_uq_name = Path(resolved_uq).name

    cal = llpr.get_calibration(delta)
    q_atom = float(cal["q_atom"])
    q_atom_by_z: dict[int, float] = dict(cal["q_atom_by_z"])
    q_struct = float(cal["q_struct"])

    mace_calc_holder: list[Any] = []

    def _get_mace_calc() -> Any:
        if not mace_calc_holder:
            mace_calc_holder.append(
                MACECalculator(
                    model_paths=str(Path(mlip_strategy.model_file).resolve()),
                    device=device,
                    default_dtype=dtype,
                )
            )
        return mace_calc_holder[0]

    def _extract_and_score(
        indices: Sequence[int],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        mace_calc = _get_mace_calc()
        descs_list: list[np.ndarray] = []
        nums_list: list[np.ndarray] = []
        struct_scores: list[float] = []
        for idx in indices:
            at = pool_configs[idx]
            active_idx = llpr._get_active_indices(at, atom_indices, exclude_fixed)
            full_atom_uq = np.zeros(len(at), dtype=np.float64)
            if len(active_idx) == 0:
                descs_list.append(
                    np.empty((0, llpr.feature_dim or 0), dtype=np.float64)
                )
                nums_list.append(np.empty(0, dtype=int))
                struct_uq = 0.0
            else:
                desc = llpr._extract_descriptors(at, mace_calc)[active_idx]
                nums = at.get_atomic_numbers()[active_idx]
                raw = llpr._score_feature_array(desc, nums)
                cal_atom = llpr._apply_atom_calibration(raw, nums, q_atom, q_atom_by_z)
                full_atom_uq[active_idx] = cal_atom
                struct_uq = q_struct * float(np.max(cal_atom))
                descs_list.append(desc)
                nums_list.append(nums)
            at.arrays[atom_uq_key] = full_atom_uq
            at.info[max_uq_key] = float(struct_uq)
            struct_scores.append(float(struct_uq))
        return descs_list, nums_list, struct_scores

    inline_fast_path = use_inline_uq and _can_use_inline_uq(
        pool_configs,
        max_uq_key=max_uq_key,
        uq_delta_key=uq_delta_key,
        uq_model_key=uq_model_key,
        delta=delta,
        resolved_uq_name=resolved_uq_name,
    )

    all_descs: list[np.ndarray] | None = None
    all_nums: list[np.ndarray] | None = None
    selected_indices: list[int] = []

    if inline_fast_path:
        passing_indices = [
            i
            for i, at in enumerate(pool_configs)
            if float(at.info[max_uq_key]) > force_tol_eVA
        ]
        if passing_indices:
            pass_descs, pass_nums, _ = _extract_and_score(passing_indices)
            rel_selected = llpr.select_sequential_batch(
                candidate_descs=pass_descs,
                candidate_numbers=pass_nums,
                force_tol_eVA=force_tol_eVA,
                delta=delta,
            )
            selected_indices = [passing_indices[r] for r in rel_selected]
    else:
        all_indices = list(range(len(pool_configs)))
        all_descs, all_nums, init_scores = _extract_and_score(all_indices)
        passing_indices = [
            i for i, score in enumerate(init_scores) if score > force_tol_eVA
        ]
        if passing_indices:
            rel_selected = llpr.select_sequential_batch(
                candidate_descs=[all_descs[i] for i in passing_indices],
                candidate_numbers=[all_nums[i] for i in passing_indices],
                force_tol_eVA=force_tol_eVA,
                delta=delta,
            )
            selected_indices = [passing_indices[r] for r in rel_selected]

    # Optional diversity-preserving fallback when n_passing == 0
    if not selected_indices and min_select_count > 0:
        target = force_tol_eVA if element_error_target is None else element_error_target
        if not effective_metrics:
            logger.warning(
                "min_select_count=%d requested with n_passing=0, but element_metrics "
                "is unavailable in state and %s; suppressing fallback and returning 0 structures.",
                min_select_count,
                resolved_uq,
            )
        elif _element_exceeds_target(effective_metrics, target, element_error_metric):
            if all_descs is None or all_nums is None:
                all_descs, all_nums, _ = _extract_and_score(
                    list(range(len(pool_configs)))
                )
            selected_indices = llpr.select_sequential_batch(
                candidate_descs=all_descs,
                candidate_numbers=all_nums,
                force_tol_eVA=force_tol_eVA,
                delta=delta,
                top_k=min_select_count,
            )

    selected_configs = [pool_configs[i] for i in selected_indices]
    _write_selected_configs(out_file, selected_configs)

    logger.info(
        "Sequential Batch LLPR selected %d / %d structures (%d passed tau_F=%.3f eV/A at delta=%.2f) -> %s",
        len(selected_configs),
        len(pool_configs),
        len(passing_indices),
        force_tol_eVA,
        delta,
        out_file,
    )
    return selected_configs
