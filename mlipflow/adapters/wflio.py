from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import MACECalculator
from quippy.potential import Potential
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from wfl.calculators.wfl_fileio_calculator import WFLFileIOCalculator
from wfl.generate.md.abort_base import AbortSimBase

from mlipflow.data.llpr import PerElementLLPR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

logger = logging.getLogger(__name__)

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
_default_keep_files = ["*.out"]
_default_properties = ["energy", "forces"]  # , "stress"


# calculator classes for remote jobs
class GAPCalc(WFLFileIOCalculator, Potential):
    """
    Calculator class for GAP potentials that integrates with WFL workflows.
    """

    def __init__(
        self,
        keep_files: str = "default",
        rundir_prefix: str = "run_GAP_",
        workdir: str | None = None,
        scratchdir: str | None = None,
        **kwargs,
    ):
        """
        Initialise the GAPCalc.

        Args:
            keep_files (str): Files to keep. Defaults to 'default'.
            rundir_prefix (str): Prefix for the run directory. Defaults to 'run_GAP_'.
            workdir (str | None): Working directory. Defaults to None.
            scratchdir (str | None): Scratch directory. Defaults to None.
            **kwargs: Additional arguments for the calculator.
        """

        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors
        super().__init__(
            keep_files=keep_files,
            rundir_prefix=rundir_prefix,
            workdir=workdir,
            scratchdir=scratchdir,
            **kwargs,
        )

    def calculate(
        self,
        atoms: object = None,
        properties: list[str] = _default_properties,
        system_changes: list[str] = all_changes,
    ) -> None:
        """
        Perform the calculation.

        Args:
            atoms (ase.Atoms): Atoms object.
            properties (list[str]): List of properties to calculate.
            system_changes (list[str]): List of changes in the system.
        """
        # from WFLFileIOCalculator
        self.setup_rundir()

        try:
            super().calculate(
                atoms=atoms, properties=properties, system_changes=system_changes
            )
            if "FAILED_GAP" in atoms.info:
                del atoms.info["FAILED_GAP"]
        except Exception as exc:
            atoms.info["FAILED_GAP"] = True
            raise exc


#        finally:
#            # from WFLFileIOCalculator
#            self.clean_rundir(_default_keep_files, calculation_succeeded)


class MACECalc(WFLFileIOCalculator, MACECalculator):
    """
    Calculator class for MACE potentials that integrates with WFL workflows
    """

    def __init__(
        self,
        keep_files: str = "default",
        rundir_prefix: str = "run_MACE_",
        workdir: str | None = None,
        scratchdir: str | None = None,
        **kwargs,
    ):
        """
        Initialise the MACECalc.

        Args:
            keep_files (str): Files to keep. Defaults to 'default'.
            rundir_prefix (str): Prefix for the run directory. Defaults to 'run_MACE_'.
            workdir (str | None): Working directory. Defaults to None.
            scratchdir (str | None): Scratch directory. Defaults to None.
            **kwargs: Additional arguments for the calculator.
        """

        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors
        super().__init__(
            keep_files=keep_files,
            rundir_prefix=rundir_prefix,
            workdir=workdir,
            scratchdir=scratchdir,
            **kwargs,
        )

    def calculate(
        self,
        atoms: object = None,
        properties: list[str] = _default_properties,
        system_changes: list[str] = all_changes,
    ) -> None:
        """
        Perform the calculation.

        Args:
            atoms (ase.Atoms): Atoms object.
            properties (list[str]): List of properties to calculate.
            system_changes (list[str]): List of changes in the system.
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        # from WFLFileIOCalculator
        self.setup_rundir()

        try:
            super().calculate(
                atoms=atoms, properties=properties, system_changes=system_changes
            )
            calculation_succeeded = True
            if "FAILED_MACE" in atoms.info:
                del atoms.info["FAILED_MACE"]
        except Exception as exc:
            atoms.info["FAILED_MACE"] = True
            calculation_succeeded = False
            raise exc
        finally:
            # from WFLFileIOCalculator
            self.clean_rundir(_default_keep_files, calculation_succeeded)


class MACELLPRCalculator(Calculator):
    """Pickle-safe ASE calculator combining MACE, optional DFT-D3, and inline LLPR UQ.

    Initialized purely from serializable parameters so ``pickle`` and ``wfl``'s
    ``construct_calculator_picklesafe`` work seamlessly across local and remote
    ``expyre`` workers. Live PyTorch calculators and ``PerElementLLPR`` state are
    constructed lazily on first use and dropped on serialization.

    When ``compute_uq=True`` and ``uq_file`` is provided, computes calibrated
    per-atom and worst-atom force-error bounds in a single descriptor pass via
    :meth:`PerElementLLPR.score_atoms_multi_delta` and attaches:

    - ``atoms.arrays[f"{prefix}atom_uq"]``: per-atom bound at ``primary_delta``
    - ``atoms.info[f"{prefix}max_uq"]``: worst-atom bound at ``primary_delta``
    - ``atoms.info[f"{prefix}max_uq_p90"]``: worst-atom bound at ``p90_delta`` (when calibrated)
    - ``atoms.info[f"{prefix}uq_delta"]``: ``float(primary_delta)`` provenance tag
    - ``atoms.info[f"{prefix}uq_model"]``: ``Path(uq_file).name`` provenance tag
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(
        self,
        model_paths: str | Path,
        uq_file: str | Path | None = None,
        dispersion: bool = True,
        dtype: str = "float32",
        device: str = "cpu",
        atom_indices: slice | Sequence[int] | np.ndarray | None = None,
        exclude_fixed: bool = True,
        prefix: str = "MACE_",
        primary_delta: float = 0.50,
        p90_delta: float | None = 0.10,
        compute_uq: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise the pickle-safe ``MACELLPRCalculator``.

        Parameters
        ----------
        model_paths : str or Path
            Path to the MACE ``.model`` file.
        uq_file : str or Path or None, optional
            Path to the serialized ``PerElementLLPR`` ``.pt`` artifact.
        dispersion : bool, default=True
            If True, adds BJ-damped ``TorchDFTD3Calculator`` energy/forces/stress.
        dtype : str, default="float32"
            Floating-point precision (``"float32"`` or ``"float64"``).
        device : str, default="cpu"
            PyTorch device for MACE, DFT-D3, and LLPR evaluation.
        atom_indices : slice or Sequence of int or np.ndarray, optional
            Optional subset of active atom indices for LLPR scoring.
        exclude_fixed : bool, default=True
            If True, excludes ASE ``FixAtoms`` constrained atoms from LLPR scoring.
        prefix : str, default="MACE_"
            Prefix for ``atoms.arrays`` and ``atoms.info`` UQ keys.
        primary_delta : float, default=0.50
            Miscoverage level for ``{prefix}atom_uq`` and ``{prefix}max_uq``.
        p90_delta : float or None, default=0.10
            Optional secondary miscoverage level for ``{prefix}max_uq_p90``.
        compute_uq : bool, default=True
            If True (and ``uq_file`` is provided), evaluate and attach inline LLPR UQ.
        **kwargs : Any
            Additional keyword arguments forwarded to ``ase.calculators.calculator.Calculator``.
        """
        super().__init__(**kwargs)
        self.model_paths = str(model_paths)
        self.uq_file = None if uq_file is None else str(uq_file)
        self.dispersion = bool(dispersion)
        self.dtype = str(dtype)
        self.device = str(device)
        self.atom_indices = atom_indices
        self.exclude_fixed = bool(exclude_fixed)
        self.prefix = str(prefix)
        self.primary_delta = float(primary_delta)
        self.p90_delta = None if p90_delta is None else float(p90_delta)
        self.compute_uq = bool(compute_uq)

        self._mace_calc: MACECalculator | None = None
        self._dispersion_calc: TorchDFTD3Calculator | None = None
        self._llpr: PerElementLLPR | None = None
        self._cached_uq_payload: dict[str, Any] | None = None
        self._warned_missing_primary = False

    def __getstate__(self) -> dict[str, Any]:
        """Drop live PyTorch/LLPR objects and cached results before pickling."""
        state = self.__dict__.copy()
        state["_mace_calc"] = None
        state["_dispersion_calc"] = None
        state["_llpr"] = None
        state["_cached_uq_payload"] = None
        state["atoms"] = None
        state["results"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore serializable configuration with lazy handles uninitialized."""
        self.__dict__.update(state)
        self._mace_calc = None
        self._dispersion_calc = None
        self._llpr = None
        self._cached_uq_payload = None

    def _ensure_initialized(self) -> None:
        """Lazily construct internal MACE, optional D3, and LLPR objects."""
        if self._mace_calc is None:
            self._mace_calc = MACECalculator(
                model_paths=self.model_paths,
                device=self.device,
                default_dtype=self.dtype,
            )
        if self.dispersion and self._dispersion_calc is None:
            torch_dtype = torch.float32 if self.dtype == "float32" else torch.float64
            self._dispersion_calc = TorchDFTD3Calculator(
                damping="bj", old=False, device=self.device, dtype=torch_dtype
            )
        if self.compute_uq and self.uq_file is not None and self._llpr is None:
            self._llpr = PerElementLLPR.load(self.uq_file, device=self.device)

    def get_descriptors(
        self, atoms: Atoms | None = None, invariants_only: bool = True
    ) -> np.ndarray:
        """Expose underlying MACE per-atom descriptors without unwrapping."""
        self._ensure_initialized()
        target = atoms if atoms is not None else self.atoms
        return self._mace_calc.get_descriptors(target, invariants_only=invariants_only)

    def _attach_cached_uq(self, target: Atoms) -> None:
        """Attach cached UQ arrays and info metadata onto ``target`` without aliasing."""
        if self._cached_uq_payload is None:
            return
        atom_uq = self._cached_uq_payload.get("atom_uq")
        if atom_uq is not None:
            target.arrays[f"{self.prefix}atom_uq"] = atom_uq.copy()
        for key, val in self._cached_uq_payload.get("info", {}).items():
            target.info[key] = val

    def get_property(
        self, name: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
        """Return property and ensure cached UQ annotations are attached on cache hits."""
        result = super().get_property(
            name, atoms=atoms, allow_calculation=allow_calculation
        )
        target = atoms if atoms is not None else self.atoms
        if target is not None and self.compute_uq and self.uq_file is not None:
            self._attach_cached_uq(target)
        return result

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = _default_properties,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Evaluate MACE (+ optional D3) and optional single-pass multi-delta LLPR UQ."""
        super().calculate(atoms, list(properties), system_changes)
        self._ensure_initialized()

        if system_changes:
            self.results = {}
            self._cached_uq_payload = None

        eval_props = list(properties)
        if "forces" in eval_props and "energy" not in eval_props:
            eval_props = ["energy", *eval_props]

        missing_props = (
            eval_props
            if system_changes
            else [p for p in eval_props if p not in self.results]
        )

        # Always pass a detached copy to internal sub-calculators so atoms.calc is untouched
        work_atoms = self.atoms.copy()

        if missing_props:
            self._mace_calc.calculate(work_atoms, missing_props, all_changes)
            for prop in missing_props:
                if (
                    prop in self._mace_calc.results
                    and self._mace_calc.results[prop] is not None
                ):
                    val = self._mace_calc.results[prop]
                    self.results[prop] = (
                        val.copy() if isinstance(val, np.ndarray) else val
                    )

            if self._dispersion_calc is not None:
                self._dispersion_calc.calculate(work_atoms, missing_props, all_changes)
                for prop in missing_props:
                    if (
                        prop in self._dispersion_calc.results
                        and self._dispersion_calc.results[prop] is not None
                        and prop in self.results
                        and self.results[prop] is not None
                    ):
                        self.results[prop] = (
                            self.results[prop] + self._dispersion_calc.results[prop]
                        )

        if self.compute_uq and self._llpr is not None and self.uq_file is not None:
            if self._cached_uq_payload is None:
                deltas_to_score = [self.primary_delta]
                if self.p90_delta is not None and not math.isclose(
                    self.p90_delta, self.primary_delta, rel_tol=0.0, abs_tol=1e-9
                ):
                    deltas_to_score.append(self.p90_delta)

                scored = self._llpr.score_atoms_multi_delta(
                    work_atoms,
                    self._mace_calc,
                    deltas=deltas_to_score,
                    atom_indices=self.atom_indices,
                    exclude_fixed=self.exclude_fixed,
                )

                primary_res = scored.get(self.primary_delta)
                p90_res = (
                    scored.get(self.p90_delta) if self.p90_delta is not None else None
                )

                info_payload: dict[str, Any] = {}
                atom_uq_arr: np.ndarray | None = None

                if primary_res is not None:
                    atom_uq_arr, max_uq = primary_res
                    self.results["atom_uq"] = atom_uq_arr
                    self.results["max_uq"] = float(max_uq)
                    info_payload[f"{self.prefix}max_uq"] = float(max_uq)
                    info_payload[f"{self.prefix}uq_delta"] = float(self.primary_delta)
                    info_payload[f"{self.prefix}uq_model"] = Path(self.uq_file).name
                elif not self._warned_missing_primary:
                    logger.warning(
                        "Primary calibration delta=%s not found in %s; skipping primary inline UQ.",
                        self.primary_delta,
                        self.uq_file,
                    )
                    self._warned_missing_primary = True

                if p90_res is not None:
                    _, max_uq_p90 = p90_res
                    self.results["max_uq_p90"] = float(max_uq_p90)
                    info_payload[f"{self.prefix}max_uq_p90"] = float(max_uq_p90)

                if atom_uq_arr is not None or info_payload:
                    self._cached_uq_payload = {
                        "atom_uq": atom_uq_arr,
                        "info": info_payload,
                    }

            if atoms is not None:
                self._attach_cached_uq(atoms)


###############################################################################################################


# abort criteria class for MD-simulations
class ForceCheck(AbortSimBase):
    """Abort criteria class for MD simulations based on force magnitude and optional LLPR UQ."""

    def __init__(
        self,
        threshold: float,
        n_failed_steps: int = 10,
        uq_threshold: float | None = None,
        uq_key: str = "MACE_max_uq_p90",
    ):
        """Initialise the ForceCheck abort criteria.

        Parameters
        ----------
        threshold : float
            Maximum allowed absolute force component (eV/A).
        n_failed_steps : int, default=10
            Number of consecutive failed steps required to trigger MD abort.
        uq_threshold : float or None, optional
            Optional maximum allowed calibrated worst-atom uncertainty bound (eV/A).
            When ``None``, preserves legacy raw-force-only abort behavior.
        uq_key : str, default="MACE_max_uq_p90"
            ``atoms.info`` key read for uncertainty gating when ``uq_threshold`` is set.
        """
        super().__init__(n_failed_steps)
        self.threshold = threshold
        self.uq_threshold = uq_threshold
        self.uq_key = uq_key

    def atoms_ok(self, at: object) -> bool:
        """Check whether forces and optional LLPR uncertainty are within thresholds.

        Parameters
        ----------
        at : ase.Atoms
            Current MD configuration.

        Returns
        -------
        bool
            True if within force (and optional UQ) thresholds, False otherwise.
        """
        if not bool(self.threshold > np.max(np.abs(at.calc.get_forces()))):
            return False

        if self.uq_threshold is not None:
            uq_val = at.info.get(self.uq_key)
            if uq_val is None and at.calc is not None:
                results = getattr(at.calc, "results", None) or {}
                fallback_key = (
                    "max_uq" if self.uq_key.endswith("max_uq") else "max_uq_p90"
                )
                uq_val = results.get(fallback_key)
            if uq_val is not None:
                uq_float = float(uq_val)
                if not np.isfinite(uq_float) or uq_float > self.uq_threshold:
                    return False

        return True


# trajectory selection function
def select_config(traj: list[object]) -> list[object]:
    """
    Select configurations from a trajectory.

    Args:
        traj (list[ase.Atoms]): Trajectory list of Atoms objects.

    Returns:
        list[ase.Atoms]: Selected configurations.
    """
    if len(traj) < 10 or len(traj) == 500:
        return traj[-1:]

    else:
        return traj[-10:-9]
