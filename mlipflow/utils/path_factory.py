"""Centralised path factory for deterministic filesystem routing.

This module consolidates all filesystem path construction into a single
source of truth, replacing ad-hoc string manipulation (e.g.
``.replace(".xyz", ...)``). Workflow nodes consult these functions
rather than predicting or transforming paths using raw strings.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_step_path(
    input_file: str | Path,
    step_suffix: str,
    iteration: int,
    workdir: str | Path = "iterations",
) -> str:
    """Build a deterministic output path for a workflow step.

    Parameters
    ----------
    input_file : str or Path
        The original input file whose stem is used to derive the
        output filename.
    step_suffix : str
        A short identifier for the workflow step, appended to the
        clean stem (e.g. ``"dft_sp"``, ``"mace_sp"``, ``"md"``).
    iteration : int
        The current active-learning iteration index.
    workdir : str or Path, optional
        Root directory under which iteration folders are created.
        Defaults to ``"iterations"``.

    Returns
    -------
    str
        Absolute-style string path of the form
        ``<workdir>/iter_<iteration>/<clean_stem>_<step_suffix>.xyz``.
    """
    base_path = Path(input_file)
    stem_name = base_path.stem

    # Strip away any previous internal loop designations to prevent
    # path-nesting bugs when files are re-processed across iterations.
    for tag in (
        ".cleaned",
        ".dft",
        ".mace",
        "_dft_sp",
        "_mace_sp",
        "_md",
        "_opt",
        "_neb",
    ):
        stem_name = stem_name.split(tag)[0]

    target_dir = Path(workdir) / f"iter_{iteration}"
    target_dir.mkdir(parents=True, exist_ok=True)

    resolved = str(target_dir / f"{stem_name}_{step_suffix}.xyz")
    logger.debug("Resolved step path: %s", resolved)
    return resolved


def create_iteration_directory(
    iteration: int, workdir: str | Path = "iterations"
) -> dict[str, str]:
    """Create the standard directory hierarchy for one iteration.

    Absorbs the legacy ``DataManager._create_folder_structure`` logic
    into the centralised path factory.

    Parameters
    ----------
    iteration : int
        The iteration index (e.g. ``0``, ``1``, ``2``).
    workdir : str or Path, optional
        Root working directory.  Defaults to ``"iterations"``.

    Returns
    -------
    dict[str, str]
        A mapping of logical names to created directory paths::

            {
                "iter_dir": "<workdir>/iter_<iteration>",
                "mlip_dir": "<workdir>/iter_<iteration>/MLIP",
                "sgen_dir": "<workdir>/iter_<iteration>/SGEN",
                "ensemble_dir": "<workdir>/ENSEMBLE",
            }
    """
    base = Path(workdir)
    iter_dir = base / f"iter_{iteration}"
    mlip_dir = iter_dir / "MLIP"
    sgen_dir = iter_dir / "SGEN"
    ensemble_dir = base / "ENSEMBLE"

    for folder in (iter_dir, mlip_dir, sgen_dir, ensemble_dir):
        folder.mkdir(parents=True, exist_ok=True)

    dirs = {
        "iter_dir": str(iter_dir),
        "mlip_dir": str(mlip_dir),
        "sgen_dir": str(sgen_dir),
        "ensemble_dir": str(ensemble_dir),
    }
    logger.info("Created iteration directory structure: %s", dirs)
    return dirs
