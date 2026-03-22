# ruff: noqa: A005
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ase.io import read, write


def setup_logging():
    logging.basicConfig(
        filename="LOG.log",
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_model_name(mlip_dir: str, fit_idx: int, mlip_prefix: str) -> str:
    """Get the model name based on the MLIP prefix and fit index."""
    model_fmt = {"MACE": "model", "GAP": "xml"}.get(mlip_prefix)
    return str(Path(mlip_dir) / f"{mlip_prefix}_{fit_idx}.{model_fmt}")


def initialise_ensembles(ensemble_traj: str, files: dict) -> None:
    """Initialise ensemble from trajectory file."""
    if not Path(ensemble_traj).exists():
        raise FileNotFoundError(f"Ensemble trajectory file {ensemble_traj} not found")
    try:
        shutil.copy2(ensemble_traj, files.get("ensemble_traj"))
        configs = read(files.get("ensemble_traj"), ":")
        write(files.get("ensemble_xyz"), configs)
    except Exception as e:
        raise RuntimeError(f"Failed to Initialise ensembles: {e!s}") from e


def move_mace_model_file(mlip_dir: str, file_prefix: str) -> None:
    """Move the compiled MACE model file to the MLIP directory."""
    shutil.copy2(
        Path(mlip_dir) / "MACE_model" / f"{file_prefix}_compiled.model",
        Path(mlip_dir) / f"{file_prefix}.model",
    )


def clean_up(key: str = "_chunk_") -> None:
    """
    Clean up directories and files matching the key.

    Args:
        key (str): Substring to identify files/directories to remove.
    Raises:
        RuntimeError: If an error occurs during removal.
    Returns:
        None
    """
    try:
        run_dirs = [rd for rd in Path().iterdir() if key in rd.name]
        for rd in run_dirs:
            if rd.is_dir():
                shutil.rmtree(rd)
            elif rd.is_file():
                rd.unlink()
    except Exception as e:
        raise RuntimeError(f"Error removing {rd}: {e}") from e
