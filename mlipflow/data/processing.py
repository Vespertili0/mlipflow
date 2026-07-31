from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io import read, write
from wfl.configset import ConfigSet, OutputSpec
from wfl.map import map as wfl_map

from mlipflow.data import setup_logging

if TYPE_CHECKING:
    from ase import Atoms

setup_logging()
logger = logging.getLogger(__name__)


def update_training_data(
    training_xyz: str, add_xyz: str, out_file: str
) -> None:  # !!! modify to ConfigSet
    """Update training data by merging two XYZ files."""
    update_configset_tag(
        [training_xyz, add_xyz], out_file, {"data_type": "train"}, tag_type="info"
    )


def check_maxforce_and_cleanarrays(
    in_file: str | list[Atoms] | ConfigSet,
    out_file: str | None = None,
    mlip_prefix: str = "MACE",
    calc: str = "opt",
    max_force: float = 12.0,
) -> list[Atoms]:
    """Remove structures with forces exceeding threshold and rename energy/force keys.
    Also acts as a strict filter to drop configurations missing energy data.
    """
    if isinstance(in_file, (str, Path)):
        configs = read(str(in_file), ":")
    elif isinstance(in_file, ConfigSet):
        configs = list(in_file)
    else:
        configs = list(in_file)

    keys = {"md": "md", "opt": "optimize"}
    calc_key = keys.get(calc, calc)
    op_forces_key = f"last_op__{calc_key}_forces"
    op_energy_key = f"last_op__{calc_key}_energy"
    clean_prefix = mlip_prefix.rstrip("_")

    selected_configs = []
    dropped_missing_energy = 0  # Counter for missing energies

    for config in configs:
        forces = None
        if "DFT_forces" in config.arrays:
            forces = config.arrays["DFT_forces"]
        elif op_forces_key in config.arrays:
            forces = config.arrays[op_forces_key]
        elif f"{clean_prefix}_forces" in config.arrays:
            forces = config.arrays[f"{clean_prefix}_forces"]
        elif "forces" in config.arrays:
            forces = config.arrays["forces"]

        if forces is not None and np.max(np.abs(forces)) > max_force:
            continue

        # Strict Energy Check
        energy = None
        for key in [
            "DFT_energy",
            op_energy_key,
            f"{clean_prefix}_energy",
            f"{mlip_prefix}energy",
            "energy",
        ]:
            if key in config.info:
                energy = config.info[key]
                break

        if energy is None:
            dropped_missing_energy += 1
            continue

        selected_configs.append(config)

    # Clean batched logging
    if dropped_missing_energy > 0:
        logger.info(
            f"Dropped {dropped_missing_energy} configurations due to missing energy data."
        )

    # Rename keys for downstream consistency
    for config in selected_configs:
        if op_forces_key in config.arrays:
            config.arrays[f"{clean_prefix}_forces"] = config.arrays[op_forces_key]
            del config.arrays[op_forces_key]
        if op_energy_key in config.info:
            config.info[f"{clean_prefix}_energy"] = config.info[op_energy_key]
            del config.info[op_energy_key]

    if out_file is not None:
        write(out_file, selected_configs)

    return selected_configs


def _rename_configset_tags(at, old_tag, new_tag, tag_type="info") -> Atoms:
    """Function to rename tags in a wfl.configset ConfigSet object using wfl.map-function"""
    assert tag_type in ["array", "info"], "invalid tag"
    if tag_type == "array":
        at.arrays = {
            (new_tag if key == old_tag else key): v for key, v in at.arrays.items()
        }
    elif tag_type == "info":
        at.info = {
            (new_tag if key == old_tag else key): v for key, v in at.info.items()
        }
    return at


def update_configset_tag(in_config, out_file, tag_dict, tag_type) -> None:
    """Update tags in a wfl.configset ConfigSet object using wfl.map-function."""

    def _apply_renames(at, tags, t_type):
        for old, new in tags.items():
            _rename_configset_tags(at, old, new, t_type)
        return at

    # Create temp file to avoid overwriting input if in_config matches out_file
    dir_name = Path(out_file).parent if out_file else Path()
    fd, tmp_path = tempfile.mkstemp(suffix=".xyz", dir=dir_name)
    os.close(fd)

    try:
        wfl_map(
            inputs=ConfigSet(in_config),
            outputs=OutputSpec(tmp_path, overwrite=True),
            map_func=_apply_renames,
            args=[tag_dict, tag_type],
        )
        Path(tmp_path).replace(out_file)
    except Exception:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise


def add_configset_tag(in_config, out_file, tag_dict) -> None:
    """Update a wfl.configset with a new tag."""
    configs = ConfigSet(in_config)
    OutputSpec(out_file, tags=tag_dict, overwrite=True).write(configs)


def filter_info_dict(info_dict: dict, keep_info: list) -> dict:
    """Filter the info dictionary to keep only specified keys."""
    return {key: info_dict[key] for key in keep_info if key in info_dict}


def split_success_failed_configs(
    configs: list, key: str = "DFT_energy"
) -> tuple[list, list]:
    """Split configurations into successful and failed based on a key in the info dictionary."""
    success_configs = []
    failed_configs = []
    for config in configs:
        if key not in config.info:
            failed_configs.append(config)
        else:
            success_configs.append(config)

    return success_configs, failed_configs


def merge_clean_chunks(
    in_files: list, out_file: str, keep_info_keys: list | None = None
) -> None:
    """Merge all chunks into one file and clean up the data.info."""
    if keep_info_keys is None:
        keep_info_keys = ["DFT_energy"]
    success_configs = []
    failed_configs = []

    for file in in_files:
        success, failed = split_success_failed_configs(
            configs=read(file, ":"), key="DFT_energy"
        )
        for config in success:
            config.info = filter_info_dict(
                info_dict=config.info, keep_info=keep_info_keys
            )
        success_configs += success
        failed_configs += failed
    OutputSpec(out_file, tags={"data_type": "train"}).write(ConfigSet(success_configs))
    name, ext = Path(out_file).stem, Path(out_file).suffix
    OutputSpec(f"{name}_failed{ext}").write(ConfigSet(failed_configs))


def _clean_atoms_attributes(
    at: Atoms, keep_info_keys: list, keep_array_keys: list
) -> Atoms:
    """
    Helper function to clean atoms.info and atoms.arrays dictionaries.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object to clean.
    keep_info_keys : list
        List of keys to keep in atoms.info.
    keep_array_keys : list
        List of keys to keep in atoms.arrays.

    Returns
    -------
    ase.Atoms
        Cleaned Atoms object.
    """
    at.info = {k: v for k, v in at.info.items() if k in keep_info_keys}
    at.arrays = {k: v for k, v in at.arrays.items() if k in keep_array_keys}

    return at


def clean_configset_data(inputs, outputs, keep_info_keys=None, keep_array_keys=None):
    """
    Clean the info and array attributes of a configset.

    Parameters
    ----------
    inputs : wfl.configset.ConfigSet or list(Atoms) or list(str)
        Input configurations.
    outputs : wfl.configset.OutputSpec or str
        Output configuration.
    keep_info_keys : list, optional
        List of keys to keep in atoms.info. Default is ['slab', 'species'].
    keep_array_keys : list, optional
        List of keys to keep in atoms.arrays. Default is ['numbers', 'positions', 'tags'].

    Returns
    -------
    wfl.configset.OutputSpec
        The output specification containing the processed configurations.
    """
    if keep_info_keys is None:
        keep_info_keys = ["slab", "species"]
    if keep_array_keys is None:
        keep_array_keys = ["numbers", "positions", "tags"]

    # Merge defaults with user provided lists to ensure minimum keys are always kept
    default_info = {"slab", "species"}
    default_arrays = {"numbers", "positions", "tags"}

    keep_info_keys = list(set(keep_info_keys) | default_info)
    keep_array_keys = list(set(keep_array_keys) | default_arrays)

    return wfl_map(
        inputs=inputs,
        outputs=outputs,
        map_func=_clean_atoms_attributes,
        args=[keep_info_keys, keep_array_keys],
    )
