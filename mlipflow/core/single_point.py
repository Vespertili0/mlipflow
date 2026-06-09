from __future__ import annotations

import logging

from ase.io import read
from wfl.autoparallelize import AutoparaInfo
from wfl.calculators.generic import calculate as generic_calc
from wfl.configset import ConfigSet, OutputSpec

from mlipflow.data import clean_up, merge_clean_chunks, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def run_single_point(
    in_file: str,
    out_file: str,
    output_prefix: str,
    calculator: tuple,
    remote_info: object = None,
) -> None:
    """
    Run a single point calculation using the provided calculator.

    Args:
        in_file (str): Input file path.
        out_file (str): Output file path.
        output_prefix (str): Prefix for output properties.
        calculator (tuple): Calculator tuple (class, args, kwargs).
        remote_info (object | None): Remote info object. Defaults to None.
    """

    in_config = ConfigSet(in_file)
    out_config = OutputSpec(out_file)
    if remote_info is None:
        generic_calc(
            inputs=in_config,
            outputs=out_config,
            calculator=calculator,
            output_prefix=output_prefix,
            properties=["energy", "forces"],
        )
    else:
        generic_calc(
            inputs=in_config,
            outputs=out_config,
            calculator=calculator,
            output_prefix=output_prefix,
            properties=["energy", "forces"],
            autopara_info=AutoparaInfo(
                remote_info=remote_info, num_inputs_per_python_subprocess=1
            ),
        )


def run_chunked_qe_sp(
    in_file: str,
    out_file: str,
    chunk_size: int,
    qchem_strategy: object,
    keep_info_keys: list | None = None,
    ecut_eV: int = 450,
    kpts: tuple = (3, 3, 1),
    num_inputs_per_queued_job: int = 2,
    dipole: bool = False,
    dftd3: bool = False,
) -> None:
    """
    Run a chunked single point calculation using the provided calculator.

    This function splits the input file into chunks of a specified size,
    runs single point calculations on each chunk, and merges the results
    into a single output file.

    Args:
        in_file (str): Input file path.
        out_file (str): Output file path.
        chunk_size (int): Size of chunks.
        qchem_strategy (object): Strategy object for QChem calculation.
        ecut_eV (int): Energy cutoff in eV. Defaults to 450.
        kpts (tuple): K-points tuple. Defaults to (3,3,1).
        num_inputs_per_queued_job (int): Number of inputs per queued job. Defaults to 2.
        dipole (bool): Whether to include dipole correction. Defaults to False.
        dftd3 (bool): Whether to include DFT-D3 correction. Defaults to False.
    """
    if keep_info_keys is None:
        keep_info_keys = ["DFT_energy"]
    chunk_list = _chunk_indices(in_file, chunk_size=chunk_size)
    chunk_files = [f"tmp_{n}.xyz" for n in range(len(chunk_list))]

    logger.info(f"Preparing {len(chunk_list)} batches of {chunk_size} for DFT-SP")

    # Load all configurations once if we are passed a list of files
    all_atoms_list = None
    if isinstance(in_file, list):
        all_atoms_list = list(ConfigSet(in_file))

    # run single point calculations on each chunk
    for n, chunk in enumerate(chunk_list):
        if all_atoms_list is not None:
            if ":" in chunk:
                parts = chunk.split(":")
                start = int(parts[0]) if parts[0] else None
                end = int(parts[1]) if len(parts) > 1 and parts[1] else None
                step = int(parts[2]) if len(parts) > 2 and parts[2] else None
                atoms = all_atoms_list[slice(start, end, step)]
            else:
                atoms = [all_atoms_list[int(chunk)]]
        else:
            atoms = read(in_file, index=chunk)

        run_single_point(
            in_file=atoms,
            out_file=f"tmp_{n}.xyz",
            output_prefix=qchem_strategy.qe_prefix,
            calculator=qchem_strategy.get_calculator(
                job_name="QE_",
                ecut_eV=ecut_eV,
                kpts=kpts,
                dipole=dipole,
                dftd3=dftd3,
                num_inputs_per_queued_job=num_inputs_per_queued_job,
            ),
            remote_info=qchem_strategy.remote_info,
        )
        clean_up()

    # merge all chunks into one file
    logger.info("Merging chunks into single file")

    if isinstance(out_file, list) and len(out_file) == 1:
        out_file_write = out_file[0]
    else:
        out_file_write = out_file

    merge_clean_chunks(
        in_files=chunk_files, out_file=out_file_write, keep_info_keys=keep_info_keys
    )

    # remove the temporary files
    clean_up(key="tmp_")
    logger.info("Chunked DFT-SP calculations completed successfully")


def _chunk_indices(in_file: str, chunk_size: int = 50) -> list[str]:
    """
    Generate chunk indices for splitting the file.

    Args:
        in_file (str): Input file path or ConfigSet object.
        chunk_size (int): Size of chunks. Defaults to 50.

    Returns:
        list[str]: List of index strings in format 'start:end'.
    """
    configs = list(ConfigSet(in_file))
    n_configs = len(configs)
    return [
        f"{i}:{min(i + chunk_size, n_configs)}" for i in range(0, n_configs, chunk_size)
    ]
