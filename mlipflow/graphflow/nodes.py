from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from typing_extensions import NotRequired
from wfl.configset import ConfigSet, OutputSpec
from wfl.map import map as wfl_map

from mlipflow.core.neb_pairing import create_neb_pairs
from mlipflow.core.single_point import run_chunked_qe_sp, run_single_point
from mlipflow.data import clean_up, setup_logging
from mlipflow.data.labeller import relabel_configs
from mlipflow.data.processing import (
    check_maxforce_and_cleanarrays,
    clean_configset_data,
    update_configset_tag,
)
from mlipflow.data.selection import (
    select_by_uncertainty,
    split_configset_by_force_agreement,
)
from mlipflow.data.selector import ConfigurationSelector
from mlipflow.strategies.structure_generators import (
    MDGen,
    NEBGen,
    OPTGen,
    StructureGenStrategy,
)
from mlipflow.utils.path_factory import resolve_step_path

if TYPE_CHECKING:
    from ase import Atoms

    from mlipflow.strategies.dft import QChemStrategy
    from mlipflow.strategies.mlip import MLIPStrategy

else:
    QChemStrategy = Any
    MLIPStrategy = Any

setup_logging()
logger = logging.getLogger(__name__)
#####################################################


class EnsembleState(TypedDict):
    """
    State dictionary for the DFT-MACE fitting workflow
    Attributes:
        configs (list[str]): List of configuration file paths
        outfile (NotRequired[list[str]]): Optional output file paths
        qchem_strategy (QChemStrategy): Quantum chemistry calculation strategy
        mlip_strategy (MLIPStrategy): MLIP fitting strategy
        structure_gen_strategy (NotRequired[StructureGenStrategy]): Strategy for structure generation (MD/OPT)
        calculation_kwargs (NotRequired[dict[str, Any]]): Dictionary containing parameters for various calculation steps.
            Expected keys:
            - 'dft_scf': kwargs for DFT single point calculations (e.g. kpts, ecut_eV)
            - 'mlip_sp': kwargs for MLIP single point calculations (e.g. dispersion)
            - 'mlip_gen': kwargs for MLIP structure generation (e.g. dispersion)
            - 'initial_sampling': kwargs for initial sampling nodes
                - 'basin_constraints': list of constraints for basin sampling, e.g. FixAtoms
                - 'basin__mlip_gen': kwargs for basin MLIP structure generation (e.g. dispersion)
                - 'basin__structure_gen_params': kwargs for MD structure generation (e.g. rxn_constraints_dict, n_images)
                - 'neb_config': dict of neb configuration parameters
                - 'neb__mlip_gen': kwargs for neb MLIP MD generation (e.g. dispersion)
                - 'neb__structure_gen_params': kwargs for MD generation (e.g. rxn_constraints_dict, n_images)
            - 'fps_selection': kwargs for configuration selection
                - 'descriptor_key': key for descriptor storage (e.g. 'SOAP')
                - 'descriptor_string': descriptor string for quippy
                - 'info_field': info field for histogram selection
                - 'n_optimal': optimal number of configs (optional)
                - 'n_max': max configs for optimal N search (optional)
            - 'neb_analysis': kwargs for NEB trajectory analysis
                - 'prefixes': method prefixes for energy/force keys (default: ("MACE", "DFT"))
                - 'n_frames': frames per NEB pathway (default: None -> single pathway)
                - 'dpi': figure resolution (default: 150)
                - 'fmt': output format, e.g. "png", "pdf", "svg" (default: "png")
    """

    configs: list[str]
    outfile: NotRequired[list[str]]
    iteration: NotRequired[int]
    step_counter: NotRequired[int]
    qchem_strategy: NotRequired[QChemStrategy]
    mlip_strategy: NotRequired[MLIPStrategy]
    structure_gen_strategy: NotRequired[StructureGenStrategy]
    calculation_kwargs: NotRequired[dict[str, Any]]
    original_configs: NotRequired[list[str]]
    last_training_configs: NotRequired[list[str]]


@dataclass
class WorkflowError(Exception):
    """Base exception for workflow errors"""

    message: str
    state: EnsembleState | None = None


def _apply_static_constraints(at: Atoms, constraints: list[Any]) -> Atoms:
    """
    Applies a list of ASE constraints to the Atoms object.

    Parameters
    ----------
    at : Atoms
        The atoms object to modify.
    constraints : List[Any]
        A list of ASE constraint objects (e.g., FixAtoms, FixBondLength).
    """
    if constraints:
        # We replace existing constraints to match the behavior of the original script
        at.constraints = constraints
    return at


def _log_config_counts(configs: ConfigSet, msg: str = "") -> None:
    """Log the number of configurations in a ConfigSet."""
    logger.debug(f"...{len(list(configs))} configurations {msg}")

    return


def merge_configs(state: EnsembleState) -> EnsembleState:
    """Merge original and generated configurations."""
    logger.info("Merging original and generated configurations")

    if not state.get("original_configs"):
        logger.error("No original configurations provided in state")
        raise ValueError("No original configurations provided in state")

    step_counter = state.get("step_counter", 1)
    current_iter = state.get("iteration", 0)

    outfile_path = resolve_step_path(
        step_suffix="merged", iteration=current_iter, step_counter=step_counter
    )

    if Path(outfile_path).exists() and Path(outfile_path).stat().st_size > 0:
        logger.info("Valid merged configs output detected on disk. Skipping execution.")
        return {**state, "configs": [outfile_path], "step_counter": step_counter + 1}

    merged = ConfigSet(state["original_configs"]) + ConfigSet(state["configs"])
    _log_config_counts(merged, msg="after merging")
    OutputSpec(outfile_path).write(merged)

    return {**state, "configs": [outfile_path], "step_counter": step_counter + 1}


def switch_to_neb_generation(state: EnsembleState) -> EnsembleState:
    """Switch from OPT to NEB generation."""
    logger.info("Switching from OPT to NEB generation")

    calc_kwargs = state.get("calculation_kwargs", {})
    calc_kwargs["mlip_gen"] = calc_kwargs.get("neb__mlip_gen", {})
    neb_params = calc_kwargs.get("neb__structure_gen_params", {})
    new_strategy = NEBGen(params=neb_params)

    return {
        **state,
        "structure_gen_strategy": new_strategy,
        "calculation_kwargs": calc_kwargs,
    }


def switch_to_pathmd_generation(state: EnsembleState) -> EnsembleState:
    """
    Switch the active structure generation strategy back to MDGen
    to handle downstream path-sampling MD.
    """
    logger.info("Switching state strategy from OPT back to MD for path-sampling")

    calc_kwargs = state.get("calculation_kwargs", {})
    calc_kwargs["mlip_gen"] = calc_kwargs.get("neb__mlip_gen", {})
    sampling_kwargs = calc_kwargs.get("initial_sampling", {})
    neb_md_params = sampling_kwargs.get("neb__structure_gen_params", {})
    uncertainty_thrs = calc_kwargs.get("fps_selection", {}).get("uncertainty_thrs", 12)

    new_strategy = MDGen(
        uncertainty_thrs=uncertainty_thrs, n_failed_steps=10, params=neb_md_params
    )

    return {
        **state,
        "structure_gen_strategy": new_strategy,
        "calculation_kwargs": calc_kwargs,
    }


def switch_to_opt_generation(state: EnsembleState) -> EnsembleState:
    """
    Switch the active structure generation strategy from MDGen to OPTGen.
    """
    logger.info("Switching state strategy from MD to OPT geometry optimization")

    calc_kwargs = state.get("calculation_kwargs", {})
    calc_kwargs["mlip_gen"] = calc_kwargs.get("opt__mlip_gen", {})
    opt_params = calc_kwargs.get(
        "opt__structure_gen_params", {"fmax": 0.005, "steps": 250}
    )
    traj_subselect = calc_kwargs.get("opt__traj_subselect", "last")
    new_strategy = OPTGen(traj_subselect=traj_subselect, params=opt_params)

    return {
        **state,
        "structure_gen_strategy": new_strategy,
        "calculation_kwargs": calc_kwargs,
    }


def run_dft_sp(state: EnsembleState) -> EnsembleState:
    """Run DFT single-point calculations on configurations.

    Parameters
    ----------
    state : EnsembleState
        Current workflow state.

    Returns
    -------
    EnsembleState
        Updated workflow state.

    Raises
    ------
    ValueError
        If configs list is empty.
    """
    logger.info("Running DFT single-point calculations")

    if not state.get("configs"):
        logger.error("No configurations provided in state")
        raise ValueError("No configurations provided in state")

    configs = state["configs"]
    current_iter = state.get("iteration", 0)
    step_counter = state.get("step_counter", 1)

    if len(configs) == 1:
        outfile = [
            resolve_step_path(
                step_suffix="dft_sp", iteration=current_iter, step_counter=step_counter
            )
        ]
    else:
        outfile = [
            resolve_step_path(
                step_suffix=f"dft_sp_{i}",
                iteration=current_iter,
                step_counter=step_counter,
            )
            for i in range(len(configs))
        ]

    # Idempotency guard: skip execution if outputs already exist
    if all(
        Path(out_f).exists() and Path(out_f).stat().st_size > 0 for out_f in outfile
    ):
        logger.info("Valid DFT-SP outputs detected on disk. Skipping execution.")
        return {
            **state,
            "configs": outfile,
            "outfile": None,
            "step_counter": step_counter + 1,
        }

    dft_kwargs = state.get("calculation_kwargs", {}).get("dft_scf", {})

    try:
        run_single_point(
            in_file=configs,
            out_file=outfile,
            output_prefix=state["qchem_strategy"].qe_prefix,
            calculator=state["qchem_strategy"].get_calculator(
                job_name="QE_", **dft_kwargs
            ),
            remote_info=state["qchem_strategy"].remote_info,
        )
        logger.debug("DFT calculations completed successfully")
    except Exception as e:
        logger.error(f"DFT calculation failed: {e!s}")
        raise RuntimeError(f"DFT calculation failed: {e!s}") from e
    finally:
        logger.debug("Cleaning up DFT calculation files")
        clean_up()
        _log_config_counts(ConfigSet(outfile), msg="after DFT-SP")

    return {
        **state,
        "configs": outfile,
        "outfile": None,
        "step_counter": step_counter + 1,
    }


def run_dft_sp_chunked(state: EnsembleState) -> EnsembleState:
    """Run chunked DFT single-point calculations on configurations.

    Parameters
    ----------
    state : EnsembleState
        Current workflow state.

    Returns
    -------
    EnsembleState
        Updated workflow state.

    Raises
    ------
    ValueError
        If configs list is empty.
    """
    logger.info("Running chunked DFT single-point calculations")

    if not state.get("configs"):
        logger.error("No configurations provided in state")
        raise ValueError("No configurations provided in state")

    configs = state["configs"]
    current_iter = state.get("iteration", 0)
    step_counter = state.get("step_counter", 1)

    if len(configs) == 1:
        outfile = [
            resolve_step_path(
                step_suffix="dft_sp", iteration=current_iter, step_counter=step_counter
            )
        ]
    else:
        outfile = [
            resolve_step_path(
                step_suffix=f"dft_sp_{i}",
                iteration=current_iter,
                step_counter=step_counter,
            )
            for i in range(len(configs))
        ]

    # Idempotency guard: skip execution if outputs already exist
    if all(
        Path(out_f).exists() and Path(out_f).stat().st_size > 0 for out_f in outfile
    ):
        logger.info(
            "Valid chunked DFT-SP outputs detected on disk. Skipping execution."
        )
        return {
            **state,
            "configs": outfile,
            "outfile": None,
            "step_counter": step_counter + 1,
        }

    # Get DFT kwargs from state or use defaults
    dft_kwargs = state.get("calculation_kwargs", {}).get("dft_scf", {})

    try:
        run_chunked_qe_sp(
            in_file=configs,
            out_file=outfile,
            qchem_strategy=state["qchem_strategy"],
            kpts=dft_kwargs.get("kpts", (3, 3, 1)),
            dipole=dft_kwargs.get("dipole", False),
            dftd3=dft_kwargs.get("dftd3", False),
            ecut_eV=dft_kwargs.get("ecut_eV", 450),
            chunk_size=dft_kwargs.get("chunk_size", 50),
            #    max_time=dft_kwargs.get("max_time", "01:30:00"),
            #    job_name=dft_kwargs.get("job_name", "QE_"),
            keep_info_keys=dft_kwargs.get(
                "keep_info_keys", ["DFT_energy", "slab", "species"]
            ),
            num_inputs_per_queued_job=dft_kwargs.get("num_inputs_per_queued_job", 1),
        )
        logger.debug("DFT calculations completed successfully")
    except Exception as e:
        logger.error(f"DFT calculation failed: {e!s}")
        raise RuntimeError(f"DFT calculation failed: {e!s}") from e
    finally:
        _log_config_counts(ConfigSet(outfile), msg="after chunked DFT-SP")

    return {
        **state,
        "configs": outfile,
        "outfile": None,
        "step_counter": step_counter + 1,
    }


def run_mlip_sp(state: EnsembleState) -> EnsembleState:
    """Run MLIP single-point calculations on configurations.

    Parameters
    ----------
    state : EnsembleState
        Current workflow state.

    Returns
    -------
    EnsembleState
        Updated workflow state.

    Raises
    ------
    KeyError
        If ``mlip_strategy`` is missing from state.
    ValueError
        If configs list is empty.
    """
    logger.info("Running MLIP-SP calculations")

    if not state.get("configs"):
        logger.error("No configurations provided in state")
        raise ValueError("No configurations provided in state")

    if "mlip_strategy" not in state:
        logger.error("mlip_strategy not found in state")
        raise KeyError("mlip_strategy not found in state")

    mlip = state["mlip_strategy"]
    configs = state["configs"]
    current_iter = state.get("iteration", 0)
    step_counter = state.get("step_counter", 1)

    if len(configs) == 1:
        outfile = [
            resolve_step_path(
                step_suffix="mace_sp", iteration=current_iter, step_counter=step_counter
            )
        ]
    else:
        outfile = [
            resolve_step_path(
                step_suffix=f"mace_sp_{i}",
                iteration=current_iter,
                step_counter=step_counter,
            )
            for i in range(len(configs))
        ]

    # Idempotency guard: skip execution if outputs already exist
    if all(
        Path(out_f).exists() and Path(out_f).stat().st_size > 0 for out_f in outfile
    ):
        logger.info("Valid MLIP-SP outputs detected on disk. Skipping execution.")
        return {
            **state,
            "configs": outfile,
            "outfile": None,
            "step_counter": step_counter + 1,
        }

    # Get MLIP kwargs from state
    mlip_kwargs = state.get("calculation_kwargs", {}).get("mlip_sp", {}).copy()
    dispersion = mlip_kwargs.pop("dispersion", True)
    num_inputs = mlip_kwargs.pop("num_inputs_per_queued_job", 300)
    max_time = mlip_kwargs.pop("max_time", "01:30:00")

    try:
        run_single_point(
            in_file=configs,
            out_file=outfile,
            output_prefix=mlip.mlip_prefix,
            calculator=mlip.get_calculator(
                job_name="mSP_",
                dispersion=dispersion,
                num_inputs_per_queued_job=num_inputs,
                max_time=max_time,
                **mlip_kwargs,
            ),
            remote_info=mlip.remote_info,
        )
        logger.debug("MLIP calculations completed successfully")
    except Exception as e:
        logger.error(f"MLIP calculation failed: {e!s}")
        raise RuntimeError(f"MLIP calculation failed: {e!s}") from e
    finally:
        logger.debug("Cleaning up MLIP calculation files")
        clean_up()
        _log_config_counts(ConfigSet(outfile), msg="after MLIP-SP")

    return {
        **state,
        "configs": outfile,
        "outfile": None,
        "step_counter": step_counter + 1,
    }


def assess_n_select(state: EnsembleState) -> EnsembleState:
    main_suffix = "train"
    side_suffix = "test"

    configs = state["configs"]
    step_counter = state.get("step_counter", 1)
    current_iter = state.get("iteration", 0)

    train_file = resolve_step_path(
        step_suffix="train_dft", iteration=current_iter, step_counter=step_counter
    )
    test_file = resolve_step_path(
        step_suffix="test_dft", iteration=current_iter, step_counter=step_counter
    )

    out_files = {main_suffix: train_file, side_suffix: test_file}

    # Idempotency guard: skip execution if outputs already exist
    if (
        Path(train_file).exists()
        and Path(train_file).stat().st_size > 0
        and Path(test_file).exists()
        and Path(test_file).stat().st_size > 0
    ):
        logger.info("Valid DFT split outputs detected on disk. Skipping execution.")
        return {
            **state,
            "configs": [train_file],
            "outfile": [test_file],
            "step_counter": step_counter + 1,
        }

    split_configset_by_force_agreement(
        in_file=configs,
        out_file=out_files,
        pair_tuple=(
            state["qchem_strategy"].qe_prefix,
            state["mlip_strategy"].mlip_prefix,
        ),
        main_suffix=main_suffix,
        side_suffix=side_suffix,
    )

    return {
        **state,
        "configs": [train_file],
        "outfile": [test_file],
        "step_counter": step_counter + 1,
    }


def run_mace_fit(state: EnsembleState) -> EnsembleState:
    configs = state["configs"]
    test_configs = state["outfile"]

    state["mlip_strategy"].fit_new_model(
        in_file=configs, test_configs=test_configs, seed=123, restart=False
    )
    step_counter = state.get("step_counter", 1)
    return {**state, "outfile": None, "step_counter": step_counter + 1}


def _run_structure_generation_logic(
    state: EnsembleState,
    configs: Any,
    step_suffix: str | None = None,
    mlip_gen_kwargs_override: dict | None = None,
    structure_gen_params_override: dict | None = None,
) -> EnsembleState:
    """
    Internal helper to run structure generation logic on specific configs.
    configs can be list[str], ConfigSet, or list[Atoms].
    """
    logger.info("...Running structure generation logic")

    if "mlip_strategy" not in state:
        logger.error("mlip_strategy not found in state")
        raise KeyError("mlip_strategy not found in state")

    if "structure_gen_strategy" not in state:
        logger.error("structure_gen_strategy not found in state")
        raise KeyError("structure_gen_strategy not found in state")

    # Get MLIP kwargs for structure generation
    # Allow override for specific steps (e.g. basin vs neb)
    mlip_kwargs = state.get("calculation_kwargs", {}).get("mlip_gen", {})
    if mlip_gen_kwargs_override:
        mlip_kwargs = {**mlip_kwargs, **mlip_gen_kwargs_override}

    # Determine the property prefix based on the strategy type
    prefix_map = {"md": "md", "opt": "optimize", "neb": "neb"}

    mlip = state["mlip_strategy"]
    structure_generator = state["structure_gen_strategy"]

    # Set default structure generation parameters
    sg_params = structure_generator.params.copy()
    if structure_gen_params_override:
        sg_params = {**structure_generator.params, **structure_gen_params_override}
        structure_generator.params = sg_params

    calc_type = getattr(structure_generator, "calc_prefix", "opt")
    op_name = prefix_map.get(calc_type, "optimize")

    energy_key = f"last_op__{op_name}_energy"
    force_key = f"last_op__{op_name}_forces"

    # current_files = state["configs"]
    current_iter = state.get("iteration", 0)
    step_counter = state.get("step_counter", 1)

    if step_suffix is None:
        step_suffix = f"structure_{calc_type}"

    outfile_path = resolve_step_path(
        step_suffix=step_suffix, iteration=current_iter, step_counter=step_counter
    )
    outfile = [outfile_path]

    # Idempotency guard: skip execution if outputs already exist
    if all(
        Path(out_f).exists() and Path(out_f).stat().st_size > 0 for out_f in outfile
    ):
        logger.info(
            "Valid structure generation outputs detected on disk. Skipping execution."
        )
        return {
            **state,
            "configs": outfile,
            "outfile": None,
            "step_counter": step_counter + 1,
        }

    # Handle case where input is list of Atoms (e.g. from NEB generation) but outfile is list of files.
    # We merge all outputs to the first file in the list.
    output_arg = outfile
    is_atoms_input = isinstance(configs, list) and (
        len(configs) == 0 or not isinstance(configs[0], str)
    )

    if is_atoms_input and isinstance(outfile, list) and len(outfile) > 0:
        output_arg = outfile[0]
        outfile = [output_arg]

    # Clean up any previous .mace.xyz files if we are overwriting, to avoid appending to old runs?
    # WFL/ASE append by default sometimes.
    # But here we probably want fresh files for this step.
    # We rely on wfl/ase write modes.

    try:
        logger.info(
            f"Generating new structures using {structure_generator.__class__.__name__}"
        )
        logger.debug(f"MLIP run settings: {mlip_kwargs}")
        logger.debug(f"StructureGen parameters: {sg_params}")

        structure_generator.generate_new_structures(
            in_file=configs,
            out_file=output_arg,
            calculator=mlip.get_calculator(job_name="mSG_", **mlip_kwargs),
            remote_info=mlip.remote_info,
        )
        tag_dict = {
            "info": {energy_key: f"{mlip.mlip_prefix}energy"},
            "array": {force_key: f"{mlip.mlip_prefix}forces"},
        }
        for tag_type, tags in tag_dict.items():
            for xyz in outfile:
                update_configset_tag(
                    in_config=xyz, out_file=xyz, tag_dict=tags, tag_type=tag_type
                )
    except Exception as e:
        logger.error(f"Structure generation failed: {e!s}")
        raise RuntimeError(f"Structure generation failed: {e!s}") from e
    finally:
        logger.debug("Cleaning up structure generation files")
        clean_up()
        _log_config_counts(ConfigSet(outfile), msg="after structure generation")

    return {
        **state,
        "configs": outfile,
        "outfile": None,
        "step_counter": step_counter + 1,
    }


def run_mlip_structure_generation(state: EnsembleState) -> EnsembleState:
    """Generate new structures via MLIP MD/OPT/DyNEB."""
    if not state.get("configs"):
        raise ValueError("No configurations provided in state")

    # Just pass the files directly
    return _run_structure_generation_logic(state, state["configs"])


def run_apply_basin_constraints(state: EnsembleState) -> EnsembleState:
    """
    Apply basin constraints and run structure generation immediately.
    This avoids writing intermediate files which would lose constraint information.
    """
    logger.info("Running Basin-MD with applied constraints")

    if not state.get("configs"):
        logger.error("No configurations provided in state")
        raise ValueError("No configurations provided in state")

    sampling_kwargs = state.get("calculation_kwargs", {}).get("initial_sampling", {})
    basin_constraints = sampling_kwargs.get("basin_constraints", [])

    logger.info(
        f"Applying {len(basin_constraints)} basin constraints and running generation"
    )

    map_func = partial(_apply_static_constraints, constraints=basin_constraints)

    # Create constrained input iterable
    # We use ConfigSet to read, apply map_func to get iterator of constrained atoms
    constrained_inputs = wfl_map(
        inputs=ConfigSet(state["configs"]),
        outputs=OutputSpec(),  # No output file, returns iterable/ConfigSet-like
        map_func=map_func,
    )

    # Check for override kwargs
    basin_mlip_gen = sampling_kwargs.get("basin__mlip_gen", None)
    basin_structure_gen_params = sampling_kwargs.get(
        "basin__structure_gen_params", None
    )

    # Pass directly to generation logic
    return _run_structure_generation_logic(
        state=state,
        configs=constrained_inputs,
        step_suffix="basin_md",
        mlip_gen_kwargs_override=basin_mlip_gen,
        structure_gen_params_override=basin_structure_gen_params,
    )


def run_generate_neb_pairs(state: EnsembleState) -> EnsembleState:
    """
    Generate NEB pairs and run structure generation immediately.
    """
    logger.info("Generating NEB pairs for structure generation")
    neb_structure_gen_params = None
    neb_mlip_gen = None

    if not state.get("configs"):
        logger.error("No configurations provided in state")
        raise ValueError("No configurations provided in state")

    # Check for override kwargs
    if isinstance(state["structure_gen_strategy"], MDGen):
        logger.info(
            "Structure generation strategy is MD on NEB pairs - initial sampling"
        )
        sampling_kwargs = state.get("calculation_kwargs", {}).get(
            "initial_sampling", {}
        )
        neb_config = sampling_kwargs.get("neb_config", {})
        neb_mlip_gen = sampling_kwargs.get("neb__mlip_gen", None)
        neb_structure_gen_params = sampling_kwargs.get(
            "neb__structure_gen_params", None
        )

    elif isinstance(state["structure_gen_strategy"], NEBGen):
        logger.info("Structure generation strategy is NEB")
        neb_config = state.get("calculation_kwargs", {}).get("neb_config", {})

    if not neb_config or "rxn_constraints_dict" not in neb_config:
        logger.error(
            "neb_config with 'rxn_constraints_dict' is required for NEB pair generation."
        )
        raise ValueError(
            "neb_config with 'rxn_constraints_dict' is required for NEB pair generation."
        )

    # Capture the input configs (from Basin MD) to merge later
    original_configs = state["configs"]

    all_prepared_configs = []

    for config_file in state["configs"]:
        # Clean data first
        current_iter = state.get("iteration", 0)
        cleaned_file = resolve_step_path(config_file, "cleaned", current_iter)
        clean_configset_data(
            ConfigSet(config_file), OutputSpec(cleaned_file, overwrite=True)
        )

        results = create_neb_pairs(xyz_file=cleaned_file, **neb_config)

        # Flatten results (list of lists of Atoms/ConfigSets)
        for res in results:
            all_prepared_configs.extend(list(ConfigSet(res)))

        # cleanup clean file
        if Path(cleaned_file).exists():
            Path(cleaned_file).unlink()

    # Pass flattened list of prepared atoms to generation logic
    gen_result = _run_structure_generation_logic(
        state=state,
        configs=all_prepared_configs,
        step_suffix="neb_md"
        if isinstance(state["structure_gen_strategy"], MDGen)
        else "neb_opt",
        mlip_gen_kwargs_override=neb_mlip_gen,
        structure_gen_params_override=neb_structure_gen_params,
    )

    # Update state: configs now contains ALL generated structures (Basin + NEB) so MLIP SP runs on everything
    return {**gen_result, "original_configs": original_configs}


def run_config_fps_selection(state: EnsembleState) -> EnsembleState:
    """
    Run two-stage configuration selection.
    Calculates global descriptors first, then runs selection.
    """
    logger.info("Running configuration FPS selection...")

    if not state.get("configs"):
        logger.error("No configurations provided in state")
        raise ValueError("No configurations provided in state")

    selection_kwargs = state.get("calculation_kwargs", {}).get("fps_selection", {})

    # Required parameters check
    if "descriptor_string" not in selection_kwargs:
        raise ValueError(
            "selection kwargs must contain 'descriptor_string' for descriptor calculation."
        )

    # Defaults
    desc_key = selection_kwargs.get("descriptor_key", "SOAP")
    step_counter = state.get("step_counter", 1)
    current_iter = state.get("iteration", 0)
    output_prefix_resolved = resolve_step_path(
        step_suffix="selection", iteration=current_iter, step_counter=step_counter
    )
    output_prefix = output_prefix_resolved.replace(".xyz", "")
    seed = selection_kwargs.get("seed", 10)

    logger.info("Initializing ConfigurationSelector")
    selector = ConfigurationSelector(
        inputs=state["configs"], output_prefix=output_prefix, seed=seed
    )

    logger.info("Calculating global descriptors")
    selector.calculate_global_descriptors(
        descs=[selection_kwargs["descriptor_string"]], key=desc_key
    )

    # Extract kwargs for run_two_stage_selection
    run_kwargs = {
        k: v
        for k, v in selection_kwargs.items()
        if k not in ["descriptor_string", "descriptor_key", "seed"]
    }

    logger.info("Running two-stage selection")
    _selected_configs = selector.run_two_stage_selection(**run_kwargs)

    # The selector returns ConfigSets/lists of atoms.
    # We need to flatten them and make them accessible for the next step.
    # Ideally, we write them to a file to update state['configs'].
    # The selector method select_final writes 'final_selection.xyz'.
    # We can point state['configs'] to that, or to the specific output generated.

    final_output_file = f"{output_prefix}_final_selection.xyz"
    # If the file exists, we use it.
    # run_two_stage_selection calls select_final which calls write_selected_and_clean...
    # wait, select_final writes to f'{self.output_prefix}_final_selection.xyz'.
    _log_config_counts(ConfigSet(final_output_file), msg="after selection")

    return {**state, "configs": [final_output_file], "step_counter": step_counter + 1}


def run_config_uncertainty_selection(state: EnsembleState) -> EnsembleState:
    """ """
    logger.info("Running uncertainty-based configuration selection...")

    step_counter = state.get("step_counter", 1)
    current_iter = state.get("iteration", 0)

    selected_configs = resolve_step_path(
        step_suffix="selected_configs",
        iteration=current_iter,
        step_counter=step_counter,
    )

    select_by_uncertainty(
        train_file=state["last_training_configs"],
        pool_file=state["configs"],
        out_file=selected_configs,
        mlip_strategy=state["mlip_strategy"],
        # certainty_threshold=state['calculation_kwargs']['selection']['certainty_threshold'],
        # pca_variance_threshold: float = 0.95,
        # max_gmm_components: int = 30,
        # gmm_n_init: int = 5,
        # device: str = 'cpu',
        # dtype: "torch.dtype" = None,
    )

    return {**state, "configs": [selected_configs], "step_counter": step_counter + 1}


# def run_gmm_relabel(state: EnsembleState) -> EnsembleState:
#    """
#    Run semi-supervised GMM re-labelling on configurations.
#    """
#    logger.info("Running semi-supervised GMM re-labelling...")
#
#    from mlipflow.data.semi_supervised_gmm import GMMLabelChecker
#
#    if not state.get('configs'):
#        logger.error("No configurations provided in state")
#        raise ValueError("No configurations provided in state")
#
#    if not state.get('last_training_configs'):
#        logger.error("last_training_configs not found in state")
#        raise ValueError("last_training_configs not found in state")
#
#    gmm_kwargs = state.get('calculation_kwargs', {}).get('gmm_relabel', {}).copy()
#    device = gmm_kwargs.pop('device', 'cpu')
#
#    checker = GMMLabelChecker(
#        train_file=state['last_training_configs'],
#        pool_file=state['configs'],
#        mlip_strategy=state['mlip_strategy'],
#        device=device,
#        **gmm_kwargs
#    )
#
#    certain_configs, uncertain_configs, _ = checker.run()
#
#    out_file = 'relabelled_configs.xyz'
#    OutputSpec(out_file).write(ConfigSet(certain_configs))
#    OutputSpec('uncertain_configs.xyz').write(ConfigSet(uncertain_configs))
#
#    return {**state, 'configs': [out_file]}


def run_topology_relabel(state: EnsembleState) -> EnsembleState:
    """
    Run topology-based re-labelling on configurations.
    """
    logger.info("Running topology-based re-labelling...")

    if not state.get("configs"):
        logger.error("No configurations provided in state")
        raise ValueError("No configurations provided in state")

    gcml_kwargs = state.get("calculation_kwargs", {}).get("relabel_check", {}).copy()
    if not gcml_kwargs.get("reference_configs"):
        logger.error("reference_configs not found in gcml_kwargs")
        raise ValueError("reference_configs not found in gcml_kwargs")

    known_configs, unknown_configs = relabel_configs(
        in_file=state["configs"], **gcml_kwargs
    )

    step_counter = state.get("step_counter", 1)
    current_iter = state.get("iteration", 0)

    out_file = resolve_step_path(
        step_suffix="known_configs", iteration=current_iter, step_counter=step_counter
    )
    unknown_file = resolve_step_path(
        step_suffix="unknown_configs", iteration=current_iter, step_counter=step_counter
    )

    if not known_configs:
        logger.warning(
            "No known configs found by relabeller, falling back to treating all configs as known"
        )
        known_configs = list(ConfigSet(state["configs"]))

    OutputSpec(out_file).write(ConfigSet(known_configs))
    OutputSpec(unknown_file).write(ConfigSet(unknown_configs))

    return {**state, "configs": [out_file], "step_counter": step_counter + 1}


# def validate_state(state: EnsembleState, required_keys: list[str]) -> None:
#    """Validate state contains required keys with non-empty values"""
#    missing = [key for key in required_keys if not state.get(key)]
#    if missing:
#        raise WorkflowError(f"Missing required state keys: {missing}", state)
#
#
# def validate_workflow(workflow, initial_state: EnsembleState) -> None:
#    """Validate workflow configuration and initial state"""
#    # Validate nodes
#    required_nodes = {'dft_sp', 'train_mace'}
#    missing_nodes = required_nodes - set(workflow.nodes)
#    if missing_nodes:
#        raise ValueError(f"Missing required nodes: {missing_nodes}")
#
#    # Validate initial state
#    validate_state(initial_state, ['configs', 'qchem_strategy', 'mlip_strategy'])


# def evalute_mlip_error(state: EnsembleState) -> EnsembleState:
#    """
#    Evaluate MLIP error against DFT reference data.
#
#    Args:
#        state (EnsembleState): Current workflow state
#    Returns:
#        EnsembleState: Updated workflow state
#    Raises:
#        KeyError: If required state keys are missing
#        ValueError: If configs list is empty
#    """
#    if not state.get('configs'):
#        raise ValueError("No configurations provided in state")
#
#    try:
#        calculate_mlip_error(
#            in_configs=state['configs'],
#            out_file='error_analysis.xyz',
#            calc_property_prefix=state['mlip_strategy'].mlip_prefix,
#            fig_dir='.'
#            )
#    except Exception as e:
#        logger.error(f"MLIP error evaluation failed: {str(e)}")
#        raise RuntimeError(f"MLIP error evaluation failed: {str(e)}")


def run_rematch_basin_collapse(state: EnsembleState) -> EnsembleState:
    """Collapse redundant PES minima via species-chunked Louvain clustering.

    Configurations are cleaned, bucketed by species, and clustered using a
    topo-energetic network graph with Louvain community detection.  Within
    each basin the lowest-energy representative is retained.  Survivors
    are deterministically sorted, written to disk, and the state is
    updated for downstream NEB pairing.

    Parameters
    ----------
    state : EnsembleState
        Current workflow state.  Must contain ``"configs"`` and
        ``"mlip_strategy"`` keys.  Optional
        ``"calculation_kwargs"["rematch"]`` dict may supply
        ``energy_tol``, ``network_threshold``, ``resolution``,
        ``gamma``, ``zeta``, ``block_size``, ``max_force``, and
        ``atom_slice``.

    Returns
    -------
    EnsembleState
        Updated state with collapsed configuration file list.

    Raises
    ------
    ValueError
        If ``state["configs"]`` is missing or empty.
    KeyError
        If ``state["mlip_strategy"]`` is missing.
    """
    import torch
    from mace.calculators import MACECalculator

    from mlipflow.data.rematch import species_chunked_louvain_clustering

    logger.info("Executing ReMATCH structural basin collapse.")

    if not state.get("configs"):
        raise ValueError("No configurations provided in state.")

    if "mlip_strategy" not in state:
        raise KeyError("mlip_strategy not found in state.")

    # Read parameters from state
    rematch_kwargs = state.get("calculation_kwargs", {}).get("rematch", {})
    energy_tol = rematch_kwargs.get("energy_tol", 0.05)
    network_threshold = rematch_kwargs.get("network_threshold", 0.95)
    resolution = rematch_kwargs.get("resolution", 1.0)
    gamma = rematch_kwargs.get("gamma", 0.1)
    zeta = rematch_kwargs.get("zeta", 1)
    block_size = rematch_kwargs.get("block_size", 512)
    max_force = rematch_kwargs.get("max_force", 15.0)
    atom_slice = rematch_kwargs.get("atom_slice", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_files = state["configs"]

    # Clean and standardise configurations
    mlip = state["mlip_strategy"]
    calc_type = getattr(state.get("structure_gen_strategy"), "calc_prefix", "opt")
    atoms_list = list(ConfigSet(current_files))
    atoms_list = check_maxforce_and_cleanarrays(
        in_file=atoms_list,
        out_file=None,
        mlip_prefix=mlip.mlip_prefix,
        calc=calc_type,
        max_force=max_force,
    )

    num_structures = len(atoms_list)

    if num_structures < 2:
        logger.info("Fewer than 2 configurations; skipping basin collapse.")
        return {**state}

    # Resolve MLIP prefix
    prefix = getattr(mlip, "mlip_prefix", "MACE_")
    if not isinstance(prefix, str):
        prefix = "MACE_"

    # Lazy-load calculator and delegate to core clustering
    mlip_calc = MACECalculator(mlip.model_file, device=device)
    collapsed_configs = species_chunked_louvain_clustering(
        atoms_list,
        mlip_calc,
        prefix,
        device,
        energy_tol=energy_tol,
        network_threshold=network_threshold,
        resolution=resolution,
        gamma=gamma,
        zeta=zeta,
        block_size=block_size,
        atom_slice=atom_slice,
    )

    step_counter = state.get("step_counter", 1)
    current_iter = state.get("iteration", 0)

    # Write survivors to disk
    out_file = resolve_step_path(
        step_suffix="collapsed_basin_configs",
        iteration=current_iter,
        step_counter=step_counter,
    )
    OutputSpec(out_file).write(ConfigSet(collapsed_configs))

    logger.info(
        "Basin collapse complete. Compressed %d -> %d distinct states.",
        num_structures,
        len(collapsed_configs),
    )

    return {**state, "configs": [out_file], "step_counter": step_counter + 1}


# def clean_dft_data(state: EnsembleState) -> EnsembleState:
#    """Clean DFT data by removing unnecessary info and standardizing keys.
#
#    Args:
#        state (EnsembleState): Current workflow state
#    Returns:
#        EnsembleState: Updated workflow state
#    Raises:
#        KeyError: If required state keys are missing
#        ValueError: If configs list is empty
#    """
#    check_maxforce_and_cleanarrays


def analyse_neb_pathways(state: EnsembleState) -> EnsembleState:
    """Execute NEB trajectory analysis and write figures to the iteration directory.

    Parses configuration files in ``state["configs"]``, runs :class:`NEBAnalysis`,
    and exports standardised matplotlib figures directly to the active iteration's
    output directory. Does not mutate state dictionary payloads beyond step counting.

    Parameters
    ----------
    state : EnsembleState
        Current workflow state. Must contain ``"configs"``. Optional
        ``"calculation_kwargs"["neb_analysis"]`` dict may supply
        ``prefixes`` (default: ``("MACE", "DFT")``), ``n_frames``, ``dpi``,
        and ``fmt``.

    Returns
    -------
    EnsembleState
        State with incremented ``step_counter``.

    Raises
    ------
    ValueError
        If ``state["configs"]`` is missing or empty.
    """
    from mlipflow.core import NEBAnalysis

    logger.info("Executing NEB trajectory analysis and writing figures.")

    if not state.get("configs"):
        logger.error("No configurations provided in state.")
        raise ValueError("No configurations provided in state.")

    # Read kwargs and state parameters
    analysis_kwargs = state.get("calculation_kwargs", {}).get("neb_analysis", {})
    prefixes = analysis_kwargs.get("prefixes", ("MACE", "DFT"))
    n_frames = analysis_kwargs.get("n_frames", None)
    dpi = analysis_kwargs.get("dpi", 150)
    fmt = analysis_kwargs.get("fmt", "png")

    current_iter = state.get("iteration", 0)
    step_counter = state.get("step_counter", 1)

    # Resolve output directory under iteration folder
    output_dir = Path("iterations") / f"iter_{current_iter}" / "NEB_plots"

    fig_count = 0
    for config_file in state["configs"]:
        try:
            # Isolate each config file's outputs in its own subdirectory
            file_stem = Path(config_file).stem
            file_out_dir = output_dir / file_stem
            file_out_dir.mkdir(parents=True, exist_ok=True)

            analyzer = NEBAnalysis.from_file(config_file, n_frames=n_frames)
            saved_dict = analyzer.save_plots(
                output_dir=file_out_dir, prefixes=prefixes, dpi=dpi, fmt=fmt
            )
            fig_count += sum(len(lst) for lst in saved_dict.values())
        except Exception as exc:
            logger.warning("NEBAnalysis skipped for file '%s': %s", config_file, exc)

    logger.info("NEB analysis complete. Saved %d figures to %s", fig_count, output_dir)

    return {**state, "step_counter": step_counter + 1}
