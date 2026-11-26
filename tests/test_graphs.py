from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from wfl.configset import ConfigSet, OutputSpec

from mlipflow.graphflow.graphs import (
    execute_initial_basin_pathsampling_md_block,
    execute_opt_neb_combination_block,
)
from mlipflow.graphflow.nodes import EnsembleState, run_mlip_structure_generation
from mlipflow.strategies.dft import EMTCalc
from mlipflow.strategies.mlip import MACEModel
from mlipflow.strategies.structure_generators import MDGen, OPTGen
from mlipflow.utils.path_factory import resolve_step_path


def test_execute_initial_basin_pathsampling_md_block(tmp_path):
    """
    Test execute_initial_basin_pathsampling_md_block with MACE and MDGen.
    We mock MACE calculator with EMTCalc to ensure compatibility.
    """
    # Setup paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    src_xyz = os.path.join(test_dir, "data", "test_data.xyz")
    # src_model = os.path.join(test_dir, 'data', 'mace_test.model')

    config_file = tmp_path / "test_data.xyz"
    # model_file = tmp_path / "mace_test.model"

    shutil.copy2(src_xyz, config_file)
    # shutil.copy2(src_model, model_file)

    # Change directory to tmp_path to capture outputs
    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Instantiate strategies
        # mlip_name should be 'mace_test' so it looks for 'mace_test.model'
        # We don't need the actual model file if we mock get_calculator
        mlip_strategy = MACEModel(mlip_name="mace_test", run_mode="local")

        # Mock get_calculator to return EMT calculator
        # The signature of get_calculator is (job_name, dispersion=True, ...)
        # We return (EMT, [], kwargs) so wfl uses EMT(**kwargs)
        mlip_strategy.remote_info = None
        # Use generic parameters for EMT
        # Use EMTCalc().get_calculator for the mock side_effect to match the return signature
        mlip_strategy.get_calculator = MagicMock(side_effect=EMTCalc().get_calculator)

        # MDGen with minimal steps
        structure_gen_strategy = MDGen(
            uncertainty_thrs=float(
                "inf"
            ),  # Infinite threshold to ensure MD runs even with bad random structures
            n_failed_steps=2,
            params={
                "steps": 5,
                "dt": 1.0,
                "temperature": 300.0,
                "traj_step_interval": 1,
            },
        )

        # QChem strategy (dummy)
        qchem_strategy = EMTCalc()

        # Constraints
        # Fix first atom (index 0)
        constraints = [FixAtoms(indices=[0])]

        # Reaction constraints for NEB
        # tFUR+2H -> tFURao+H
        rxn_constraints_dict = {"tFUR+2H -> tFURao+H": constraints}

        calculation_kwargs = {
            "initial_sampling": {
                "basin_constraints": constraints,
                "neb_config": {
                    "rxn_constraints_dict": rxn_constraints_dict,
                    "method": "random",
                    "n_pathways": 1,
                    "n_images": 3,
                },
                "neb__structure_gen_params": {
                    "steps": 5,
                    "dt": 1.0,
                    "temperature": 300.0,
                },
            },
            "mlip_gen": {
                "dispersion": False
            },  # Disable dispersion to avoid dftd3 dependency issues if any
            "mlip_sp": {"dispersion": False},
            "fps_selection": {
                "descriptor_string": "soap n_species=4 species_Z={1 6 8 29} l_max=6 n_max=8 cutoff=3.5 atom_sigma=0.5 zeta=6",
                "info_field": "MACE_energy",
                "n_optimal": 5,
            },
            "relabel_check": {
                "reference_configs": config_file,
                "idx_org": slice(64, None),
                "idx_h": [75, 76],
                "sigma_isomer": 50,
                "sigma_reaction": 10,
                "threshold_reaction": 0.85,
            },
            "rematch": {
                "energy_tol": 0.05,
                "network_threshold": 0.95,
                "resolution": 1.0,
                "gamma": 0.1,
                "zeta": 1,
                "block_size": 512,
            },
        }

        state = EnsembleState(
            configs=["test_data.xyz"],
            qchem_strategy=qchem_strategy,
            mlip_strategy=mlip_strategy,
            structure_gen_strategy=structure_gen_strategy,
            calculation_kwargs=calculation_kwargs,
        )

        # Compile and run
        # We mock run_rematch_basin_collapse to avoid loading the real MACE model on complex real-world test data
        def mock_rematch_basin_collapse(state):
            from wfl.configset import ConfigSet, OutputSpec

            from mlipflow.utils.path_factory import resolve_step_path

            step_counter = state.get("step_counter", 1)
            current_iter = state.get("iteration", 0)
            out_file = resolve_step_path(
                step_suffix="collapsed_basin_configs",
                iteration=current_iter,
                step_counter=step_counter,
            )
            OutputSpec(out_file).write(ConfigSet(state["configs"]))
            return {**state, "configs": [out_file], "step_counter": step_counter + 1}

        def mock_apply_basin_constraints(state):
            from wfl.configset import ConfigSet, OutputSpec

            from mlipflow.utils.path_factory import resolve_step_path

            step_counter = state.get("step_counter", 1)
            current_iter = state.get("iteration", 0)
            out_file = resolve_step_path(
                step_suffix="basin_md",
                iteration=current_iter,
                step_counter=step_counter,
            )
            OutputSpec(out_file).write(ConfigSet(state["configs"]))
            return {**state, "configs": [out_file], "step_counter": step_counter + 1}

        def mock_generate_neb_pairs(state):
            from wfl.configset import ConfigSet, OutputSpec

            from mlipflow.utils.path_factory import resolve_step_path

            step_counter = state.get("step_counter", 1)
            current_iter = state.get("iteration", 0)

            original_configs = state["configs"]

            out_file = resolve_step_path(
                step_suffix="neb_md", iteration=current_iter, step_counter=step_counter
            )
            OutputSpec(out_file).write(ConfigSet(state["configs"]))
            return {
                **state,
                "configs": [out_file],
                "original_configs": original_configs,
                "step_counter": step_counter + 1,
            }

        with (
            patch(
                "mlipflow.graphflow.graphs.run_rematch_basin_collapse",
                side_effect=mock_rematch_basin_collapse,
            ),
            patch(
                "mlipflow.graphflow.graphs.run_apply_basin_constraints",
                side_effect=mock_apply_basin_constraints,
            ),
            patch(
                "mlipflow.graphflow.graphs.run_generate_neb_pairs",
                side_effect=mock_generate_neb_pairs,
            ),
        ):
            result = execute_initial_basin_pathsampling_md_block().invoke(state)

        # Verify
        assert result["outfile"] is None
        assert result["configs"] is not None
        assert len(result["configs"]) > 0

        # Check files
        for f in result["configs"]:
            assert os.path.exists(f)
            assert f.endswith("final_selection.xyz")

            atoms = read(f, ":")
            assert len(atoms) > 0

            # Check for MACE results
            # run_mlip_sp uses output_prefix=mlip.mlip_prefix which is 'MACE_'
            for at in atoms:
                assert "MACE_energy" in at.info
                # For EMT, forces are computed.
                # Note: 'MACE_forces' might be missing if single_point didn't run properly?
                # But run_mlip_sp calls run_single_point which requests properties=["energy", "forces"]
                assert "MACE_forces" in at.arrays

    finally:
        os.chdir(cwd)


def test_execute_opt_neb_combination_block_rattle_and_unknown_bypass(tmp_path):
    """End-to-end test of execute_opt_neb_combination_block with rattle + unknown bypass.

    Verifies:
    - Active atoms are rattled prior to OPTGen while constrained slab atoms stay fixed.
    - Novel/rearranged 'unknown' frames bypass basin_collapse and NEB pairing.
    - 'unknown' frames survive into merged.xyz, run_mlip_sp, and uncertainty selection.
    """
    src_xyz = Path(__file__).resolve().parent / "data" / "test_data.xyz"
    ref_atoms = read(src_xyz, ":")

    # Pick 2 known frames + 1 rearranged 'unknown' frame (H-H < 1.0 A -> H2_gas)
    known_1 = next(at.copy() for at in ref_atoms if at.info["species"] == "tFUR+2H")
    known_2 = next(at.copy() for at in ref_atoms if at.info["species"] == "tFURao+H")
    unknown_frame = known_1.copy()
    pos = unknown_frame.get_positions()
    pos[76] = pos[75] + np.array([0.74, 0.0, 0.0])
    unknown_frame.set_positions(pos)
    unknown_frame.info["origin_tag"] = "novel_unknown"

    initial_frames = [known_1, known_2, unknown_frame]
    in_xyz = tmp_path / "iter1_input.xyz"
    write(in_xyz, initial_frames)

    fixed_indices = list(range(48))
    constraints = [FixAtoms(indices=fixed_indices)]

    mlip_strategy = MACEModel(mlip_name="mace_test", run_mode="local")
    mlip_strategy.remote_info = None
    mlip_strategy.get_calculator = MagicMock(side_effect=EMTCalc().get_calculator)

    state = EnsembleState(
        configs=[str(in_xyz)],
        iteration=1,
        qchem_strategy=EMTCalc(),
        mlip_strategy=mlip_strategy,
        structure_gen_strategy=OPTGen(params={"fmax": 0.05, "steps": 2}),
        calculation_kwargs={
            "initial_sampling": {"basin_constraints": constraints},
            "thermal_perturb": {"sigma": 0.15, "seed": 42},
            "mlip_gen": {"dispersion": False},
            "mlip_sp": {"dispersion": False},
            "relabel_check": {
                "reference_configs": str(src_xyz),
                "idx_org": slice(64, None),
                "idx_h": [75, 76],
                "sigma_isomer": 50,
                "sigma_reaction": 10,
                "threshold_reaction": 0.85,
            },
        },
    )

    pre_opt_files: list[str] = []
    basin_input_species: list[str] = []
    neb_input_species: list[str] = []
    selection_input_files: list[str] = []

    def spy_run_optimisation(es):
        pre_opt_files.extend(es["configs"])
        return run_mlip_structure_generation(es)

    def mock_basin_collapse(es):
        basin_input_species.extend(
            at.info.get("species", "") for at in ConfigSet(es["configs"])
        )
        step_counter = es.get("step_counter", 1)
        out_file = resolve_step_path(
            step_suffix="collapsed_basin_configs",
            iteration=es.get("iteration", 1),
            step_counter=step_counter,
        )
        OutputSpec(out_file).write(ConfigSet(es["configs"]))
        return {
            **es,
            "pre_collapsed_configs": es["configs"],
            "configs": [out_file],
            "step_counter": step_counter + 1,
        }

    def mock_gen_neb_pairs(es):
        neb_input_species.extend(
            at.info.get("species", "") for at in ConfigSet(es["configs"])
        )
        step_counter = es.get("step_counter", 1)
        out_file = resolve_step_path(
            step_suffix="neb_opt",
            iteration=es.get("iteration", 1),
            step_counter=step_counter,
        )
        OutputSpec(out_file).write(ConfigSet(es["configs"]))
        return {
            **es,
            "original_configs": es["configs"],
            "configs": [out_file],
            "step_counter": step_counter + 1,
        }

    def spy_uncertainty_selection(es):
        selection_input_files.extend(es["configs"])
        return {**es, "step_counter": es.get("step_counter", 1) + 1}

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with (
            patch(
                "mlipflow.graphflow.graphs.run_mlip_structure_generation",
                side_effect=spy_run_optimisation,
            ),
            patch(
                "mlipflow.graphflow.graphs.run_rematch_basin_collapse",
                side_effect=mock_basin_collapse,
            ),
            patch(
                "mlipflow.graphflow.graphs.run_generate_neb_pairs",
                side_effect=mock_gen_neb_pairs,
            ),
            patch(
                "mlipflow.graphflow.graphs.run_config_uncertainty_selection",
                side_effect=spy_uncertainty_selection,
            ),
        ):
            result = execute_opt_neb_combination_block().invoke(state)

        # 1. Verify rattle perturbation ran BEFORE OPTGen and respected FixAtoms
        assert len(pre_opt_files) == 1
        assert "rattle_perturb" in Path(pre_opt_files[0]).name
        pre_opt_atoms = read(pre_opt_files[0], ":")
        assert len(pre_opt_atoms) == len(initial_frames)
        for orig, rattled in zip(initial_frames, pre_opt_atoms):
            disp = np.linalg.norm(
                rattled.get_positions() - orig.get_positions(), axis=1
            )
            assert np.all(disp[:48] == 0.0)
            assert np.all(disp[48:] > 0.0)

        # 2. Verify basin_collapse and NEBPairFinder never saw 'unknown' frames
        assert len(basin_input_species) > 0
        assert "unknown" not in basin_input_species
        assert len(neb_input_species) > 0
        assert "unknown" not in neb_input_species

        # 3. Verify 'unknown' frame survived into merged.xyz -> mlip_sp -> selection
        merged_files = sorted(Path("iterations/iter_1").glob("*_merged.xyz"))
        assert len(merged_files) == 1
        merged_atoms = read(merged_files[0], ":")
        assert any(at.info.get("origin_tag") == "novel_unknown" for at in merged_atoms)
        assert any(at.info.get("species") == "unknown" for at in merged_atoms)

        assert len(selection_input_files) == 1
        sp_atoms = read(selection_input_files[0], ":")
        unknown_sp = [
            at for at in sp_atoms if at.info.get("origin_tag") == "novel_unknown"
        ]
        assert len(unknown_sp) == 1
        assert "MACE_energy" in unknown_sp[0].info
        assert "MACE_forces" in unknown_sp[0].arrays
        assert result["unknown_configs"]
    finally:
        os.chdir(cwd)
