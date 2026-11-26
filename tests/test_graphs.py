from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixAtoms
from ase.io import read, write
from mace.calculators import MACECalculator
from wfl.configset import ConfigSet, OutputSpec

from mlipflow.data.llpr import PerElementLLPR
from mlipflow.graphflow.activelearner import route_mlip_training
from mlipflow.graphflow.graphs import (
    execute_initial_basin_pathsampling_md_block,
    execute_mlip_training_block,
    execute_opt_neb_combination_block,
)
from mlipflow.graphflow.nodes import (
    EnsembleState,
    run_mlip_structure_generation,
    run_post_fit_calibration,
)
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


def test_execute_mlip_training_block_post_fit_calibration(tmp_path, caplog):
    """End-to-end execute_mlip_training_block test with real mace_test.model and Cu(100) test_data.xyz.

    Verifies:
    - Fresh v_k forces are evaluated even when stale bogus MACE_forces exist on test_configs.
    - {model_name}_v{iter}_llpr.pt is written and reloadable via torch.load(..., weights_only=True).
    - Per-element MAE, p90, MaxErr, and Spearman are logged and stored in state['element_metrics'].
    - state['uq_file'], state['last_training_configs'], and state['last_test_configs'] are populated.
    - dispersion=True wraps MACECalculator + TorchDFTD3Calculator in SumCalculator.
    """
    data_dir = Path(__file__).resolve().parent / "data"
    src_model = data_dir / "mace_test.model"
    src_xyz = data_dir / "test_data.xyz"

    all_frames = read(src_xyz, ":")
    atom_slice = slice(48, None)

    real_calc = MACECalculator(
        model_paths=str(src_model), device="cpu", default_dtype="float32"
    )
    train_frames = all_frames[:12]
    test_frames = [at.copy() for at in all_frames[12:]]
    rng = np.random.default_rng(21)
    for at in test_frames:
        at_c = at.copy()
        at_c.calc = real_calc
        f_true_mace = at_c.get_forces()
        # True DFT forces are close to v_k MACE forces (~0.04 eV/A error)
        at.arrays["DFT_forces"] = f_true_mace + rng.normal(
            0.0, 0.04, size=f_true_mace.shape
        )
        # Inject stale v_{k-1} MACE_forces with huge bogus offset (+50 eV/A)
        at.arrays["MACE_forces"] = f_true_mace + 50.0

    train_xyz = tmp_path / "07_dft_train.xyz"
    test_xyz = tmp_path / "07_dft_test.xyz"
    write(train_xyz, train_frames)
    write(test_xyz, test_frames)

    mlip_strategy = MACEModel(mlip_name="pot_v0", run_mode="local")

    def _fake_fit_new_model(in_file, test_configs, **_kwargs):
        _ = in_file, test_configs
        shutil.copy2(src_model, mlip_strategy.model_file)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        state = EnsembleState(
            configs=[str(train_xyz)],
            outfile=[str(test_xyz)],
            iteration=0,
            mlip_strategy=mlip_strategy,
            calculation_kwargs={
                "llpr_calibration": {
                    "dispersion": False,
                    "atom_indices": atom_slice,
                    "exclude_fixed": False,
                    "tune_lambdas": (1e-4,),
                }
            },
        )

        caplog.clear()
        with (
            patch.object(
                mlip_strategy, "fit_new_model", side_effect=_fake_fit_new_model
            ),
            caplog.at_level("INFO"),
        ):
            result = execute_mlip_training_block().invoke(state)

        uq_path = Path(result["uq_file"])
        assert uq_path.name == "pot_v0_llpr.pt"
        assert uq_path.exists()
        assert uq_path.stat().st_size > 0

        # Verify weights_only=True reloadability
        raw_payload = torch.load(str(uq_path), map_location="cpu", weights_only=True)
        assert isinstance(raw_payload, dict)
        reloaded = PerElementLLPR.load(uq_path, device="cpu")
        assert set(reloaded.calibrations.keys()) == {0.50, 0.20, 0.10}

        # Fresh v_k forces must have been evaluated (MAE << 1.0, NOT ~50 eV/A from stale MACE_forces)
        assert result["last_training_configs"] == [str(train_xyz)]
        assert result["last_test_configs"] == [str(test_xyz)]
        elem_metrics = result["element_metrics"]
        assert set(elem_metrics.keys()) == {1, 6, 8, 29}
        for z, stats in elem_metrics.items():
            assert stats["mae"] < 1.0, (
                f"Stale MACE_forces used for Z={z}: MAE={stats['mae']}"
            )
            assert stats["p90"] >= stats["mae"]
            assert stats["max_err"] >= stats["p90"]
            assert "spearman" in stats

        # Verify per-element metrics were logged
        log_text = "\n".join(rec.message for rec in caplog.records)
        assert "MAE" in log_text
        assert "p90" in log_text
        assert "MaxErr" in log_text

        # Verify dispersion=True wraps MACECalculator + TorchDFTD3Calculator in SumCalculator
        captured_force_calcs = []
        orig_calibrate = PerElementLLPR.calibrate

        def _spy_calibrate(self, *args, **kwargs):
            captured_force_calcs.append(kwargs.get("force_calc"))
            # Avoid running real D3 forces here by swapping force_calc to mace_calc after inspection
            kwargs["force_calc"] = kwargs.get("mace_calc")
            return orig_calibrate(self, *args, **kwargs)

        state_disp = EnsembleState(
            configs=[str(train_xyz)],
            outfile=[str(test_xyz)],
            iteration=0,
            mlip_strategy=mlip_strategy,
            calculation_kwargs={
                "llpr_calibration": {
                    "dispersion": True,
                    "atom_indices": atom_slice,
                    "exclude_fixed": False,
                    "tune_lambdas": (1e-4,),
                }
            },
        )
        with (
            patch.object(
                mlip_strategy, "fit_new_model", side_effect=_fake_fit_new_model
            ),
            patch.object(PerElementLLPR, "calibrate", new=_spy_calibrate),
        ):
            execute_mlip_training_block().invoke(state_disp)

        assert len(captured_force_calcs) == 1
        assert isinstance(captured_force_calcs[0], SumCalculator)
        assert len(captured_force_calcs[0].mixer.calcs) == 2
        assert result["model_file"] == mlip_strategy.model_file
    finally:
        os.chdir(cwd)


def test_execute_mlip_training_block_skips_calibration_on_missing_model(tmp_path):
    """Verify calibration skips gracefully when run_mace_fit fails to produce model_file."""
    mlip_strategy = MACEModel(mlip_name="pot_v0", run_mode="local")
    mlip_strategy.fit_new_model = MagicMock()  # writes nothing to disk

    state = EnsembleState(
        configs=["train.xyz"],
        outfile=["test.xyz"],
        iteration=0,
        mlip_strategy=mlip_strategy,
    )

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = execute_mlip_training_block().invoke(state)
        assert "model_file" not in result
        assert "uq_file" not in result
        assert route_mlip_training({"ensemble_state": result}) == "loop"
    finally:
        os.chdir(cwd)


def test_run_post_fit_calibration_raises_on_missing_datasets(tmp_path):
    """Verify fail-fast ValueError when model_file exists but train/test configs are missing."""
    model_file = tmp_path / "pot_v0.model"
    model_file.write_text("trained weights", encoding="utf-8")
    mlip_strategy = MACEModel(mlip_name=str(tmp_path / "pot_v0"), run_mode="local")

    state_no_test = EnsembleState(
        configs=["train.xyz"],
        last_training_configs=["train.xyz"],
        last_test_configs=[],
        model_file=str(model_file),
        mlip_strategy=mlip_strategy,
    )
    with pytest.raises(ValueError, match="test"):
        run_post_fit_calibration(state_no_test)

    state_no_train = EnsembleState(
        configs=[],
        last_training_configs=[],
        last_test_configs=["test.xyz"],
        model_file=str(model_file),
        mlip_strategy=mlip_strategy,
    )
    with pytest.raises(ValueError, match="training"):
        run_post_fit_calibration(state_no_train)
