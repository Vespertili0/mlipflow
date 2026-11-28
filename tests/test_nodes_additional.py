from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

from mlipflow.graphflow.nodes import (
    EnsembleState,
    merge_configs,
    run_dft_sp,
    run_dft_sp_chunked,
    run_generate_neb_pairs,
    run_mace_fit,
    run_mlip_sp,
    run_mlip_structure_generation,
    run_rattle_perturbation,
    run_topology_relabel,
    switch_to_neb_generation,
)
from mlipflow.strategies.mlip import MACEModel
from mlipflow.strategies.structure_generators import MDGen, NEBGen, OPTGen


def test_merge_configs_value_error():
    """Test merge_configs raises ValueError if original_configs is missing."""
    state = EnsembleState(configs=["c1.xyz"])
    with pytest.raises(
        ValueError,
        match="No original or pre-collapsed configurations provided in state",
    ):
        merge_configs(state)


def test_switch_to_neb_generation():
    """Test switch_to_neb_generation correctly updates strategy and parameters."""
    state = EnsembleState(
        configs=["c1.xyz"],
        structure_gen_strategy=MDGen(params={"steps": 10}, uncertainty_thrs=0.1),
        calculation_kwargs={
            "mlip_gen": {"dispersion": True},
            "neb__mlip_gen": {"dispersion": False},
            "neb__structure_gen_params": {"dt": 0.5},
        },
    )

    new_state = switch_to_neb_generation(state)
    assert isinstance(new_state["structure_gen_strategy"], NEBGen)
    assert new_state["calculation_kwargs"]["mlip_gen"] == {"dispersion": False}
    assert new_state["structure_gen_strategy"].params == {"dt": 0.5}


@patch("mlipflow.graphflow.nodes.run_single_point")
@patch("mlipflow.graphflow.nodes.resolve_step_path")
def test_run_dft_sp_idempotency(mock_resolve, mock_run, tmp_path):
    """Test run_dft_sp skip execution when outputs exist (idempotency)."""
    out_file = tmp_path / "mocked_out.xyz"
    out_file.write_text("dummy xyz content")

    mock_resolve.return_value = str(out_file)

    state = EnsembleState(configs=["input.xyz"], iteration=2)

    new_state = run_dft_sp(state)
    # Check that it returns expected state without running
    assert new_state["configs"] == [str(out_file)]
    assert new_state["outfile"] is None
    mock_run.assert_not_called()
    mock_resolve.assert_called_once_with(
        step_suffix="dft_sp", iteration=2, step_counter=1
    )


@patch("mlipflow.graphflow.nodes.run_chunked_qe_sp")
@patch("mlipflow.graphflow.nodes.resolve_step_path")
def test_run_dft_sp_chunked_idempotency(mock_resolve, mock_run_chunked, tmp_path):
    """Test run_dft_sp_chunked skip execution when outputs exist (idempotency)."""
    out_file = tmp_path / "mocked_out.xyz"
    out_file.write_text("dummy xyz content")

    mock_resolve.return_value = str(out_file)

    state = EnsembleState(configs=["input.xyz"], iteration=2)

    new_state = run_dft_sp_chunked(state)
    # Check that it returns expected state without running
    assert new_state["configs"] == [str(out_file)]
    assert new_state["outfile"] is None
    mock_run_chunked.assert_not_called()
    mock_resolve.assert_called_once_with(
        step_suffix="dft_sp", iteration=2, step_counter=1
    )


@patch("mlipflow.graphflow.nodes.run_single_point")
@patch("mlipflow.graphflow.nodes.resolve_step_path")
def test_run_mlip_sp_idempotency(mock_resolve, mock_run, tmp_path):
    """Test run_mlip_sp skip execution when outputs exist (idempotency)."""
    out_file = tmp_path / "mocked_out.xyz"
    out_file.write_text("dummy xyz content")

    mock_resolve.return_value = str(out_file)
    mlip_mock = MagicMock()
    mlip_mock.mlip_prefix = "MACE_"

    state = EnsembleState(configs=["input.xyz"], mlip_strategy=mlip_mock, iteration=1)

    new_state = run_mlip_sp(state)
    assert new_state["configs"] == [str(out_file)]
    assert new_state["outfile"] is None
    mock_run.assert_not_called()
    mock_resolve.assert_called_once_with(
        step_suffix="mace_sp", iteration=1, step_counter=1
    )


@patch("mlipflow.graphflow.nodes.resolve_step_path")
def test_run_mlip_structure_generation_idempotency(mock_resolve, tmp_path):
    """Test structure generation skip execution when outputs exist (idempotency)."""
    out_file = tmp_path / "mocked_out.xyz"
    out_file.write_text("dummy xyz content")

    mock_resolve.return_value = str(out_file)

    mlip_mock = MagicMock()
    strategy_mock = MagicMock()
    strategy_mock.calc_prefix = "md"
    strategy_mock.params = {}

    state = EnsembleState(
        configs=["input.xyz"],
        mlip_strategy=mlip_mock,
        structure_gen_strategy=strategy_mock,
        iteration=3,
    )

    new_state = run_mlip_structure_generation(state)
    assert new_state["configs"] == [str(out_file)]
    assert new_state["outfile"] is None
    strategy_mock.generate_new_structures.assert_not_called()
    mock_resolve.assert_called_once_with(
        step_suffix="structure_md", iteration=3, step_counter=1
    )


def test_run_generate_neb_pairs_missing_neb_config():
    """Test run_generate_neb_pairs raises ValueError when neb_config is missing."""
    state = EnsembleState(
        configs=["input.xyz"],
        structure_gen_strategy=NEBGen(params={}),
        calculation_kwargs={},
    )
    with pytest.raises(
        ValueError,
        match="neb_config with 'rxn_constraints_dict' is required for NEB pair generation.",
    ):
        run_generate_neb_pairs(state)


def test_run_topology_relabel_missing_reference_configs():
    """Test run_topology_relabel raises ValueError when reference_configs is missing."""
    state = EnsembleState(
        configs=["input.xyz"], calculation_kwargs={"relabel_check": {}}
    )
    with pytest.raises(ValueError, match="reference_configs not found in gcml_kwargs"):
        run_topology_relabel(state)


@patch("mlipflow.graphflow.nodes.relabel_configs")
@patch("mlipflow.graphflow.nodes.OutputSpec")
@patch("mlipflow.graphflow.nodes.resolve_step_path")
def test_run_topology_relabel_basic(
    mock_resolve, mock_output_spec, mock_relabel, tmp_path
):
    """Test run_topology_relabel normal path logic."""
    ref_file = tmp_path / "ref.xyz"
    ref_file.touch()

    # Mock resolve_step_path to return simple file names for testing
    mock_resolve.side_effect = lambda *args, **kwargs: (
        f"{kwargs.get('step_suffix')}.xyz"
    )

    # Mock relabel_configs to return list of Atoms
    atoms1 = Atoms("H")
    atoms2 = Atoms("He")
    mock_relabel.return_value = ([atoms1], [atoms2])

    state = EnsembleState(
        configs=["input.xyz"],
        calculation_kwargs={
            "relabel_check": {
                "reference_configs": str(ref_file),
                "idx_org": slice(0, None),
                "idx_h": [0],
            }
        },
    )

    new_state = run_topology_relabel(state)
    assert new_state["configs"] == ["known_configs.xyz"]
    assert new_state["unknown_configs"] == ["unknown_configs.xyz"]
    assert new_state["step_counter"] == 2

    # Check that OutputSpec was called to write the output configurations
    mock_output_spec.assert_any_call("known_configs.xyz")
    mock_output_spec.assert_any_call("unknown_configs.xyz")


def test_run_rattle_perturbation_respects_fixatoms_and_seed(tmp_path):
    """Test rattle perturbation displaces active atoms while keeping FixAtoms fixed."""
    src_xyz = Path(__file__).resolve().parent / "data" / "test_data.xyz"
    orig_atoms = read(src_xyz, ":4")
    fixed_indices = list(range(48))
    for at in orig_atoms:
        at.set_constraint(FixAtoms(indices=fixed_indices))

    in_file = tmp_path / "input.xyz"
    write(in_file, orig_atoms)

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        state_a = EnsembleState(
            configs=[str(in_file)],
            iteration=1,
            step_counter=1,
            calculation_kwargs={"thermal_perturb": {"sigma": 0.15, "seed": 42}},
        )
        res_a = run_rattle_perturbation(state_a)
        assert res_a["step_counter"] == 2
        out_a = Path(res_a["configs"][0])
        assert out_a.exists()
        assert "rattle_perturb" in out_a.name

        perturbed_a = read(out_a, ":")
        assert len(perturbed_a) == len(orig_atoms)

        for orig, pert in zip(orig_atoms, perturbed_a):
            disp = np.linalg.norm(pert.get_positions() - orig.get_positions(), axis=1)
            # Fixed slab atoms must have exact zero displacement
            assert np.all(disp[:48] == 0.0)
            # Active atoms must be displaced
            assert np.all(disp[48:] > 0.0)
            # FixAtoms must survive extxyz serialization for downstream OPTGen
            assert len(pert.constraints) > 0
            assert set(pert.constraints[0].get_indices()) == set(fixed_indices)

        # Same iteration + seed reproduces bit-for-bit
        out_a.unlink()
        res_a_repeat = run_rattle_perturbation(state_a)
        perturbed_repeat = read(res_a_repeat["configs"][0], ":")
        for p1, p2 in zip(perturbed_a, perturbed_repeat):
            np.testing.assert_allclose(p1.get_positions(), p2.get_positions())

        # Subsequent iteration uses offset seed -> distinct perturbation
        state_iter2 = EnsembleState(
            configs=[str(in_file)],
            iteration=2,
            step_counter=1,
            calculation_kwargs={"thermal_perturb": {"sigma": 0.15, "seed": 42}},
        )
        res_iter2 = run_rattle_perturbation(state_iter2)
        perturbed_iter2 = read(res_iter2["configs"][0], ":")
        assert not np.allclose(
            perturbed_a[0].get_positions()[48:], perturbed_iter2[0].get_positions()[48:]
        )
    finally:
        os.chdir(cwd)


def test_run_rattle_perturbation_basin_constraints_fallback(tmp_path):
    """Test mandatory fallback to initial_sampling.basin_constraints when frames have no constraints."""
    src_xyz = Path(__file__).resolve().parent / "data" / "test_data.xyz"
    orig_atoms = read(src_xyz, ":3")
    assert all(len(at.constraints) == 0 for at in orig_atoms)

    in_file = tmp_path / "unconstrained.xyz"
    write(in_file, orig_atoms)

    fixed_indices = list(range(48))
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        state = EnsembleState(
            configs=[str(in_file)],
            iteration=1,
            step_counter=1,
            calculation_kwargs={
                "initial_sampling": {
                    "basin_constraints": [FixAtoms(indices=fixed_indices)]
                }
            },
        )
        res = run_rattle_perturbation(state)
        perturbed = read(res["configs"][0], ":")

        for orig, pert in zip(orig_atoms, perturbed):
            disp = np.linalg.norm(pert.get_positions() - orig.get_positions(), axis=1)
            assert np.all(disp[:48] == 0.0)
            assert np.all(disp[48:] > 0.0)
            assert len(pert.constraints) > 0
            assert set(pert.constraints[0].get_indices()) == set(fixed_indices)
    finally:
        os.chdir(cwd)


def test_run_rattle_perturbation_idempotency_and_validation(tmp_path):
    """Test idempotency guard and input validation on run_rattle_perturbation."""
    with pytest.raises(ValueError, match="No configurations provided in state"):
        run_rattle_perturbation(EnsembleState(configs=[]))

    src_xyz = Path(__file__).resolve().parent / "data" / "test_data.xyz"
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError, match="sigma"):
            run_rattle_perturbation(
                EnsembleState(
                    configs=[str(src_xyz)],
                    calculation_kwargs={"thermal_perturb": {"sigma": -0.1}},
                )
            )

        out_dir = tmp_path / "iterations" / "iter_1"
        out_dir.mkdir(parents=True, exist_ok=True)
        existing_file = out_dir / "01_rattle_perturb.xyz"
        existing_file.write_text("existing perturbed content", encoding="utf-8")

        res = run_rattle_perturbation(
            EnsembleState(configs=[str(src_xyz)], iteration=1, step_counter=1)
        )
        assert res["configs"] == [str(Path("iterations/iter_1/01_rattle_perturb.xyz"))]
        assert res["step_counter"] == 2
        assert existing_file.read_text(encoding="utf-8") == "existing perturbed content"
    finally:
        os.chdir(cwd)


def test_run_topology_relabel_preserves_and_accumulates_unknown_configs(tmp_path):
    """Test real GaussianCMLabeller routes rearranged unknown frames to state['unknown_configs']."""
    src_xyz = Path(__file__).resolve().parent / "data" / "test_data.xyz"
    ref_atoms = read(src_xyz, ":")

    # Take 2 known frames + 1 rearranged frame (H-H < 1.0 A -> H2_gas -> 'unknown')
    known_1 = ref_atoms[0].copy()
    known_2 = ref_atoms[1].copy()
    unknown_frame = ref_atoms[0].copy()
    pos = unknown_frame.get_positions()
    pos[76] = pos[75] + np.array([0.74, 0.0, 0.0])
    unknown_frame.set_positions(pos)
    unknown_frame.info["tag_id"] = "novel_h2"

    in_file = tmp_path / "mixed_input.xyz"
    write(in_file, [known_1, known_2, unknown_frame])

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        state = EnsembleState(
            configs=[str(in_file)],
            iteration=1,
            step_counter=1,
            calculation_kwargs={
                "relabel_check": {
                    "reference_configs": str(src_xyz),
                    "idx_org": slice(64, None),
                    "idx_h": [75, 76],
                    "sigma_isomer": 50,
                    "sigma_reaction": 10,
                    "threshold_reaction": 0.85,
                }
            },
        )
        res_1 = run_topology_relabel(state)

        known_out = read(res_1["configs"][0], ":")
        assert len(known_out) == 2
        assert all(at.info["species"] != "unknown" for at in known_out)

        assert len(res_1["unknown_configs"]) == 1
        unknown_out = read(res_1["unknown_configs"][0], ":")
        assert len(unknown_out) == 1
        assert unknown_out[0].info["species"] == "unknown"
        assert unknown_out[0].info["tag_id"] == "novel_h2"

        # Second relabel pass in same iteration accumulates rather than overwriting
        res_2 = run_topology_relabel({**res_1, "configs": [str(in_file)]})
        assert len(res_2["unknown_configs"]) == 2
        assert res_2["unknown_configs"][0] != res_2["unknown_configs"][1]
    finally:
        os.chdir(cwd)


def test_merge_configs_includes_unknown_configs_and_handles_empty(tmp_path):
    """Test merge_configs includes non-empty unknown_configs and skips empty files gracefully."""
    src_xyz = Path(__file__).resolve().parent / "data" / "test_data.xyz"
    frames = read(src_xyz, ":4")

    known_a = tmp_path / "pre_collapsed.xyz"
    neb_b = tmp_path / "neb_out.xyz"
    unknown_c = tmp_path / "unknown_nonempty.xyz"
    unknown_empty = tmp_path / "unknown_empty.xyz"

    frames[0].info["origin"] = "pre_0"
    frames[1].info["origin"] = "pre_1"
    frames[2].info["origin"] = "neb_0"
    frames[3].info["origin"] = "unknown_0"
    frames[3].info["species"] = "unknown"

    write(known_a, frames[:2])
    write(neb_b, [frames[2]])
    write(unknown_c, [frames[3]])
    unknown_empty.write_text("", encoding="utf-8")

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        # Case 1: Mix of non-empty and empty unknown_configs files
        state = EnsembleState(
            pre_collapsed_configs=[str(known_a)],
            configs=[str(neb_b)],
            unknown_configs=[str(unknown_c), str(unknown_empty)],
            iteration=1,
            step_counter=5,
        )
        merged_state = merge_configs(state)
        merged_atoms = read(merged_state["configs"][0], ":")
        assert len(merged_atoms) == 4
        assert {at.info["origin"] for at in merged_atoms} == {
            "pre_0",
            "pre_1",
            "neb_0",
            "unknown_0",
        }
        assert any(at.info["species"] == "unknown" for at in merged_atoms)

        # Case 2: Only empty unknown_configs file (must not fail)
        state_empty = EnsembleState(
            original_configs=[str(known_a)],
            configs=[str(neb_b)],
            unknown_configs=[str(unknown_empty)],
            iteration=1,
            step_counter=6,
        )
        merged_empty = merge_configs(state_empty)
        merged_empty_atoms = read(merged_empty["configs"][0], ":")
        assert len(merged_empty_atoms) == 3
    finally:
        os.chdir(cwd)


def test_run_mace_fit_requires_nonempty_model_file(tmp_path):
    """Verify run_mace_fit only sets result['model_file'] when a non-empty model exists."""
    model_path = tmp_path / "pot_v0.model"
    mock_strategy = MagicMock()
    mock_strategy.model_file = str(model_path)

    # Missing model file after failed fit -> model_file must not be set
    state_missing = EnsembleState(
        configs=["train.xyz"],
        outfile=["test.xyz"],
        mlip_strategy=mock_strategy,
        model_file="stale.model",
    )
    res_missing = run_mace_fit(state_missing)
    assert "model_file" not in res_missing

    # Empty 0-byte model file -> model_file must not be set
    model_path.write_text("", encoding="utf-8")
    res_empty = run_mace_fit(
        EnsembleState(
            configs=["train.xyz"], outfile=["test.xyz"], mlip_strategy=mock_strategy
        )
    )
    assert "model_file" not in res_empty

    # Non-empty model file -> model_file is propagated
    model_path.write_text("trained weights", encoding="utf-8")
    res_ok = run_mace_fit(
        EnsembleState(
            configs=["train.xyz"], outfile=["test.xyz"], mlip_strategy=mock_strategy
        )
    )
    assert res_ok["model_file"] == str(model_path)


def test_structure_generation_compute_uq_defaults(tmp_path):
    """Verify OPTGen/NEBGen default to compute_uq=False (overridable) and MDGen defaults to True."""
    src_xyz = Path(__file__).resolve().parent / "data" / "test_data.xyz"
    in_file = tmp_path / "in.xyz"
    write(in_file, read(src_xyz, ":1"))

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        for step_idx, (gen_strat, mlip_gen_kw, expected_compute_uq) in enumerate(
            [
                (OPTGen(params={"steps": 1}), {}, False),
                (NEBGen(params={"steps": 1}), {}, False),
                (OPTGen(params={"steps": 1}), {"compute_uq": True}, True),
                (MDGen(uncertainty_thrs=10.0, params={"steps": 1}), {}, True),
            ],
            start=1,
        ):
            mlip = MACEModel(mlip_name="pot_v0", run_mode="local")
            captured_kw: dict = {}

            def _spy_get_calc(job_name, _captured=captured_kw, **kw):
                _captured.clear()
                _captured.update(kw)
                return ("dummy_calc", [], {})

            mlip.get_calculator = MagicMock(side_effect=_spy_get_calc)
            gen_strat.generate_new_structures = MagicMock(
                side_effect=lambda in_file, out_file, **_kw: write(
                    out_file[0] if isinstance(out_file, list) else out_file,
                    read(in_file[0] if isinstance(in_file, list) else in_file, ":"),
                )
            )

            state = EnsembleState(
                configs=[str(in_file)],
                iteration=1,
                step_counter=step_idx,
                mlip_strategy=mlip,
                structure_gen_strategy=gen_strat,
                uq_file="pot_v0_llpr.pt",
                calculation_kwargs={
                    "mlip_gen": mlip_gen_kw,
                    "uncertainty_selection": {
                        "atom_indices": slice(48, None),
                        "exclude_fixed": False,
                    },
                },
            )
            run_mlip_structure_generation(state)
            assert captured_kw["compute_uq"] is expected_compute_uq
            if expected_compute_uq:
                assert captured_kw["uq_file"] == "pot_v0_llpr.pt"
                assert captured_kw["atom_indices"] == slice(48, None)
                assert captured_kw["exclude_fixed"] is False
    finally:
        os.chdir(cwd)
