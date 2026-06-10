from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlipflow.graphflow.nodes import (
    EnsembleState,
    merge_configs,
    run_dft_sp,
    run_dft_sp_chunked,
    run_generate_neb_pairs,
    run_mlip_sp,
    run_mlip_structure_generation,
    run_topology_relabel,
    switch_to_neb_generation,
)
from mlipflow.strategies.structure_generators import MDGen, NEBGen


def test_merge_configs_value_error():
    """Test merge_configs raises ValueError if original_configs is missing."""
    state = EnsembleState(configs=["c1.xyz"])
    with pytest.raises(
        ValueError, match="No original configurations provided in state"
    ):
        merge_configs(state)


def test_switch_to_neb_generation():
    """Test switch_to_neb_generation correctly updates strategy and parameters."""
    state = EnsembleState(
        configs=["c1.xyz"],
        structure_gen_strategy=MDGen(params={"steps": 10}),
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
    mock_resolve.assert_called_once_with("input.xyz", "dft_sp", 2)


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
    mock_resolve.assert_called_once_with("input.xyz", "dft_sp", 2)


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
    mock_resolve.assert_called_once_with("input.xyz", "mace_sp", 1)


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
    mock_resolve.assert_called_once_with("input.xyz", "md", 3)


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
def test_run_topology_relabel_basic(mock_output_spec, mock_relabel, tmp_path):
    """Test run_topology_relabel normal path logic."""
    ref_file = tmp_path / "ref.xyz"
    ref_file.touch()

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

    # Check that OutputSpec was called to write the output configurations
    mock_output_spec.assert_any_call("known_configs.xyz")
    mock_output_spec.assert_any_call("unknown_configs.xyz")
