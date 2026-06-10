from __future__ import annotations

from unittest.mock import MagicMock, patch

from mlipflow.graphflow.activelearner import (
    _propagate_iteration,
    calculate_dft_level,
    check_mlip_training_completion,
    finalising_learning_loop,
    generate_new_structures,
    route_mlip_training,
    run_active_learning_loop,
    train_new_mlip_model,
)


def test_propagate_iteration():
    """Test _propagate_iteration helper.

    Verifies iteration index propagation and deep copy structure.
    """
    state = {
        "reaction_data": ["r1.xyz"],
        "training_data": ["t1.xyz"],
        "iteration": 3,
        "model_name": "pot",
        "ensemble_state": {"configs": ["c1.xyz"]},
    }

    es = _propagate_iteration(state)
    assert es["iteration"] == 3
    assert es["configs"] == ["c1.xyz"]
    # Verify deep copy was made
    assert es is not state["ensemble_state"]


@patch("mlipflow.graphflow.activelearner.execute_mlip_structure_generation_block")
def test_generate_new_structures(mock_block):
    """Test generate_new_structures node execution.

    Verifies propagation of state iteration down to the MD/OPT subgraph.
    """
    mock_invoke = MagicMock()
    mock_invoke.invoke.return_value = {"configs": ["gen_structs.xyz"], "iteration": 1}
    mock_block.return_value = mock_invoke

    state = {
        "reaction_data": [],
        "training_data": [],
        "iteration": 1,
        "model_name": "pot",
        "ensemble_state": {"configs": ["input.xyz"]},
    }

    result = generate_new_structures(state)
    assert result["ensemble_state"]["configs"] == ["gen_structs.xyz"]
    assert result["iteration"] == 1
    mock_invoke.invoke.assert_called_once()


@patch("mlipflow.graphflow.activelearner.execute_dft_single_point_block")
def test_calculate_dft_level(mock_block):
    """Test calculate_dft_level node execution.

    Verifies call propagation to the single-point DFT subgraph.
    """
    mock_invoke = MagicMock()
    mock_invoke.invoke.return_value = {"configs": ["dft_out.xyz"], "iteration": 2}
    mock_block.return_value = mock_invoke

    state = {
        "reaction_data": [],
        "training_data": [],
        "iteration": 2,
        "model_name": "pot",
        "ensemble_state": {"configs": ["input.xyz"]},
    }

    result = calculate_dft_level(state)
    assert result["ensemble_state"]["configs"] == ["dft_out.xyz"]
    mock_invoke.invoke.assert_called_once()


@patch("mlipflow.graphflow.activelearner.execute_mlip_training_block")
def test_train_new_mlip_model(mock_block):
    """Test train_new_mlip_model node execution.

    Verifies accumulation of training data and update of the fit prefix name.
    """
    mock_invoke = MagicMock()
    mock_invoke.invoke.return_value = {"configs": ["trained_model.xyz"], "iteration": 3}
    mock_block.return_value = mock_invoke

    mock_mlip_strategy = MagicMock()

    state = {
        "reaction_data": [],
        "training_data": ["train1.xyz"],
        "iteration": 3,
        "model_name": "pot",
        "ensemble_state": {
            "configs": ["new_train.xyz"],
            "mlip_strategy": mock_mlip_strategy,
        },
    }

    result = train_new_mlip_model(state)
    assert result["training_data"] == ["train1.xyz", "new_train.xyz"]
    assert mock_mlip_strategy.mlip_name == "pot_v3"
    mock_invoke.invoke.assert_called_once()


def test_check_mlip_training_completion():
    """Test check_mlip_training_completion is a passthrough."""
    state = {"iteration": 1, "model_name": "test"}
    assert check_mlip_training_completion(state) == state


def test_route_mlip_training():
    """Test route_mlip_training route checks.

    Checks routing to DONE or looping based on model file existence.
    """
    # Should route to DONE if model_file is present
    state_done = {"model_file": "model.model"}
    assert route_mlip_training(state_done) == "DONE"

    # Should route to loop if model_file is missing
    state_loop = {}
    assert route_mlip_training(state_loop) == "loop"


def test_finalising_learning_loop():
    """Test finalising_learning_loop is a passthrough."""
    state = {"final": True}
    assert finalising_learning_loop(state) == state


@patch("mlipflow.graphflow.activelearner.execute_mlip_structure_generation_block")
@patch("mlipflow.graphflow.activelearner.execute_dft_single_point_block")
@patch("mlipflow.graphflow.activelearner.execute_mlip_training_block")
def test_run_active_learning_loop(mock_train, mock_dft, mock_gen, tmp_path):
    """Test compilation and execution of active learning loop with SqliteSaver.

    Verifies compilation of StateGraph and SQL database persistence checks.
    """
    # Setup mocks for subgraph nodes
    mock_gen_invoke = MagicMock()
    mock_gen_invoke.invoke.return_value = {"configs": ["gen_structs.xyz"]}
    mock_gen.return_value = mock_gen_invoke

    mock_dft_invoke = MagicMock()
    mock_dft_invoke.invoke.return_value = {"configs": ["dft_out.xyz"]}
    mock_dft.return_value = mock_dft_invoke

    mock_train_invoke = MagicMock()
    mock_train_invoke.invoke.return_value = {
        "configs": ["trained_model.xyz"],
        "model_file": "final.model",  # Ensures training routes to DONE
    }
    mock_train.return_value = mock_train_invoke

    db_path = tmp_path / "test_persist.db"

    mock_mlip_strategy = MagicMock()
    initial_state = {
        "reaction_data": [],
        "training_data": ["init.xyz"],
        "iteration": 1,
        "model_name": "persist_pot",
        "ensemble_state": {
            "configs": ["new_configs.xyz"],
            "mlip_strategy": mock_mlip_strategy,
        },
    }

    final_state = run_active_learning_loop(
        initial_state=initial_state, storage_db_path=str(db_path)
    )

    assert final_state["training_data"] == ["init.xyz", "new_configs.xyz"]
    assert db_path.exists()
    assert db_path.stat().st_size > 0
