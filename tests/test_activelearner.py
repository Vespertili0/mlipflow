from __future__ import annotations

from unittest.mock import MagicMock, patch

from mlipflow.graphflow.activelearner import (
    _propagate_iteration,
    calculate_dft_level,
    generate_new_structures,
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


@patch("mlipflow.graphflow.activelearner.execute_opt_neb_combination_block")
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


class DummyStrategy:
    def __init__(self):
        self.mlip_name = "pot"

    def __deepcopy__(self, memo):
        return self


@patch("mlipflow.graphflow.activelearner.execute_mlip_training_block")
def test_train_new_mlip_model(mock_block):
    mock_invoke = MagicMock()
    mock_invoke.invoke.return_value = {"configs": ["trained_model.xyz"], "iteration": 3}
    mock_block.return_value = mock_invoke

    mock_mlip_strategy = DummyStrategy()

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


@patch("mlipflow.graphflow.activelearner.SqliteSaver.from_conn_string")
@patch("mlipflow.graphflow.activelearner.execute_opt_neb_combination_block")
@patch("mlipflow.graphflow.activelearner.execute_dft_single_point_block")
@patch("mlipflow.graphflow.activelearner.execute_mlip_training_block")
def test_run_active_learning_loop(
    mock_train, mock_dft, mock_gen, mock_sqlite, tmp_path
):
    mock_mlip_strategy = DummyStrategy()

    mock_gen_invoke = MagicMock()
    mock_gen_invoke.invoke.return_value = {
        "configs": ["gen_structs.xyz"],
        "mlip_strategy": mock_mlip_strategy,
    }
    mock_gen.return_value = mock_gen_invoke

    mock_dft_invoke = MagicMock()
    mock_dft_invoke.invoke.return_value = {
        "configs": ["dft_out.xyz"],
        "mlip_strategy": mock_mlip_strategy,
    }
    mock_dft.return_value = mock_dft_invoke

    mock_train_invoke = MagicMock()
    mock_train_invoke.invoke.return_value = {
        "configs": ["trained_model.xyz"],
        "model_file": "final.model",
        "mlip_strategy": mock_mlip_strategy,
    }
    mock_train.return_value = mock_train_invoke

    mock_sqlite.return_value.__enter__.return_value = None

    db_path = tmp_path / "test_persist.db"

    initial_state = {
        "reaction_data": [],
        "training_data": ["init.xyz"],
        "iteration": 1,
        "model_name": "persist_pot",
        "ensemble_state": {
            "configs": ["new_configs.xyz"],
            "mlip_strategy": DummyStrategy(),
        },
    }

    final_state = run_active_learning_loop(
        initial_state=initial_state, storage_db_path=str(db_path)
    )

    assert final_state["training_data"] == ["init.xyz", "dft_out.xyz"]
