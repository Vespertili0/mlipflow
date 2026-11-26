from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ase.io import read

from mlipflow.graphflow.activelearner import (
    _propagate_iteration,
    calculate_dft_level,
    generate_new_structures,
    run_active_learning_loop,
    train_new_mlip_model,
)
from mlipflow.strategies.dft import EMTCalc
from mlipflow.strategies.mlip import MACEModel


def test_propagate_iteration():
    """Test _propagate_iteration helper.

    Verifies iteration index propagation and deep copy structure.
    """
    state = {
        "reaction_data": ["r1.xyz"],
        "training_data": ["t1.xyz"],
        "test_data": [],
        "iteration": 3,
        "model_name": "pot",
        "ensemble_state": {
            "configs": ["c1.xyz"],
            "iteration": 2,
            "unknown_configs": ["stale_unknown.xyz"],
        },
    }

    es = _propagate_iteration(state)
    assert es["iteration"] == 3
    assert es["configs"] == ["c1.xyz"]
    assert "unknown_configs" not in es
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
        "test_data": [],
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
        "test_data": [],
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
        self.model_file = "pot.model"
        self.uq_file = "pot_llpr.pt"

    def set_model_name(self, mlip_name: str, uq_file: str | None = None) -> None:
        self.mlip_name = mlip_name
        self.model_file = f"{mlip_name}.model"
        self.uq_file = uq_file if uq_file is not None else f"{mlip_name}_llpr.pt"

    def __deepcopy__(self, memo):
        return self


@patch("mlipflow.graphflow.activelearner.execute_mlip_training_block")
def test_train_new_mlip_model(mock_block):
    mock_invoke = MagicMock()
    mock_invoke.invoke.side_effect = lambda es: {**es, "outfile": None}
    mock_block.return_value = mock_invoke

    mace_strategy = MACEModel(mlip_name="pot", run_mode="local")

    state = {
        "reaction_data": [],
        "training_data": ["train1.xyz"],
        "test_data": ["test0.xyz"],
        "iteration": 3,
        "model_name": "pot",
        "ensemble_state": {
            "configs": ["new_train.xyz"],
            "outfile": ["new_test.xyz"],
            "mlip_strategy": mace_strategy,
        },
    }

    result = train_new_mlip_model(state)
    expected_train = ["train1.xyz", "new_train.xyz"]
    expected_test = ["test0.xyz", "new_test.xyz"]

    assert result["training_data"] == expected_train
    assert result["test_data"] == expected_test
    assert result["ensemble_state"]["last_training_configs"] == expected_train
    assert result["ensemble_state"]["last_test_configs"] == expected_test

    updated_strategy = result["ensemble_state"]["mlip_strategy"]
    assert updated_strategy.mlip_name == "pot_v3"
    assert updated_strategy.model_file == "pot_v3.model"
    assert updated_strategy.uq_file == "pot_v3_llpr.pt"
    mock_invoke.invoke.assert_called_once()
    invoked_state = mock_invoke.invoke.call_args.args[0]
    assert invoked_state["outfile"] == expected_test
    assert invoked_state["last_test_configs"] == expected_test


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
        "outfile": ["dft_test_out.xyz"],
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
        "test_data": [],
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
    assert final_state["test_data"] == ["dft_test_out.xyz"]
    assert final_state["ensemble_state"]["last_training_configs"] == [
        "init.xyz",
        "dft_out.xyz",
    ]


@patch("mlipflow.graphflow.activelearner.execute_opt_neb_combination_block")
@patch("mlipflow.graphflow.activelearner.execute_dft_single_point_block")
@patch("mlipflow.graphflow.activelearner.execute_mlip_training_block")
def test_run_active_learning_loop_short_circuits_on_empty_selection(
    mock_train, mock_dft, mock_gen, tmp_path
):
    """Verify 0 selected structures short-circuits straight to finalise without DFT or retraining."""
    empty_sel = tmp_path / "09_selected_configs.xyz"
    empty_sel.write_text("", encoding="utf-8")

    mock_gen_invoke = MagicMock()
    mock_gen_invoke.invoke.return_value = {
        "configs": [str(empty_sel)],
        "selected_count": 0,
        "mlip_strategy": DummyStrategy(),
    }
    mock_gen.return_value = mock_gen_invoke

    initial_state = {
        "reaction_data": [],
        "training_data": ["init_train.xyz"],
        "test_data": ["init_test.xyz"],
        "iteration": 1,
        "model_name": "pot",
        "ensemble_state": {"configs": ["seed.xyz"], "mlip_strategy": DummyStrategy()},
    }

    final_state = run_active_learning_loop(
        initial_state=initial_state, storage_db_path="memory"
    )

    mock_gen_invoke.invoke.assert_called_once()
    mock_dft.assert_not_called()
    mock_train.assert_not_called()
    assert final_state["training_data"] == ["init_train.xyz"]
    assert final_state["test_data"] == ["init_test.xyz"]


def test_multi_iteration_active_learning_dry_run_with_emt(tmp_path):
    """Multi-iteration (v0 -> v1 -> v2) dry run with EMTCalc, real test_data.xyz, and real LLPR calibration.

    Verifies:
    - mlip_strategy.model_file and uq_file advance cleanly (v0 -> v1 -> v2).
    - Post-fit LLPR calibration produces reloadable pot_v{it}_llpr.pt and element_metrics.
    - ensemble_state['last_training_configs'], ['last_test_configs'], ['uq_file'], and
      ['element_metrics'] propagate cleanly into subsequent iterations' structure generation.
    - state['test_data'] accumulates held-out dft_test files across iterations
      without leaking into state['training_data'].
    """
    test_dir = Path(__file__).resolve().parent
    src_xyz = test_dir / "data" / "test_data.xyz"
    seed_xyz = tmp_path / "seed_configs.xyz"
    shutil.copy2(src_xyz, seed_xyz)

    observed_gen_states: list[
        tuple[int, str, list[str] | None, str | None, dict | None]
    ] = []
    fitted_models: list[tuple[str, str, str | None]] = []

    def _fake_gen_invoke(es):
        it = es["iteration"]
        observed_gen_states.append(
            (
                it,
                es["mlip_strategy"].model_file,
                list(es["last_training_configs"])
                if "last_training_configs" in es
                else None,
                es.get("uq_file"),
                es.get("element_metrics"),
            )
        )
        iter_in = tmp_path / f"gen_iter_{it}.xyz"
        shutil.copy2(seed_xyz, iter_in)
        return {
            **es,
            "configs": [str(iter_in)],
            "step_counter": es.get("step_counter", 1) + 1,
        }

    fitted_test_sets: list[list[str]] = []

    def _fake_fit_new_model(self, in_file, test_configs, **_kwargs):
        _ = in_file
        fitted_models.append((self.mlip_name, self.model_file, self.uq_file))
        fitted_test_sets.append(list(test_configs))
        Path(self.model_file).write_text("trained model weights", encoding="utf-8")

    class _FastMACECalc:
        """Fast deterministic calculator for 3-iteration dry-run calibration."""

        def __init__(self, **_kwargs):
            pass

        def get_descriptors(self, atoms, invariants_only: bool = True):
            _ = invariants_only
            pos = atoms.get_positions()
            nums = atoms.get_atomic_numbers()[:, None]
            return np.hstack([pos, nums, pos**2]).astype(np.float64)

        def get_forces(self, atoms=None):
            return np.zeros((len(atoms), 3), dtype=np.float64)

    mock_gen_block = MagicMock()
    mock_gen_block.invoke.side_effect = _fake_gen_invoke

    mace_strategy = MACEModel(mlip_name="pot", run_mode="local")
    qchem_strategy = EMTCalc()

    state = {
        "reaction_data": [],
        "training_data": [],
        "test_data": [],
        "iteration": 0,
        "model_name": "pot",
        "ensemble_state": {
            "configs": [str(seed_xyz)],
            "qchem_strategy": qchem_strategy,
            "mlip_strategy": mace_strategy,
            "calculation_kwargs": {
                "dft_scf": {"chunk_size": 20},
                "dft_split": {
                    "split_mode": "stratified_random",
                    "test_fraction": 0.2,
                    "seed": 42,
                    "max_force": 1e6,
                },
                "llpr_calibration": {
                    "dispersion": False,
                    "atom_indices": slice(48, None),
                    "exclude_fixed": False,
                    "tune_lambdas": (1e-4,),
                },
            },
        },
    }

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with (
            patch(
                "mlipflow.graphflow.activelearner.execute_initial_basin_pathsampling_md_block",
                return_value=mock_gen_block,
            ),
            patch(
                "mlipflow.graphflow.activelearner.execute_opt_neb_combination_block",
                return_value=mock_gen_block,
            ),
            patch("mlipflow.graphflow.nodes.MACECalculator", side_effect=_FastMACECalc),
            patch.object(MACEModel, "fit_new_model", new=_fake_fit_new_model),
        ):
            for it in range(3):
                state = {**state, "iteration": it}
                state = run_active_learning_loop(
                    initial_state=state, storage_db_path="memory"
                )

                strat = state["ensemble_state"]["mlip_strategy"]
                assert strat.mlip_name == f"pot_v{it}"
                assert strat.model_file == f"pot_v{it}.model"
                assert strat.uq_file == f"pot_v{it}_llpr.pt"
                assert state["ensemble_state"]["uq_file"] == f"pot_v{it}_llpr.pt"
                assert Path(f"pot_v{it}_llpr.pt").exists()
                assert set(state["ensemble_state"]["element_metrics"].keys()) == {
                    1,
                    6,
                    8,
                    29,
                }

                assert len(state["training_data"]) == it + 1
                assert len(state["test_data"]) == it + 1
                assert (
                    state["ensemble_state"]["last_training_configs"]
                    == state["training_data"]
                )
                assert (
                    state["ensemble_state"]["last_test_configs"] == state["test_data"]
                )
                assert set(state["test_data"]).isdisjoint(set(state["training_data"]))

        # Verify model progression and accumulated test_configs across fit calls
        assert fitted_models == [
            ("pot_v0", "pot_v0.model", "pot_v0_llpr.pt"),
            ("pot_v1", "pot_v1.model", "pot_v1_llpr.pt"),
            ("pot_v2", "pot_v2.model", "pot_v2_llpr.pt"),
        ]
        assert fitted_test_sets == [
            state["test_data"][:1],
            state["test_data"][:2],
            state["test_data"][:3],
        ]

        # Verify iterations 1 and 2 saw updated model_file, last_training_configs, uq_file, and element_metrics
        assert observed_gen_states[1][1] == "pot_v0.model"
        assert observed_gen_states[1][2] == state["training_data"][:1]
        assert observed_gen_states[1][3] == "pot_v0_llpr.pt"
        assert observed_gen_states[1][4] is not None
        assert observed_gen_states[2][1] == "pot_v1.model"
        assert observed_gen_states[2][2] == state["training_data"][:2]
        assert observed_gen_states[2][3] == "pot_v1_llpr.pt"
        assert observed_gen_states[2][4] is not None

        # All accumulated dft_train and dft_test files exist on disk with expected counts
        for train_path, test_path in zip(state["training_data"], state["test_data"]):
            assert Path(train_path).exists()
            assert Path(test_path).exists()
            assert "dft_train" in train_path
            assert "dft_test" in test_path
            assert len(read(train_path, ":")) == 14
            assert len(read(test_path, ":")) == 3
    finally:
        os.chdir(cwd)
