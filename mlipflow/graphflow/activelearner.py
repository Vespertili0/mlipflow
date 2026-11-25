from __future__ import annotations

import copy
import logging
from typing import TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired

from mlipflow.graphflow.graphs import (
    execute_dft_single_point_block,
    execute_initial_basin_pathsampling_md_block,
    execute_mlip_training_block,
    execute_opt_neb_combination_block,
)
from mlipflow.graphflow.nodes import EnsembleState  # noqa: TC001

logger = logging.getLogger(__name__)


class ActiveLearningFlow(TypedDict):
    """Top-level state for the active learning loop.

    Attributes
    ----------
    reaction_data : list[str]
        Paths to reaction data files.
    training_data : list[str]
        Accumulated training data file paths.
    test_data : list[str], optional
        Accumulated held-out ``dft_test`` file paths across iterations.
    iteration : int
        Current active-learning iteration index.
    model_name : str or None
        Base name of the MLIP model being trained.
    ensemble_state : EnsembleState
        Nested state dictionary passed to subgraph invocations.
    """

    reaction_data: list[str]
    training_data: list[str]
    test_data: NotRequired[list[str]]
    iteration: int
    model_name: str | None
    ensemble_state: EnsembleState


def _as_path_list(value: str | list[str] | None) -> list[str]:
    """Normalize an optional file path or list of file paths to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _propagate_iteration(state: ActiveLearningFlow) -> EnsembleState:
    """Copy the outer iteration index into the ensemble_state payload."""
    es = copy.deepcopy(state["ensemble_state"])
    current_iter = state.get("iteration", 0)
    if es.get("iteration") != current_iter:
        es["step_counter"] = 1
        es.pop("model_file", None)
        es.pop("unknown_configs", None)
    es["iteration"] = current_iter
    return es


def generate_new_structures(state: ActiveLearningFlow) -> ActiveLearningFlow:
    """Run structure generation subgraph."""
    es = _propagate_iteration(state)
    if state.get("iteration", 0) == 0:
        subgraph_out = execute_initial_basin_pathsampling_md_block().invoke(es)
    else:
        subgraph_out = execute_opt_neb_combination_block().invoke(es)
    return {**state, "ensemble_state": subgraph_out}


def calculate_dft_level(state: ActiveLearningFlow) -> ActiveLearningFlow:
    """Run DFT single-point subgraph."""
    es = _propagate_iteration(state)
    subgraph_out = execute_dft_single_point_block().invoke(es)
    return {**state, "ensemble_state": subgraph_out}


def train_new_mlip_model(state: ActiveLearningFlow) -> ActiveLearningFlow:
    """Run MLIP training subgraph and synchronize model/dataset state."""
    ensemble_state = _propagate_iteration(state)

    # Accumulate training configs without duplicates
    new_training_data = list(state.get("training_data", []))
    for path in _as_path_list(state["ensemble_state"].get("configs")):
        if path not in new_training_data:
            new_training_data.append(path)

    # Accumulate held-out dft_test paths without leaking into training_data
    new_test_data = list(state.get("test_data", []))
    for path in _as_path_list(state["ensemble_state"].get("outfile")):
        if path not in new_test_data and path not in new_training_data:
            new_test_data.append(path)

    # Synchronize model name, model_file, and uq_file on mlip_strategy
    versioned_name = f"{state['model_name']}_v{state['iteration']}"
    strategy = ensemble_state["mlip_strategy"]
    if hasattr(strategy, "set_model_name"):
        strategy.set_model_name(versioned_name)
    else:
        strategy.mlip_name = versioned_name

    # Pass full accumulated training set and record it for subsequent UQ iterations
    ensemble_state["configs"] = list(new_training_data)
    ensemble_state["last_training_configs"] = list(new_training_data)

    subgraph_out = execute_mlip_training_block().invoke(ensemble_state)
    subgraph_out = {**subgraph_out, "last_training_configs": list(new_training_data)}

    return {
        **state,
        "training_data": new_training_data,
        "test_data": new_test_data,
        "ensemble_state": subgraph_out,
    }


def check_mlip_training_completion(state: ActiveLearningFlow) -> ActiveLearningFlow:
    """Passthrough node for training completion inspection."""
    return state


# Gate function checking MLIP training completion
def route_mlip_training(state: ActiveLearningFlow):
    """Route based on whether a trained model file exists."""
    if state.get("model_file") or state.get("ensemble_state", {}).get("model_file"):
        return "DONE"
    return "loop"


def finalising_learning_loop(state: ActiveLearningFlow) -> ActiveLearningFlow:
    """Terminal node for loop finalisation."""
    return state


def run_active_learning_loop(
    initial_state: dict, storage_db_path: str = "governance_state.db"
):
    """Compile and invoke the active learning StateGraph with persistence.

    Parameters
    ----------
    initial_state : dict
        Initial state dictionary conforming to ``ActiveLearningFlow``.
    storage_db_path : str, optional
        File path for the SQLite checkpointer database.
        Defaults to ``"governance_state.db"``.

    Returns
    -------
    dict
        Final state after completing or resuming the learning loop.
    """
    graph = StateGraph(ActiveLearningFlow)
    graph.add_node("gen_new", generate_new_structures)
    graph.add_node("run_dft", calculate_dft_level)
    graph.add_node("train_model", train_new_mlip_model)
    graph.add_node("check_training", check_mlip_training_completion)
    graph.add_node("finalise", finalising_learning_loop)

    graph.add_edge(START, "gen_new")
    graph.add_edge("gen_new", "run_dft")
    graph.add_edge("run_dft", "train_model")
    graph.add_edge("train_model", "check_training")
    graph.add_conditional_edges(
        "check_training",
        route_mlip_training,
        {"DONE": "finalise", "loop": "train_model"},
    )
    graph.add_edge("finalise", END)

    # Establish durable local persistence with timeout for concurrent access
    serde = JsonPlusSerializer(pickle_fallback=True)
    if storage_db_path == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        memory_layer = MemorySaver(serde=serde)
        runtime_graph = graph.compile(checkpointer=memory_layer)
        thread_id = f"workflow_{initial_state.get('model_name', 'unnamed_pot')}"
        execution_config = {"configurable": {"thread_id": thread_id}}
        logger.info("Invoking active learning loop with thread_id=%s", thread_id)
        return runtime_graph.invoke(initial_state, config=execution_config)
    else:
        with SqliteSaver.from_conn_string(storage_db_path) as memory_layer:
            if memory_layer is not None:
                memory_layer.serde = serde
            runtime_graph = graph.compile(checkpointer=memory_layer)
            thread_id = f"workflow_{initial_state.get('model_name', 'unnamed_pot')}"
            execution_config = {"configurable": {"thread_id": thread_id}}
            logger.info("Invoking active learning loop with thread_id=%s", thread_id)
            return runtime_graph.invoke(initial_state, config=execution_config)
