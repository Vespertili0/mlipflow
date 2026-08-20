from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from mlipflow.graphflow.graphs import (
    execute_dft_single_point_block,
    execute_initial_basin_pathsampling_md_block,
    execute_mlip_training_block,
    execute_neb_analysis_block,
    execute_opt_neb_combination_block,
)
from mlipflow.graphflow.nodes import AnalysisState, EnsembleState  # noqa: TC001

logger = logging.getLogger(__name__)


class ActiveLearningFlow(TypedDict):
    """Top-level state for the active learning loop.

    Attributes
    ----------
    reaction_data : list[str]
        Paths to reaction data files.
    training_data : list[str]
        Accumulated training data file paths.
    iteration : int
        Current active-learning iteration index.
    model_name : str or None
        Base name of the MLIP model being trained.
    ensemble_state : EnsembleState
        Nested state dictionary passed to subgraph invocations.
    """

    reaction_data: list[str]
    training_data: list[str]
    iteration: int
    model_name: str | None
    ensemble_state: EnsembleState


def _propagate_iteration(state: ActiveLearningFlow) -> EnsembleState:
    """Copy the outer iteration index into the ensemble_state payload."""
    es = copy.deepcopy(state["ensemble_state"])
    current_iter = state.get("iteration", 0)
    if es.get("iteration") != current_iter:
        es["step_counter"] = 1
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


def trigger_analysis_flow(state: ActiveLearningFlow) -> ActiveLearningFlow:
    current_iter = state.get("iteration", 0)
    if current_iter < 2:
        logger.info("Iteration %d < 2, skipping NEB analysis trigger.", current_iter)
        return state

    ensemble_state = state["ensemble_state"]
    neb_configs = [
        f for f in ensemble_state.get("configs", []) if "neb_opt" in Path(f).name
    ]
    if not neb_configs:
        logger.warning("No NEB opt configs found in ensemble_state; skipping analysis.")
        return state

    analysis_state: AnalysisState = {
        "configs": neb_configs,
        "iteration": current_iter,
        "model_name": state.get("model_name") or "pot",
        "model_dir": str(Path.cwd()),
        "qchem_strategy": ensemble_state["qchem_strategy"],
        "calculation_kwargs": ensemble_state.get("calculation_kwargs", {}),
        "step_counter": 1,
    }

    def _run_analysis_safe(analysis_st: AnalysisState) -> None:
        try:
            execute_neb_analysis_block().invoke(analysis_st)
            logger.info(
                "Detached NEB analysis completed for iteration %d.", current_iter
            )
        except Exception as exc:
            logger.error(
                "Detached NEB analysis failed for iteration %d: %s", current_iter, exc
            )

    from threading import Thread

    Thread(target=_run_analysis_safe, args=(analysis_state,), daemon=True).start()
    logger.info("Detached NEB analysis thread launched for iteration %d.", current_iter)
    return state


def calculate_dft_level(state: ActiveLearningFlow) -> ActiveLearningFlow:
    """Run DFT single-point subgraph."""
    es = _propagate_iteration(state)
    subgraph_out = execute_dft_single_point_block().invoke(es)
    return {**state, "ensemble_state": subgraph_out}


def train_new_mlip_model(state: ActiveLearningFlow) -> ActiveLearningFlow:
    """Run MLIP training subgraph."""
    ensemble_state = _propagate_iteration(state)

    # Collect all train-data in state['configs']
    new_training_data = state["training_data"] + state["ensemble_state"]["configs"]

    # Update mlip_name in mlip_strategy
    ensemble_state["configs"] = new_training_data
    ensemble_state[
        "mlip_strategy"
    ].mlip_name = f"{state['model_name']}_v{state['iteration']}"

    subgraph_out = execute_mlip_training_block().invoke(ensemble_state)
    return {**state, "training_data": new_training_data, "ensemble_state": subgraph_out}


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
    graph.add_node("trigger_analysis", trigger_analysis_flow)
    graph.add_node("run_dft", calculate_dft_level)
    graph.add_node("train_model", train_new_mlip_model)
    graph.add_node("check_training", check_mlip_training_completion)
    graph.add_node("finalise", finalising_learning_loop)

    graph.add_edge(START, "gen_new")
    graph.add_edge("gen_new", "trigger_analysis")
    graph.add_edge("trigger_analysis", "run_dft")
    graph.add_edge("run_dft", "train_model")
    graph.add_edge("train_model", "check_training")
    graph.add_conditional_edges(
        "check_training",
        route_mlip_training,
        {"DONE": "finalise", "loop": "train_model"},
    )
    graph.add_edge("finalise", END)

    # Establish durable local persistence with timeout for concurrent access
    if storage_db_path == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        memory_layer = MemorySaver()
        runtime_graph = graph.compile(checkpointer=memory_layer)
        thread_id = f"workflow_{initial_state.get('model_name', 'unnamed_pot')}"
        execution_config = {"configurable": {"thread_id": thread_id}}
        logger.info("Invoking active learning loop with thread_id=%s", thread_id)
        return runtime_graph.invoke(initial_state, config=execution_config)
    else:
        with SqliteSaver.from_conn_string(storage_db_path) as memory_layer:
            runtime_graph = graph.compile(checkpointer=memory_layer)
            thread_id = f"workflow_{initial_state.get('model_name', 'unnamed_pot')}"
            execution_config = {"configurable": {"thread_id": thread_id}}
            logger.info("Invoking active learning loop with thread_id=%s", thread_id)
            return runtime_graph.invoke(initial_state, config=execution_config)
