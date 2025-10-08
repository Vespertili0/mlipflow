from langgraph.graph import StateGraph, START
from mlipflow.graphflow.tasknodes import *
from mlipflow.data import *

# test function
def create_dft_fit_graphflow(initial_state: dict):
    """
    Test function.
    """
    graph = StateGraph(EnsembleState)

    graph.add_node('dft_sp', run_dft_sp)
    graph.add_node('train_mace', run_mace_fit)

    graph.add_edge(START, 'dft_sp')
    graph.add_edge('dft_sp', 'train_mace')

    graph.set_entry_point('dft_sp')
    graph.set_finish_point('train_mace')
    workflow = graph.compile()

    validate_workflow(workflow, initial_state)

    return workflow.invoke(initial_state)


def execute_mlip_structure_generation_block(initial_state:dict):
    """
    Workflow to generate new structures via MLIP MD/OPT/DyNEB.
    """
    pass


def execute_dft_single_point_block(initial_state:dict):
    """
    Workflow to prepare, run, postprocess DFT single-point calculations.
    """
    graph = StateGraph(EnsembleState)

    graph.add_node('gen_structs', run_structure_generation)
    graph.add_node('dft_sp', run_dft_sp)
    graph.add_node('assess&select', assess_n_select)
    graph.add_node('train_mace', run_mace_fit)

    graph.add_edge(START, 'dft_sp')
    graph.add_edge('dft_sp', 'train_mace')

    graph.set_entry_point('dft_sp')
    graph.set_finish_point('train_mace')
    workflow = graph.compile()

    validate_workflow(workflow, initial_state)

    return workflow


def execute_mlip_training_block(initial_state:dict):
    """
    Workflow to prepare, run, postprocess MLIP training.
    """
    graph = StateGraph(EnsembleState)

    graph.add_node('prepare_data', prepare_train_test_sets)
    graph.add_node('train_mace', run_mace_fit)
    graph.add_node('', )

    graph.add_edge(START, 'train_mace')
    graph.add_edge('train_mace', 'eval_mlip')

    graph.set_entry_point('train_mace')
    graph.set_finish_point('eval_mlip')
    workflow = graph.compile()

    validate_workflow(workflow, initial_state)

    return workflow