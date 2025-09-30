from langgraph.graph import StateGraph, START
from mlipflow.graphflow.tasknodes import *


def create_dft_fit_graphflow(initial_state: dict):
    """State dictionary for the DFT-MACE fitting workflow.
    initial_state attributes
    ------------------------
    configs : list[str]
        List of configuration file paths (e.g., XYZ files).
    outfile : list[str]
        List of output file paths after DFT calculations.
    qchem_strategy : mlipflow.qchem_strategy.QChemStrategy object
        Strategy for quantum chemistry calculations.
    mlip_strategy : mlipflow.mlip_strategy.MLIPStrategy object
        Strategy for machine learning interatomic potential fitting.
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