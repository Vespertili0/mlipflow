from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from mlipflow.graphflow.tasknodes import *

def create_dft_fit_graphflow(initial_state: dict):
    graph = StateGraph(EnsembleState)
    graph.add_node('dft_sp', run_dft_sp)
    graph.add_node('train_mace', run_mace_fit)
    graph.add_edge(START, 'dft_sp')
    graph.add_edge('dft_sp', 'train_mace')
    graph.add_entry_point('dft_sp')
    graph.add_finish_point('train_mace')
    workflow = graph.compile()
    return workflow.invoke(initial_state)