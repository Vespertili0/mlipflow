from typing import TypedDict
from langgraph.graph import StateGraph, START
from .taskflows import *

class ActiveLearningFlow(TypedDict):
    reaction_data: str
    training_data: str
    iteration: int
    ensemble_state:dict


def generate_new_structures(state:ActiveLearningFlow):
    subgraph_out = execute_mlip_structure_generation_block().invoke(state['ensemble_state'])
    return {**state, 'ensemble_state': subgraph_out}

def calculate_dft_level(state:ActiveLearningFlow):
    subgraph_out = execute_dft_single_point_block().invoke(state['ensemble_state'])
    return {**state, 'ensemble_state': subgraph_out}

def train_new_mlip_model(state:ActiveLearningFlow):
    subgraph_out = execute_mlip_training_block().invoke(state['ensemble_state'])
    return {**state, 'ensemble_state': subgraph_out}



def run_active_learning_loop(initial_state: dict):
    graph = StateGraph(ActiveLearningFlow)
    graph.add_node('gen_new', generate_new_structures)
    graph.add_node('run_dft', calculate_dft_level)
    graph.add_node('train_model', train_new_mlip_model)

    graph.add_edge(START, 'gen_new')
    graph.add_edge('gen_new', 'run_dft')
    graph.add_edge('run_dft', 'train_model')

    graph.set_entry_point('gen_new')
    graph.set_finish_point('train_model')

    return graph.compile().invoke(initial_state)