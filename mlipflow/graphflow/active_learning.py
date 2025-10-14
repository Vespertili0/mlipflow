import copy, glob, os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .taskflows import *

class ActiveLearningFlow(TypedDict):
    reaction_data: list[str]
    training_data: list[str]
    iteration: int
    model_name: Optional[str]
    ensemble_state: EnsembleState


def generate_new_structures(state:ActiveLearningFlow) -> ActiveLearningFlow:
    subgraph_out = execute_mlip_structure_generation_block().invoke(state['ensemble_state'])
    return {**state, 'ensemble_state': subgraph_out}


def calculate_dft_level(state:ActiveLearningFlow) -> ActiveLearningFlow:
    subgraph_out = execute_dft_single_point_block().invoke(state['ensemble_state'])
    return {**state, 'ensemble_state': subgraph_out}


def train_new_mlip_model(state:ActiveLearningFlow) -> ActiveLearningFlow:
    ensemble_state = copy.deepcopy(state['ensemble_state'])
    
    # collect all train-data in state['configs]
    new_training_data = state['training_data'] + state['ensemble_state']['configs']

    # update mlip_name in mlip_strategy
    ensemble_state['configs'] = new_training_data
    ensemble_state['mlip_strategy'].mlip_name = f"{state['model_name']}_v{state['iteration']}"

    subgraph_out = execute_mlip_training_block().invoke(ensemble_state)
    return {**state, 'training_data': new_training_data, 'ensemble_state': subgraph_out}


def check_mlip_training_completion(state:ActiveLearningFlow) -> ActiveLearningFlow:
    return state

# Gate function checking MLIP training completion
def route_mlip_training(state:ActiveLearningFlow):
    if state['model_file']:
        return 'DONE'
    else:
        return 'loop'
    

def finalising_learning_loop(state:ActiveLearningFlow) -> ActiveLearningFlow:
    return state




def run_active_learning_loop(initial_state: dict):
    graph = StateGraph(ActiveLearningFlow)
    graph.add_node('gen_new', generate_new_structures)
    graph.add_node('run_dft', calculate_dft_level)
    graph.add_node('train_model', train_new_mlip_model)
    graph.add_node('check_training', check_mlip_training_completion)
    graph.add_node('finalise', finalising_learning_loop)

    graph.add_edge(START, 'gen_new')
    graph.add_edge('gen_new', 'run_dft')
    graph.add_edge('run_dft', 'train_model')
    graph.add_conditional_edges(
        'check_training',
        route_mlip_training,
        {
            'DONE': 'finalise',
            'loop': 'train_model'
        }
    )
    graph.add_edge('finalise', END)

    return graph.compile().invoke(initial_state)