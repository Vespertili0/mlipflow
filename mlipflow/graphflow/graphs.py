from langgraph.graph import StateGraph, START, END
from mlipflow.graphflow.nodes import *
# from mlipflow.data import *

def execute_mlip_structure_generation_block():
    """
    Workflow to generate new structures via MLIP MD/OPT/DyNEB.
    """
    graph = StateGraph(EnsembleState)

    graph.add_node('gen_structs', run_mlip_structure_generation)
    graph.add_node('mlip_sp', run_mlip_sp)

    graph.add_edge(START, 'gen_structs')
    graph.add_edge('gen_structs', 'mlip_sp')
    graph.add_edge('mlip_sp', END)

    return graph.compile()


def execute_dft_single_point_block():
    """
    Workflow to prepare, run, postprocess DFT single-point calculations.
    """
    graph = StateGraph(EnsembleState)

    graph.add_node('dft_sp', run_dft_sp)
    graph.add_node('assess&select', assess_n_select)

    graph.add_edge(START, 'dft_sp')
    graph.add_edge('dft_sp', 'assess&select')
    graph.add_edge('assess&select', END)

    return graph.compile()


def execute_mlip_training_block():
    """
    Workflow to prepare, run, postprocess MLIP training.
    """
    graph = StateGraph(EnsembleState)

    #graph.add_node('prepare_data', prepare_train_test_sets)
    graph.add_node('train_mace', run_mace_fit)
    #graph.add_node('', )

    graph.add_edge(START, 'train_mace')
    #graph.add_edge('train_mace', 'eval_mlip')
    graph.add_edge('train_mace', END)
     
    return graph.compile()


def execute_initial_basin_pathsampling_md_block():
    """
    Workflow for initial MD of Basins and NEB-pathways:
    Apply constraints -> Generate Structures (Basin-MD & path-sampling MD) -> Single Point.
    Constraints are applied in-memory and passed to generation immediately.
    """
    graph = StateGraph(EnsembleState)
   graph.add_node('apply_and_run_basin', run_apply_basin_constraints)    
    graph.add_node('gen_and_run_neb', run_generate_neb_pairs)
    graph.add_node('mlip_sp', run_mlip_sp)

    graph.add_edge(START, 'apply_and_run_basin')
    graph.add_edge('apply_and_run_basin', 'gen_and_run_neb')
    graph.add_edge('gen_and_run_neb', 'mlip_sp')
    graph.add_edge('mlip_sp', END)

    return graph.compile()
