import logging
from langgraph.graph import StateGraph, START, END
from mlipflow.graphflow.nodes import *
from mlipflow.data import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def execute_mlip_structure_generation_block():
    """
    Workflow to generate new structures via MLIP MD/OPT/DyNEB.
    """
    logger.info("...Executing MLIP structure generation block...")
    graph = StateGraph(EnsembleState)

    graph.add_node('gen_structs', run_mlip_structure_generation)
    graph.add_node('mlip_sp', run_mlip_sp)
    graph.add_node('select_uncertain_configs', run_config_uncertainty_selection)

    graph.add_edge(START, 'gen_structs')
    graph.add_edge('gen_structs', 'mlip_sp')
    graph.add_edge('mlip_sp', 'select_uncertain_configs')
    graph.add_edge('select_uncertain_configs', END)

    return graph.compile()


def execute_dft_single_point_block():
    """
    Workflow to prepare, run, postprocess DFT single-point calculations.
    """
    logger.info("...Executing DFT single-point block...")
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
    logger.info("...Executing MLIP training block...")
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
    logger.info("...Executing initial basin pathsampling MD block...")
    
    graph = StateGraph(EnsembleState)
    graph.add_node('apply_and_run_basin', run_apply_basin_constraints)
    graph.add_node('check_config_labels', run_topology_relabel)
    graph.add_node('gen_and_run_neb', run_generate_neb_pairs)
    graph.add_node('merge_configs', merge_configs)
    graph.add_node('mlip_sp', run_mlip_sp)
    graph.add_node('select_configs', run_config_fps_selection)

    graph.add_edge(START, 'apply_and_run_basin')
    graph.add_edge('apply_and_run_basin', 'check_config_labels')
    graph.add_edge('check_config_labels', 'gen_and_run_neb')
    graph.add_edge('gen_and_run_neb', 'merge_configs')
    graph.add_edge('merge_configs', 'mlip_sp')
    graph.add_edge('mlip_sp', 'select_configs')
    graph.add_edge('select_configs', END)

    return graph.compile()


def execute_opt_neb_combination_block():
    """
    Workflow for Opt-NEB Combination.
    Generates structures via OPT followed by NEB and selects by uncertainty.
    """
    logger.info("...Executing opt-neb combination block...")
    graph = StateGraph(EnsembleState)

    graph.add_node('run_optimisation', run_mlip_structure_generation)
    graph.add_node('switch_to_neb', switch_to_neb_generation)
    graph.add_node('gen_and_run_neb', run_generate_neb_pairs)
    graph.add_node('mlip_sp', run_mlip_sp)
    graph.add_node('select_configs', run_config_uncertainty_selection)

    graph.add_edge(START, 'run_optimisation')
    graph.add_edge('run_optimisation', 'switch_to_neb')
    graph.add_edge('switch_to_neb', 'gen_and_run_neb')
    graph.add_edge('gen_and_run_neb', 'mlip_sp')
    graph.add_edge('mlip_sp', 'select_configs')
    graph.add_edge('select_configs', END)

    return graph.compile()
