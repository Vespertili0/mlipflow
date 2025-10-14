import logging
from typing import TypedDict, Optional
from dataclasses import dataclass
from typing_extensions import NotRequired

#from wfl.utils import logging
from mlipflow.data import *
from mlipflow.core.single_point import run_single_point, run_chunked_qe_sp
from mlipflow.core.calculate_error import calculate_mlip_error
from mlipflow.structure_generator import StructureGenStrategy
from mlipflow.mlip_strategy import MLIPStrategy
from mlipflow.qe_calculator import QChemStrategy

logger = logging.getLogger(__name__)
#####################################################

class EnsembleState(TypedDict):
    """
    State dictionary for the DFT-MACE fitting workflow    
    Attributes:
        configs (list[str]): List of configuration file paths
        outfile (NotRequired[list[str]]): Optional output file paths
        qchem_strategy (QChemStrategy): Quantum chemistry calculation strategy
        mlip_strategy (MLIPStrategy): MLIP fitting strategy
    """
    configs: list[str]
    outfile: NotRequired[list[str]]
    qchem_strategy: QChemStrategy
    mlip_strategy: MLIPStrategy
    structure_gen_strategy: NotRequired[StructureGenStrategy]


@dataclass
class WorkflowError(Exception):
    """Base exception for workflow errors"""
    message: str
    state: Optional[EnsembleState] = None



#def validate_state(state: EnsembleState, required_keys: list[str]) -> None:
#    """Validate state contains required keys with non-empty values"""
#    missing = [key for key in required_keys if not state.get(key)]
#    if missing:
#        raise WorkflowError(f"Missing required state keys: {missing}", state)
#
#
#def validate_workflow(workflow, initial_state: EnsembleState) -> None:
#    """Validate workflow configuration and initial state"""
#    # Validate nodes
#    required_nodes = {'dft_sp', 'train_mace'}
#    missing_nodes = required_nodes - set(workflow.nodes)
#    if missing_nodes:
#        raise ValueError(f"Missing required nodes: {missing_nodes}")
#        
#    # Validate initial state
#    validate_state(initial_state, ['configs', 'qchem_strategy', 'mlip_strategy'])


def run_dft_sp(state: EnsembleState) -> EnsembleState:
    """Run DFT single-point calculations on configurations.
    
    Args:
        state (EnsembleState): Current workflow state
        
    Returns:
        EnsembleState: Updated workflow state
        
    Raises:
        KeyError: If required state keys are missing
        ValueError: If configs list is empty
    """
    if not state.get('configs'):
        raise ValueError("No configurations provided in state")
        
    configs = state['configs']
    outfile = [xyz.replace('.xyz', '.dft.xyz') for xyz in configs]
    
    try:   
        run_single_point(
            in_file=configs,
            out_file=outfile,
            output_prefix=state['qchem_strategy'].qe_prefix,
            calculator=state['qchem_strategy'].get_calculator(
                job_name='QE_',
                ecut_eV=450,
                kpts=(3,3,1),
                calc_type='scf'
            ),
            remote_info=state['qchem_strategy'].remote_info
        )
        logger.debug(f"DFT calculations completed successfully")
    except Exception as e:
        logger.error(f"DFT calculation failed: {str(e)}")
        raise RuntimeError(f"DFT calculation failed: {str(e)}")
 
    return {**state, 'configs': outfile, 'outfile': None}


def run_dft_sp_block(state: EnsembleState) -> EnsembleState:
    """Run chunked DFT single-point calculations on configurations.
    
    Args:
        state (EnsembleState): Current workflow state
        
    Returns:
        EnsembleState: Updated workflow state
        
    Raises:
        KeyError: If required state keys are missing
        ValueError: If configs list is empty
    """
    if not state.get('configs'):
        raise ValueError("No configurations provided in state")
        
    configs = state['configs']
    outfile = [xyz.replace('.xyz', '.dft.xyz') for xyz in configs]
    
    try:   
        run_chunked_qe_sp(
            in_file=configs,
            out_file=outfile,
            chunk_size=70,
            qchem_strategy=state['qchem_strategy'],
            kpts=(3,3,1),
            dipole=False,
            dftd3=False,
            num_inputs_per_queued_job=1
            )
        logger.debug(f"DFT calculations completed successfully")
    except Exception as e:
        logger.error(f"DFT calculation failed: {str(e)}")
        raise RuntimeError(f"DFT calculation failed: {str(e)}")
 
    return {**state, 'configs': outfile, 'outfile': None}

#def clean_dft_data(state: EnsembleState) -> EnsembleState:
#    """Clean DFT data by removing unnecessary info and standardizing keys.
#    
#    Args:
#        state (EnsembleState): Current workflow state
#    Returns:
#        EnsembleState: Updated workflow state
#    Raises:
#        KeyError: If required state keys are missing
#        ValueError: If configs list is empty
#    """
#    check_maxforce_and_cleanarrays


def run_mlip_sp(state: EnsembleState) -> EnsembleState:
    """Run MLIP single-point calculations on configurations.
    
    Args:
        state (EnsembleState): Current workflow state
    Returns:
        EnsembleState: Updated workflow state
    Raises:
        KeyError: If required state keys are missing
        ValueError: If configs list is empty
    """
    if not state.get('configs'):
        raise ValueError("No configurations provided in state")
    
    if 'mlip_strategy' not in state:
        raise KeyError("mlip_strategy not found in state")

    mlip = state['mlip_strategy'] 
    configs = state['configs']
    outfile = [xyz.replace('.xyz', '.mace.xyz') for xyz in configs]
    
    try:   
        run_single_point(
            in_file=configs,
            out_file=outfile,
            output_prefix=mlip.mlip_prefix,
            calculator=mlip.get_calculator(
                job_name='mSP_',
                dispersion=False
                ),
            remote_info=mlip.remote_info
        )
        logger.debug(f"MLIP calculations completed successfully")
    except Exception as e:
        logger.error(f"MLIP calculation failed: {str(e)}")
        raise RuntimeError(f"MLIP calculation failed: {str(e)}")
    finally:
        clean_up()
 
    return {**state, 'configs': outfile, 'outfile': None}


#def prepare_train_test_sets(state, split_ratio=0.8):
#    pass


def assess_n_select(state: EnsembleState) -> EnsembleState:
    main_suffix='train'
    side_suffix='test'    
    
    configs = state['configs']
    split_configset_by_force_agreement(
        in_file=configs,
        out_file='dft.xyz',
        pair_tuple=(state["qchem_strategy"].qe_prefix, state["mlip_strategy"].mlip_prefix),
        main_suffix=main_suffix,
        side_suffix=side_suffix
    )
    train_data = [f'{main_suffix}_dft.xyz']
    test_data = [f'{side_suffix}_dft.xyz']

    return {**state, 'configs': train_data, 'outfile': test_data}


#def pool_mlip_training_data(state: EnsembleState) -> EnsembleState:
#
#
#    return {**state}


def run_mace_fit(state: EnsembleState) -> EnsembleState:
    configs = state['configs']
    test_configs = state['outfile']

    state['mlip_strategy'].fit_new_model(
        in_file=configs,
        test_configs=test_configs,
        seed=123, 
        restart=False
    )
    return {**state, 'outfile': None}


def run_mlip_structure_generation(state: EnsembleState) -> EnsembleState:
    """Generate new structures via MLIP MD/OPT/DyNEB."""

    if 'mlip_strategy' not in state:
        raise KeyError("mlip_strategy not found in state")
    
    if 'structure_gen_strategy' not in state:
        raise KeyError("structure_gen_strategy not found in state")
    
    if not state.get('configs'):
        raise ValueError("No configurations provided in state")
    
    mlip = state['mlip_strategy']
    structure_generator = state['structure_gen_strategy']
    configs = state['configs']
    outfile = [
        xyz.replace('.', f'_{structure_generator.calc_prefix}.') for xyz in configs
    ]
    try:
        structure_generator.generate_new_structures(
            in_file=configs,
            out_file=outfile,
            calculator=mlip.get_calculator(
                job_name='mSG_',
                dispersion=True
                ),
            remote_info=mlip.remote_info
        )
        tag_dict={
            'info':{'last_op__optimize_energy': f'{mlip.mlip_prefix}energy'},
            'array':{'last_op__optimize_forces': f'{mlip.mlip_prefix}forces'}
        }
        for tag_type, tags in tag_dict.items():
            for xyz in outfile:
                update_configset_tag(in_config=xyz, out_file=xyz, tag_dict=tags, tag_type=tag_type)
    except Exception as e:
        logger.error(f"Structure generation failed: {str(e)}")
        raise RuntimeError(f"Structure generation failed: {str(e)}")
    finally:
        clean_up()
    
    return {**state, 'configs': outfile, 'outfile': None}


#def evalute_mlip_error(state: EnsembleState) -> EnsembleState:
#    """
#    Evaluate MLIP error against DFT reference data.
#    
#    Args:
#        state (EnsembleState): Current workflow state
#    Returns:
#        EnsembleState: Updated workflow state
#    Raises:
#        KeyError: If required state keys are missing
#        ValueError: If configs list is empty
#    """
#    if not state.get('configs'):
#        raise ValueError("No configurations provided in state")
#    
#    try:
#        calculate_mlip_error(
#            in_configs=state['configs'],
#            out_file='error_analysis.xyz',
#            calc_property_prefix=state['mlip_strategy'].mlip_prefix,
#            fig_dir='.'
#            )
#    except Exception as e:
#        logger.error(f"MLIP error evaluation failed: {str(e)}")
#        raise RuntimeError(f"MLIP error evaluation failed: {str(e)}")