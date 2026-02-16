import logging
from typing_extensions import NotRequired
from typing import TypedDict, Optional, Any

#from wfl.utils import logging
from mlipflow.data.io import clean_up
from mlipflow.data.processing import update_configset_tag
from mlipflow.data.selection import split_configset_by_force_agreement
from mlipflow.core.single_point import run_single_point, run_chunked_qe_sp
from mlipflow.core.calculate_error import calculate_mlip_error
from mlipflow.strategies.structure_generators import StructureGenStrategy
from mlipflow.strategies.mlip import MLIPStrategy
from mlipflow.strategies.dft import QChemStrategy

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
    calculation_kwargs: NotRequired[dict[str, Any]]


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
                **calc_params
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
    
    # Get DFT kwargs from state or use defaults
    dft_kwargs = state.get('calculation_kwargs', {}).get('dft_scf', {})
    
    # Default parameters need to be handled carefully for run_chunked_qe_sp
    # It seems run_chunked_qe_sp takes explicit args 
    
    try:   
        run_chunked_qe_sp(
            in_file=configs,
            out_file=outfile,
            chunk_size=70,
            qchem_strategy=state['qchem_strategy'],
            kpts=dft_kwargs.get('kpts', (3,3,1)),
            dipole=dft_kwargs.get('dipole', False),
            dftd3=dft_kwargs.get('dftd3', False),
            ecut_eV=dft_kwargs.get('ecut_eV', 450),
            num_inputs_per_queued_job=dft_kwargs.get('num_inputs_per_queued_job', 1)
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
    
    # Get MLIP kwargs from state
    mlip_kwargs = state.get('calculation_kwargs', {}).get('mlip_sp', {})

    try:   
        run_single_point(
            in_file=configs,
            out_file=outfile,
            output_prefix=mlip.mlip_prefix,
            calculator=mlip.get_calculator(
                job_name='mSP_',
                dispersion=mlip_kwargs.get('dispersion', False)
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
    # Get MLIP kwargs for structure generation
    mlip_kwargs = state.get('calculation_kwargs', {}).get('mlip_gen', {})
    
    # Determine the property prefix based on the strategy type
    # MDGen -> last_op__md_
    # OPTGen -> last_op__optimize_
    prefix_map = {
        'md': 'md',
        'opt': 'optimize',
        'neb': 'neb' # Assuming standard naming, check checking wfl/neb
    }
    
    # Default to 'optimize' if unknown, but better to be safe
    # structure_generator.calc_prefix should be 'md', 'opt', etc.
    calc_type = getattr(structure_generator, 'calc_prefix', 'opt')
    op_name = prefix_map.get(calc_type, 'optimize')
    
    energy_key = f'last_op__{op_name}_energy'
    force_key = f'last_op__{op_name}_forces'

    outfile = [
        xyz.replace('.', f'_{structure_generator.calc_prefix}.') for xyz in configs
    ]
    try:
        structure_generator.generate_new_structures(
            in_file=configs,
            out_file=outfile,
            calculator=mlip.get_calculator(
                job_name='mSG_',
                dispersion=mlip_kwargs.get('dispersion', True)
                ),
            remote_info=mlip.remote_info
        )
        tag_dict={
            'info':{energy_key: f'{mlip.mlip_prefix}energy'},
            'array':{force_key: f'{mlip.mlip_prefix}forces'}
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