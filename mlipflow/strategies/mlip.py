import os, itertools, json, torch
from abc import ABC, abstractmethod

from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.calculators.mixing import SumCalculator

from universalSOAP import SOAP_hypers
from quippy.potential import Potential
from wfl.fit.gap.multistage import fit as multi_gap_fit
from wfl.configset import ConfigSet, OutputSpec
from wfl.fit.mace import fit as mace_fit
from mace.calculators import MACECalculator

from mlipflow.utils import prepare_remote
from mlipflow.adapters.wflio import GAPCalc
#####################################################################

# Base Strategy class for MLIP approach
class MLIPStrategy(ABC):
    """
    Base class for MLIP strategy classes. It defines the interface for MLIPs implemented in the MLIPFlow package.
    
    Attributes:
        mlip_name (str): Name of the MLIP.
        run_mode (str): Run mode ('local' or 'remote').
    """
    def __init__(self, mlip_name: str, run_mode: str) -> None:
        """
        Initialise the MLIPStrategy.

        Args:
            mlip_name (str): Name of the MLIP.
            run_mode (str): Run mode. Must be 'local' or 'remote'.
        """
        assert run_mode in ['local', 'remote'], 'run_mode is "local" or "remote"'
        self.run_mode = run_mode
        self.mlip_name = mlip_name


    @abstractmethod
    def get_calculator(self, job_name: str) -> tuple:
        """
        Get the calculator object.

        Args:
            job_name (str): Name of the job.

        Returns:
            tuple: Calculator configuration.
        """
        pass  
    
    @abstractmethod
    def fit_new_model(self, in_file: str, model_name: str, run_dir: str) -> None:
        """
        Fit a new model.

        Args:
            in_file (str): Input file path.
            model_name (str): Name of the model.
            run_dir (str): Run directory.
        """
        pass


# MACE strategy class
class MACEModel(MLIPStrategy):
    """
    MACE strategy class for MLIPFlow. It defines the interface for MACE implemented in the MLIPFlow package.
    
    Attributes:
        mlip_file (str): File name of the MACE model.
        run_mode (str): 'local' or 'remote'. If 'local', the MACE model is run locally in current job. 
                        If 'remote', the MACE model is submitted to run via wfl-RemoteInfo object.
    
    """
    def __init__(self, mlip_name: str, mace_config: str | None = None, run_mode: str = 'remote') -> None:
        """
        Initialise the MACEModel.

        Args:
            mlip_name (str): Name of the MLIP.
            mace_config (str | None): Path to MACE configuration file. Defaults to None.
            run_mode (str): Run mode ('local' or 'remote'). Defaults to 'remote'.
        """
        super().__init__(run_mode=run_mode, mlip_name=mlip_name)
        self.mlip_prefix = 'MACE_'
        self.mace_config = mace_config
        self.model_file = f"{mlip_name}.model"

    def get_calculator(self, job_name: str, dispersion: bool = True, dtype: str = 'float32', n_cores: int = 1,
                       max_time: str = '00:10:00', num_inputs_per_queued_job: int = 4) -> tuple:
        """
        It returns the MACE-calculator as tuple with the ase-calculator class, arguments and keyword arguments.
        Creates the remote_info object if run_mode is 'remote'.

        Args:
            job_name (str): Name of the job.
            dispersion (bool): Whether to include dispersion. Defaults to True.
            max_time (str): Maximum time for remote job. Defaults to '00:10:00'.
            dtype (str): Data type. Defaults to 'float32'.

        Returns:
            tuple: Calculator configuration.
        """
        if self.run_mode == 'local':
            self.remote_info = None

        elif self.run_mode == "remote":
            self.remote_info = prepare_remote(
                max_time=max_time, 
                n_cores=n_cores,
                num_inputs_per_queued_job=num_inputs_per_queued_job,
                job_name=job_name,
#                pre_cmds=['export WFL_NUM_PYTHON_SUBPROCESSES=4']
#                sys_name='local_mace'
            )
              
        if dispersion == False:
            calculator = (
                MACECalculator,
                [], 
                {
                    'model_paths': os.path.abspath(self.model_file),
                    'device': 'cpu',
                    'default_dtype': dtype,
                #    'dispersion': dispersion
                }
            )
        
        elif dispersion == True:
            mace_calc = MACECalculator(
                model_paths=os.path.abspath(self.model_file),
                device='cpu',
                default_dtype=dtype,
            )
            dftd3_calc = TorchDFTD3Calculator(
                damping='bj',   #"zero", "bj", "zerom", "bjm"
                old=False,      # False = use DFTD3 method, True = DFTD2
                device='cpu',
                dtype=torch.float32 if dtype == "float32" else torch.float64
            )
            calculator = (SumCalculator, [], {'calcs': [mace_calc, dftd3_calc]})
            
        return calculator 
    

    def fit_new_model(self, in_file: str, test_configs: str, ref_property_prefix: str = 'DFT_',
                      seed: int = 123, restart: bool = False, n_cores: int = 128, max_time: str = '10:00:00') -> None:
        """
        Run the wfl.fit.mace.fit function.

        Args:
            in_file (str): Path to file containing the input configs for the MACE fit.
            test_configs (str): Path to file containing the test configs.
            ref_property_prefix (str): Prefix for reference properties. Defaults to 'DFT_'.
            seed (int): Random seed. Defaults to 123.
            restart (bool): Whether to restart from latest. Defaults to False.
            n_cores (int): Number of cores. Defaults to 128.
            max_time (str): Maximum time for job. Defaults to '10:00:00'.

        Returns:
            None: The selected files are written in the defined directory.
        """
        # load MACE fitting parameters from JSON file
        with open(self.mace_config) as param_json:
            mace_params = json.load(param_json)
        
        # set random seed and restart option
        mace_params['seed'] = seed
        if restart:
            mace_params['restart_latest'] = None

        # run the MACE fitting
        mace_fit(
            fitting_configs=ConfigSet(in_file),
            test_configs=ConfigSet(test_configs),
            mace_name=self.mlip_name, 
            mace_fit_params=mace_params,
            ref_property_prefix=ref_property_prefix,
        #    run_dir=run_dir,            
        #    valid_configs=ConfigSet(valid_configs),
            remote_info=prepare_remote(
                max_time=max_time,
                n_cores=n_cores,
                num_inputs_per_queued_job=1,
                job_name='MACEfit',
                sys_name='local_mace'
            )
        )
        

# GAP strategy class    
class GAPModel(MLIPStrategy):
    """
    GAP strategy class for MLIPFlow. It defines the interface for GAP implemented in the MLIPFlow package.
    
    Attributes:
        mlip_file (str): file name of the GAP model.
        run_mode (str): 'local' or 'remote'. If 'local', the GAP model is run locally in current job. 
                        If 'remote', the GAP model is submitted to run via wfl-RemoteInfo object.
    
    """
    def __init__(self, mlip_file: str, run_mode: str = 'remote') -> None:
        """
        Initialise the GAPModel.

        Args:
            mlip_file (str): File name of the GAP model.
            run_mode (str): Run mode ('local' or 'remote'). Defaults to 'remote'.
        """
        super().__init__(run_mode=run_mode, mlip_file=mlip_file)
        self.mlip_prefix = 'GAP'

        # defining GAP-specific variables
        self.Zs = [1, 6, 8, 29]
        self.length_scales = {
            1: {"bond_len": [1.2, "NB VASP auto_length_scale"],
                "min_bond_len": [0.75, "NB VASP auto_length_scale"],
                "other links": {},
                "vol_per_atom": [3.4, "NB VASP auto_length_scale"]
                },
            6: {"bond_len": [1.4, "NB VASP auto_length_scale"],
                "min_bond_len": [1.3, "NB VASP auto_length_scale"],
                "other links": {},
                "vol_per_atom": [5.7, "NB VASP auto_length_scale"]
                },
            8: {"bond_len": [1.7, "NB VASP auto_length_scale"],
                "min_bond_len": [1.2, "NB VASP auto_length_scale"],
                "other links": {},
                "vol_per_atom": [11, "NB VASP auto_length_scale"]
                },
            29: {
                "bond_len": [2.6, "NB VASP auto_length_scale"],
                "min_bond_len": [2.2, "NB VASP auto_length_scale"],
                "other links": {},
                "vol_per_atom": [12, "NB VASP auto_length_scale"]
                }
            }

    def get_calculator(self, job_name: str) -> tuple:
        """
        Get the calculator object.

        Args:
            job_name (str): Name of the job.

        Returns:
            tuple: Calculator configuration.
        """
        if self.run_mode == 'local':
            calculator = (Potential, [], {
                'param_filename': self.mlip_file,
                'calc_args': 'local_gap_variance'}
            )
            self.remote_info = None
    
        elif self.run_mode == 'remote':
            calculator = (GAPCalc, [], {
                'keep_files': None,
                'rundir_prefix': 'GAP_',
                'param_filename': os.path.abspath(self.mlip_file),
                'calc_args':'local_gap_variance'
                }
            )
            
            self.remote_info = prepare_remote(
                max_time='02:00:00', 
                n_cores=1,
                num_inputs_per_queued_job=5,
                job_name=job_name,
                pre_cmds=['export OMP_NUM_THREADS=1']
            )
            
        return calculator


    def fit_new_model(self, in_file: str, model_name: str, run_dir: str) -> None:
        """
        Run the wfl.fit.gap_multistage fit function.

        Args:
            in_file (str): Path to file containing the input configs for the GAP fit.
            model_name (str): File name of written GAP.
            run_dir (str): Name of the directory in which the GAP files will be written.

        Returns:
            None: The selected configs are written in the out_file.
        """
        in_config = ConfigSet(in_file)

        #gap_params = prep_params(Zs=Zs, length_scales=length_scales, GAP_mlipflow=params)
        gap_params = self._get_multistage_params()
        multi_gap_fit(
            fitting_configs=in_config,
            GAP_name=model_name,
            params=gap_params,
            run_dir=run_dir, 
            ref_property_prefix="DFT_"
            )
        return None    


    def _get_multistage_params(self, stage_list: list[str] = ['2B','SOAP']) -> dict:
        """
        Get the multistage GAP parameters.

        Args:
            stage_list (list[str]): List of stages to include. Defaults to ['2B', 'SOAP'].

        Returns:
            dict: Multistage parameters dictionary.
        """
        # default settings for Two-Body
        atom_list = list(itertools.combinations_with_replacement(self.Zs, 2))
        desc_2B = {
            'distance_Nb': True, 
            'order': 2, 
            'cutoff':5.0
            }
        fit_2B = {
            'n_sparse':50,
            'theta_uniform': 1.0,
            'covariance_type': 'ard_se',
            'sparse_method': 'uniform'
            }
        
        # default settings for Many-body SOAPs
        hypers = SOAP_hypers(
            Zs=self.Zs, 
            length_scales=self.length_scales, 
            spacing=1.5, 
            no_extra_inner=True, 
            no_extra_outer=True
            )
        desc_MB = {
            'soap': True,
            'n_max': 9,
            'l_max': 3
            }
        fit_MB = {
            'zeta': 4,
            'covariance_type': 'dot_product',
            'sparse_method': 'CUR_POINTS'
            }

        stages = []
        for stage in stage_list:
            stage_dict = {'error_scale_factor': (lambda x: 1.0 if 'SOAP' in x else 10.0)(stage)}
            stage_desc = list()
        # prepare Two-Body      
            if '2B' in stage:
                for pair in atom_list:
                    pair_dict = {}
                    basic_2B = desc_2B.copy()
                    add_2B = {'Z': [pair[0], pair[1]]}
                    basic_2B.update(add_2B)
                    pair_dict.update({'descriptor': basic_2B})
                    pair_dict.update({'fit': fit_2B, 'add_species': False})
                    stage_desc.append(pair_dict)

        # prepare Many-Body SOAPs
            if 'SOAP' in stage:    
                for Z in self.Zs:
                    Z_dict = {}
                    basic_SOAP = desc_MB.copy()
                    basic_SOAP.update(hypers[Z][0])
                    basic_SOAP.update({'n_species': len(self.Zs), 'Z': Z}) , # 'delta': 0.03,
                    basic_SOAP.update({'species_Z': self.Zs})
                    Z_dict.update({'descriptor': basic_SOAP})
                    
                    fit_SOAP = fit_MB.copy()
                    fit_SOAP.update({'n_sparse': (lambda x: 100 if x!= 29 else 50)(Z)}) #!!! DEBUGGING
                    Z_dict.update({'fit': fit_SOAP})
                    Z_dict.update({'add_species': False})
                    stage_desc.append(Z_dict)

            stage_dict.update({'descriptors': stage_desc})
            stages.append(stage_dict)
        
        params_dict = {
            'default_sigma': [0.001, 0.01, 0.0, 0.0],
            'sparse_jitter': 1e-08,
            'do_copy_at_file': False,
            'sparse_separate_file': False,
            'energy_parameter_name': 'DFT_energy',
            'force_parameter_name': 'DFT_forces'
            }
        
        multi_params = {'stages': stages,
                        'gap_params': params_dict}
        
        return multi_params
