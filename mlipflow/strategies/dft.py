import os, json
from abc import ABC, abstractmethod

#from ase import units
from ase.calculators.emt import EMT
from ase.calculators.espresso import EspressoProfile
from wfl.calculators.espresso import Espresso
from wfl.calculators.vasp import Vasp

from mlipflow.utils import prepare_remote, time_str_to_seconds
#####################################################################


# Base Strategy class for QChem method
class QChemStrategy(ABC):
    """
    Abstract base class for Quantum Chemistry strategies.
    
    Attributes:
        qe_prefix (str): Prefix for QE calculations.
    """
    def __init__(self) -> None:
        """Initialize the QChemStrategy."""
        self.qe_prefix = 'DFT_'

    @abstractmethod
    def get_calculator(self, job_name: str) -> tuple:
        """
        Get the calculator object.

        Args:
            job_name (str): Name of the job.

        Returns:
            tuple: (calculator_class, args, kwargs)
        """
        pass

# EMT strategy class for debugging
class EMTCalc(QChemStrategy):
    """
    EMT strategy class for debugging and testing.
    """
    def __init__(self) -> None:
        """Initialize the EMTCalc strategy."""
        super().__init__()
        self.remote_info = None

    def get_calculator(self, job_name: str, **kwargs) -> tuple:
        """
        Get the EMT calculator.

        Args:
            job_name (str): Name of the job (unused).
            **kwargs: Additional arguments to match QECalculator signature.

        Returns:
            tuple: (EMT class, None, dict with options)
        """
        return (EMT, None, {"fixed_cutoff": True})

# Quantum Espresso DFT-strategy class
class QECalculator(QChemStrategy):
    """
    Quantum Espresso DFT-strategy class.
    
    Attributes:
        basic_params (str): Path to basic parameters JSON file.
        pseudo_dir (str): Directory containing pseudopotentials.
        pseudopotentials (dict): Map of element to pseudopotential file.
    """
    def __init__(self, basic_params: str, pseudo_dir: str) -> None:
        """
        Initialize the QECalculator.

        Args:
            basic_params (str): Path to basic parameters JSON file.
            pseudo_dir (str): Directory containing pseudopotentials.
        """
        super().__init__()
        self.basic_params = basic_params
        self.pseudo_dir = pseudo_dir
        self.pseudopotentials = {
            'Cu': 'Cu.pbe-dn-kjpaw_psl.1.0.0.UPF', 
            'O' : 'O.pbe-n-kjpaw_psl.1.0.0.UPF',
            'C' : 'C.pbe-n-kjpaw_psl.1.0.0.UPF', 
            'H' : 'H.pbe-kjpaw_psl.1.0.0.UPF',
            'N': 'N.pbe-n-kjpaw_psl.1.0.0.UPF', 
            'Pd': 'Pd.pbe-n-kjpaw_psl.1.0.0.UPF',
            'K': 'K.pbe-spn-kjpaw_psl.1.0.0.UPF'
        }
        

    def get_calculator(self, job_name: str, ecut_eV: int = 450, kpts: tuple = (3,3,1),
                       calc_type: str = 'scf', dipole: bool = False, dftd3: bool = False, spin: bool = False,
                       num_inputs_per_queued_job: int = 2) -> tuple:
        """
        Get the Quantum Espresso calculator configuration.

        Args:
            job_name (str): Name of the job.
            ecut_eV (int): Energy cutoff in eV. Defaults to 450.
            kpts (tuple): K-points tuple. Defaults to (3,3,1).
            calc_type (str): Type of calculation ('scf' or 'relax'). Defaults to 'scf'.
            dipole (bool): Whether to include dipole correction. Defaults to False.
            dftd3 (bool): Whether to include DFT-D3 correction. Defaults to False.
            spin (bool): Whether to include spin polarization. Defaults to False.
            num_inputs_per_queued_job (int): Number of inputs per queued job. Defaults to 2.

        Returns:
            tuple: (Espresso class, [], dict with options)
        """
        # prepare remote settings based on calculation type
        assert calc_type in ['scf', 'relax'], f'Unknown calculation type: {calc_type}'
        remote_settings = {
            'scf': {
                'max_time': '02:00:00' if spin else '01:25:00',
                'n_cores': 48 if spin else 32,
                'num_inputs_per_queued_job': 1 if spin else num_inputs_per_queued_job,
                'job_name': job_name,
                'sys_name': 'local_qe'
            },
            'relax': {
                'max_time': '06:25:00',
                'n_cores': 32,
                'num_inputs_per_queued_job': 1,
                'job_name': job_name,
                'sys_name': 'local_qe'
            },
        }
        self.remote_info=prepare_remote(**remote_settings[calc_type])
        self.max_time_sec = time_str_to_seconds(remote_settings[calc_type]['max_time'])
        self.max_time_sec += 300 # add 5 minutes buffer

        # prepare input-data for QE
        input_data = self._prepare_params(ecut_eV=ecut_eV, calc=calc_type)
        assert input_data, 'parameters for QE missing'

        # update calculation type
        input_data['control']['calculation'] = calc_type

        if calc_type == 'scf':
            for para in ['nstep', 'etot_conv_thr', 'forc_conv_thr']:
                input_data['control'].pop(para)

        # modify default-input removing Dipole or D3-correction
        if dipole == False:
            dipole_paras = {'system': ['eamp', 'edir', 'emaxpos', 'eopreg'],
                            'control': ['dipfield', 'tefield']}
            for key in dipole_paras.keys():
                for para in dipole_paras.get(key):
                    input_data[key].pop(para)

        if dftd3 == False:
            dftd3_paras = {'system': ['dftd3_version', 'vdw_corr']}
            for key in dftd3_paras.keys():
                for para in dftd3_paras.get(key):
                    input_data[key].pop(para)

        if spin == True:
            input_data['system'].update(
                {
                    'nbnd': 628,
                    'nspin': 2, 
                    'starting_magnetization(1)': 0.263,    # for Cu     
                }
            )
        
        # set up ase-related QE-calculator
        profile = EspressoProfile(
            command='srun pw.x', 
            pseudo_dir=self.pseudo_dir
        )
        
        calculator = (Espresso, [], {
            'keep_files': None, 
            'rundir_prefix': 'QE_',
            'profile': profile,
            'input_data': input_data,
            'pseudopotentials':self.pseudopotentials,
            'kpts': kpts
            }
        )

        return calculator


    def _prepare_params(self, ecut_eV: int, calc: str, level: str = 'fine') -> dict:
        """
        Prepare input-file parameters for Quantum Espresso (QE) pw.x computation.

        Args:
            ecut_eV (int): Energy cutoff in eV.
            calc (str): Type of calculation, 'scf' or 'relax'.
            level (str): Level of precision, 'fine' or 'normal'. Defaults to 'fine'.

        Returns:
            dict: Dictionary of QE parameters.
        """        
        #ecut_Ry = ecut_eV * units.eV / units.Ry
        
        if level == 'fine':
            conv_thr = 7.4e-9   #60 * 2e-10
            ecut_Ry = 64.97
            kps = 0.2
            kpts = None
            degauss = 0.00735   #1.47e-02

        with open(self.basic_params, 'r') as f:
            qe_params = json.loads(f.read())

        qe_params['control'].update({
            'outdir': f"./files",
            'pseudo_dir': self.pseudo_dir,
            'max_seconds': self.max_time_sec
            }
        )

        qe_params['system'].update({
            'degauss': degauss,
            'ecutwfc': round(ecut_Ry, 2),
            'ecutrho': round(ecut_Ry * 8, 2)
            }
        )
        
        qe_params['electrons'].update({
            'conv_thr': conv_thr
            }
        )
        
        return qe_params
    

# VASP DFT-strategy class
#class VASPCalculator(QChemStrategy):
#    def __init__(self) -> None:
#        super().__init__()
#
#    def get_calculator(self, job_name, encut: int = 450, kpts: tuple = (3,3,1), calc_type: str = 'scf'):
#        """
#        """
#        # prepare remote settings
#        assert calc_type in ['scf', 'opt'], f'Unknown calculation type: {calc_type}'
#        remote_settings = {
#            'scf': {
#                'max_time': '01:25:00',
#                'n_cores': 32,
#                'num_inputs_per_queued_job': 3,
#                'job_name': job_name,
#                'sys_name': 'local_qe',
#                'pre_cmds': ['module load vasp6/6.3.0'],
#                'input_files': [],
#                'env_vars': [
#                    'ASE_VASP_COMMAND="run vasp_std"',
#                    'VASP_PP_PATH=./'
#                    ]
#            },
#            'opt': {
#                'max_time': '06:25:00',
#                'n_cores': 32,
#                'num_inputs_per_queued_job': 1,
#                'job_name': job_name,
#                'sys_name': 'local_qe'
#            },
#        }        
#        self.remote_info=prepare_remote(**remote_settings[calc_type])
#        
#        # prepare calculator
#        calculator = (Vasp, [], {
#            "calculator_exec": "srun vasp_std",
#            "encut": encut,
#            "kpts": kpts,
#            "ibrion": 2,
#            "xc": "PBE",
#            "nsw": 0,
#            "ediff": 1e-06,   # stopping-criterion for ELM 1e-06 default
#            "ediffg": -0.05,  # stopping-criterion for IOM (all forces smaller 0.05 eV/Å)
#            "ispin": 1,
#            "ismear": 1,
#            "sigma": 0.05,    # smearing in eV
#            "lreal": "Auto",
#            # "potim": 0.5,   # step for ionic-motion (for MD in fs) — commented out
#            "lwave": False,
#            "lcharg": False,
#            "isym": 0,
#            "ivdw": 12,
#            "algo": "Fast",
#            "prec": "Normal",
#            "nelm": 80,
#            "txt": "vasp.out",
#            }
#        )
#        return calculator