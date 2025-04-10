import os, json
from abc import ABC, abstractmethod

from ase import units
from ase.calculators.emt import EMT
from ase.calculators.espresso import EspressoProfile
from wfl.calculators.espresso import Espresso

from mlipflow.utils import prepare_remote
#####################################################################


# Base Strategy class for QChem method
class QChemStrategy(ABC):
    def __init__(self) -> None:
        self.qe_prefix = 'DFT_'

    @abstractmethod
    def get_calculator(self):
        pass

# EMT strategy class for debugging
class EMTCalc(QChemStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.remote_info = None

    def get_calculator(self, job_name):
        return (EMT, None, {"fixed_cutoff": True})

# Quantum Espresso DFT-strategy class
class QECalculator(QChemStrategy):
    def __init__(self, basic_params: str, pseudo_dir: str, max_time_sec: int) -> None:
        super().__init__()
        self.basic_params = basic_params
        self.pseudo_dir = pseudo_dir
        self.max_time_sec = max_time_sec
        self.pseudopotentials = {
            'Cu': 'Cu.pbe-dn-kjpaw_psl.1.0.0.UPF', 
            'O' : 'O.pbe-n-kjpaw_psl.1.0.0.UPF',
            'C' : 'C.pbe-n-kjpaw_psl.1.0.0.UPF', 
            'H' : 'H.pbe-kjpaw_psl.1.0.0.UPF',
            'N': 'N.pbe-n-kjpaw_psl.1.0.0.UPF', 
            'Pd': 'Pd.pbe-n-kjpaw_psl.1.0.0.UPF'
        }
        

    def get_calculator(self, job_name, ecut_eV: int = 450, kpts: tuple = (3,3,1),
                       dipole: bool = True, dftd3: bool = True):
        """
        """
        self.remote_info=prepare_remote(
            max_time='01:15:00',
            n_cores=32,
            num_inputs_per_queued_job=3,
            job_name=job_name,
            sys_name='local_qe'
        )
        
        input_data = self._prepare_params(ecut_eV=ecut_eV)

        assert input_data, 'parameters for QE missing'

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


    def _prepare_params(self, ecut_eV: int, level='fine') -> dict:
        """
        modifying input-file for Quantum Espresso (QE) pw.x computation

        Args:
            ecut      :  int in eV
        """        
        ecut_Ry = ecut_eV * units.eV / units.Ry
        
        if level == 'fine':
            conv_thr = 60 * 2e-10
            kps = 0.2
            kpts = None
            degauss = 1.47e-02

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