import os
import numpy as np
from ase.calculators.calculator import all_changes
from wfl.calculators.wfl_fileio_calculator import WFLFileIOCalculator
from wfl.generate.md.abort_base import AbortSimBase
from mace.calculators import MACECalculator
from quippy.potential import Potential

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
_default_keep_files = ["*.out"]
_default_properties = ["energy", "forces", "stress"]

# calculator classes for remote jobs
class GAPCalc(WFLFileIOCalculator, Potential):
    """

    """
    def __init__(self, keep_files='default', rundir_prefix='run_GAP_', workdir=None, scratchdir=None, **kwargs):

        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors
        super().__init__(keep_files=keep_files, rundir_prefix=rundir_prefix,
                         workdir=workdir, scratchdir=scratchdir, **kwargs)

    def calculate(self, atoms=None, properties=_default_properties, system_changes=all_changes):
        """
        """
        # from WFLFileIOCalculator
        self.setup_rundir()

        try:
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            calculation_succeeded = True
            if 'FAILED_GAP' in atoms.info:
                del atoms.info['FAILED_GAP']
        except Exception as exc:
            atoms.info['FAILED_GAP'] = True
            calculation_succeeded = False
            raise exc
#        finally:
#            # from WFLFileIOCalculator
#            self.clean_rundir(_default_keep_files, calculation_succeeded)   


class MACECalc(WFLFileIOCalculator, MACECalculator):
    """

    """
    def __init__(self, keep_files='default', rundir_prefix='run_MACE_', workdir=None, scratchdir=None, **kwargs):
        
        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors
        super().__init__(keep_files=keep_files, rundir_prefix=rundir_prefix,
                         workdir=workdir, scratchdir=scratchdir, **kwargs)

    def calculate(self, atoms=None, properties=_default_properties, system_changes=all_changes):
        """
        """
        if atoms is not None:
            self.atoms = atoms.copy()
        
        # from WFLFileIOCalculator
        #self.setup_rundir()

        try:
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            calculation_succeeded = True
            if 'FAILED_MACE' in atoms.info:
                del atoms.info['FAILED_MACE']
        except Exception as exc:
            atoms.info['FAILED_MACE'] = True
            calculation_succeeded = False
            raise exc
#        finally:
#            # from WFLFileIOCalculator
#            self.clean_rundir(_default_keep_files, calculation_succeeded)        


###############################################################################################################

# abort criteria class for MD-simulations
class ForceCheck(AbortSimBase):
    """

    """
    def __init__(self, threshold, n_failed_steps=10):
        super().__init__(n_failed_steps)      
        self.threshold = threshold

    def atoms_ok(self, at):
        #return self.threshold > np.max(np.sqrt(at.calc.extra_results['atoms']['local_gap_variance']) * 1e3)
        return self.threshold > np.max(np.abs(at.calc.get_forces()))


# trajectory selection function
def select_config(traj):
    """
    
    """
    if len(traj) < 10:
        return traj[-1:]
    
    elif len(traj) == 500:
        return traj[-1:]
    
    else:
        return traj[-10:-9]