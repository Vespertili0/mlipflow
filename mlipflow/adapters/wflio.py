import os
import numpy as np
from ase.calculators.calculator import all_changes
from wfl.calculators.wfl_fileio_calculator import WFLFileIOCalculator
from wfl.generate.md.abort_base import AbortSimBase
from mace.calculators import MACECalculator
from quippy.potential import Potential

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
_default_keep_files = ["*.out"]
_default_properties = ["energy", "forces"] #, "stress"

# calculator classes for remote jobs
class GAPCalc(WFLFileIOCalculator, Potential):
    """
    Calculator class for GAP potentials that integrates with WFL workflows.
    """
    def __init__(self, keep_files: str = 'default', rundir_prefix: str = 'run_GAP_', 
                 workdir: str | None = None, scratchdir: str | None = None, **kwargs):
        """
        Initialize the GAPCalc.

        Args:
            keep_files (str): Files to keep. Defaults to 'default'.
            rundir_prefix (str): Prefix for the run directory. Defaults to 'run_GAP_'.
            workdir (str | None): Working directory. Defaults to None.
            scratchdir (str | None): Scratch directory. Defaults to None.
            **kwargs: Additional arguments for the calculator.
        """

        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors
        super().__init__(keep_files=keep_files, rundir_prefix=rundir_prefix,
                         workdir=workdir, scratchdir=scratchdir, **kwargs)

    def calculate(self, atoms: object = None, properties: list[str] = _default_properties, 
                  system_changes: list[str] = all_changes) -> None:
        """
        Perform the calculation.

        Args:
            atoms (ase.Atoms): Atoms object.
            properties (list[str]): List of properties to calculate.
            system_changes (list[str]): List of changes in the system.
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
    Calculator class for MACE potentials that integrates with WFL workflows
    """
    def __init__(self, keep_files: str = 'default', rundir_prefix: str = 'run_MACE_', 
                 workdir: str | None = None, scratchdir: str | None = None, **kwargs):
        """
        Initialize the MACECalc.

        Args:
            keep_files (str): Files to keep. Defaults to 'default'.
            rundir_prefix (str): Prefix for the run directory. Defaults to 'run_MACE_'.
            workdir (str | None): Working directory. Defaults to None.
            scratchdir (str | None): Scratch directory. Defaults to None.
            **kwargs: Additional arguments for the calculator.
        """
        
        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors
        super().__init__(keep_files=keep_files, rundir_prefix=rundir_prefix,
                         workdir=workdir, scratchdir=scratchdir, **kwargs)

    def calculate(self, atoms: object = None, properties: list[str] = _default_properties, 
                  system_changes: list[str] = all_changes) -> None:
        """
        Perform the calculation.

        Args:
            atoms (ase.Atoms): Atoms object.
            properties (list[str]): List of properties to calculate.
            system_changes (list[str]): List of changes in the system.
        """
        if atoms is not None:
            self.atoms = atoms.copy()
        
        # from WFLFileIOCalculator
        self.setup_rundir()

        try:
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            calculation_succeeded = True
            if 'FAILED_MACE' in atoms.info:
                del atoms.info['FAILED_MACE']
        except Exception as exc:
            atoms.info['FAILED_MACE'] = True
            calculation_succeeded = False
            raise exc
        finally:
            # from WFLFileIOCalculator
            self.clean_rundir(_default_keep_files, calculation_succeeded)        


###############################################################################################################

# abort criteria class for MD-simulations
class ForceCheck(AbortSimBase):
    """
    Abort criteria class for MD-simulations based on force threshold.
    """
    def __init__(self, threshold: float, n_failed_steps: int = 10):
        """
        Initialize the ForceCheck abort criteria.

        Args:
            threshold (float): Force threshold.
            n_failed_steps (int): Number of failed steps allowed. Defaults to 10.
        """
        super().__init__(n_failed_steps)      
        self.threshold = threshold

    def atoms_ok(self, at: object) -> bool:
        """
        Check if the atoms are within the force threshold.

        Args:
            at (ase.Atoms): Atoms object.

        Returns:
            bool: True if forces are within threshold, False otherwise.
        """
        #return self.threshold > np.max(np.sqrt(at.calc.extra_results['atoms']['local_gap_variance']) * 1e3)
        return self.threshold > np.max(np.abs(at.calc.get_forces()))


# trajectory selection function
def select_config(traj: list[object]) -> list[object]:
    """
    Select configurations from a trajectory.

    Args:
        traj (list[ase.Atoms]): Trajectory list of Atoms objects.

    Returns:
        list[ase.Atoms]: Selected configurations.
    """
    if len(traj) < 10:
        return traj[-1:]
    
    elif len(traj) == 500:
        return traj[-1:]
    
    else:
        return traj[-10:-9]