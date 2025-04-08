import os
from abc import ABC, abstractmethod
import numpy as np
from wfl.calculators.generic import calculate as generic_calc
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize import AutoparaInfo

from wfl.generate.md import md as sample_md
from wfl.generate.optimize import optimize as optimize
from mlipflow.wfl_potentials import ForceCheck, select_config
#####################################################################

# Base Strategy class for structure generator approach
class StructureGenStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate_new_structures(self):
        pass

# Strategy using MD as generator
class MDGen(StructureGenStrategy):
    def __init__(self, uncertainty_thrs, 
                 n_failed_steps=10, 
                 md_params={'steps': 500, 'dt': 1, 'temperature': 300., 'traj_step_interval': 1}) -> None:
        super().__init__()
        self.uncertainty_thrs = uncertainty_thrs
        self.n_failed_steps = n_failed_steps
        self.md_params = md_params
    
    def generate_new_structures(self, in_file, out_file, calculator, remote_info=None) -> None:
        """
        Generates new configs via the wfl.generate_configs.md sample function.

        Parameters
        ----------
        md.sample runs an MD and samples structures based on dt and steps,
        Next step is implementing furthest point sampling
        (Check md.py file in wfl -> generate_configs for details)
        parameters to set MD can be defined as arguments
        (NPT/NVT, temp, pressure, etc)
        """
        in_config = ConfigSet(in_file)
        out_config = OutputSpec(out_file)
        rng = np.random.default_rng(1)
        
        # running MD
        abort_check = ForceCheck(threshold=self.uncertainty_thrs,
                                 n_failed_steps=self.n_failed_steps)
        
        if remote_info is None:
            sample_md(
                inputs=in_config, 
                outputs=out_config, 
                calculator=calculator, 
                rng=rng, 
                traj_select_after_func=select_config, 
                abort_check=abort_check, 
                **self.md_params
            )
        
        elif remote_info:
            sample_md(
                inputs=in_config, 
                outputs=out_config, 
                calculator=calculator,  
                rng=rng, 
                traj_select_after_func=select_config, 
                abort_check=abort_check,
                autopara_info=AutoparaInfo(
                    remote_info=remote_info,
                    num_inputs_per_python_subprocess=1
                    ),
                **self.md_params
            )

# Strategy using optimisation as generator
class OPTGen(StructureGenStrategy):
    def __init__(self, traj_subselect="last_converged") -> None:
        super().__init__()
        self.traj_subselect = traj_subselect

    def generate_new_structures(self, in_file, out_file, calculator,
                                remote_info=None,  **kwargs) -> None:
        """
        Generates new configs via the wfl.generate_configs.optimize run function.

        Parameters
        ----------
        atoms:  list
            list of configs to be relaxed.
            #IMPORTANT: Constraints, such as fixed atoms need to be set previously
        out_file: str
            file in which the relaxation trajectories will be stored
        gap_file: str
            Path to GAP parameter file which we will use to create the calculator

        **kwargs:
        In general: kwargs for the run function, the two "main" ones:
            "fmax": float, the force convergence criteria for the relaxation
            "steps": int, maximum permissible number of steps during the relaxation
        """
        in_config = ConfigSet(in_file)
        out_config = OutputSpec(out_file)
        fmax=1e-3

        if remote_info is None:
            optimize(
                in_config,
                out_config,
                calculator=calculator, 
                traj_subselect=self.traj_subselect,
                fmax=fmax,
                **kwargs
                )

        elif remote_info:
            optimize(
                in_config,
                out_config, 
                calculator=calculator,
                fmax=fmax,
                traj_subselect=self.traj_subselect,
                autopara_info=AutoparaInfo(
                    remote_info=remote_info,
                    num_inputs_per_python_subprocess=1
                    ),
                **kwargs
                )

# Strategy using CI-NEB as generator
class NEBGen(StructureGenStrategy):
    def __init__(self) -> None:
        super().__init__()

    def generate_new_structures(self) -> None:
        pass