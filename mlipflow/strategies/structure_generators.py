from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from wfl.autoparallelize import AutoparaInfo
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate.neb import NEB as run_neb

from mlipflow.adapters.wflio import ForceCheck

#from wfl.generate.md import md as run_md
from mlipflow.core.md_nvt import md as run_md

#from wfl.generate.optimize import optimize as optimize
from mlipflow.core.relaxation_fire import optimize
from mlipflow.data import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
#####################################################################

# Base Strategy class for structure generator approach
class StructureGenStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.calc_prefix = "base"
        self.params = {}

    @abstractmethod
    def generate_new_structures(self):
        pass

# Strategy using MD as generator
class MDGen(StructureGenStrategy):
    def __init__(self, uncertainty_thrs,
                 n_failed_steps=10,
                 params={"steps": 500, "dt": 1, "temperature": 300., "traj_step_interval": 1}) -> None:
        super().__init__()
        self.uncertainty_thrs = uncertainty_thrs
        self.n_failed_steps = n_failed_steps
        self.params = params
        self.calc_prefix = "md"

    def generate_new_structures(self, in_file, out_file, calculator,
                                traj_select_after_func=None, remote_info=None) -> None:
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
            run_md(
                inputs=in_config,
                outputs=out_config,
                calculator=calculator,
                rng=rng,
                traj_select_after_func=traj_select_after_func,
                abort_check=abort_check,
                **self.params
            )

        elif remote_info:
            run_md(
                inputs=in_config,
                outputs=out_config,
                calculator=calculator,
                rng=rng,
                traj_select_after_func=traj_select_after_func,
                abort_check=abort_check,
                autopara_info=AutoparaInfo(
                    remote_info=remote_info,
                    num_inputs_per_python_subprocess=1,
                    num_python_subprocesses=1
                ),
                **self.params
            )

# Strategy using optimisation as generator
class OPTGen(StructureGenStrategy):
    def __init__(self, traj_subselect="last", params={"fmax": 0.005, "steps": 250}) -> None:
        super().__init__()
        self.traj_subselect = traj_subselect
        self.params = params
        self.calc_prefix = "opt"

    def generate_new_structures(self, in_file, out_file, calculator, remote_info=None) -> None:
        """
        Generates new configs via the wfl.generate_configs.optimize run function.

        Parameters
        ----------
        in_file:  list
            list of configs to be relaxed.
            #IMPORTANT: Constraints, such as fixed atoms need to be set previously
        out_file: str
            file in which the relaxation trajectories will be stored
        calculator: ase-calculator object
            calculator to be used for the relaxation
        remote_info: wfl-RemoteInfo object, optional

        **kwargs:
        In general: kwargs for the run function, the two "main" ones:
            "fmax": float, the force convergence criteria for the relaxation
            "steps": int, maximum permissible number of steps during the relaxation
        """
        in_config = ConfigSet(in_file)
        out_config = OutputSpec(out_file)

        if remote_info is None:
            optimize(
                inputs=in_config,
                outputs=out_config,
                calculator=calculator,
                traj_subselect=self.traj_subselect,
                **self.params
            )

        elif remote_info:
            optimize(
                inputs=in_config,
                outputs=out_config,
                calculator=calculator,
                traj_subselect=self.traj_subselect,
                autopara_info=AutoparaInfo(
                    remote_info=remote_info,
                    num_inputs_per_python_subprocess=1,
                    num_python_subprocesses=1
                ),
                **self.params
            )

# Strategy using CI-NEB as generator
class NEBGen(StructureGenStrategy):
    def __init__(self, traj_subselect="last", params={"fmax": 0.05, "steps": 250}, n_images=9) -> None:
        super().__init__()
        self.traj_subselect = traj_subselect
        self.params = params
        self.n_images = n_images
        self.calc_prefix = "neb"

    def generate_new_structures(self, in_file, out_file, calculator, remote_info=None) -> None:
        """
        Generates new configs via the wfl.generate.neb.NEB run function.

        Parameters
        ----------
        in_file:  list/ConfigSet
            list of configs (bands) to be relaxed.
        out_file: str
            file in which the relaxation trajectories will be stored
        calculator: ase-calculator object
        remote_info: wfl-RemoteInfo object, optional

        **kwargs:
        neb_params will be passed as kwargs
        """
        in_config = list(ConfigSet(in_file))
        n_configs = len(in_config)
        neb_sets = [in_config[i:i+self.n_images] for i in range(0, n_configs, self.n_images)]
        out_config = OutputSpec(out_file)

        if remote_info is None:
            run_neb(
                inputs=neb_sets,
                outputs=out_config,
                calculator=calculator,
                traj_subselect=self.traj_subselect,
                **self.params
            )

        elif remote_info:
            run_neb(
                inputs=neb_sets,
                outputs=out_config,
                calculator=calculator,
                traj_subselect=self.traj_subselect,
                autopara_info=AutoparaInfo(
                    remote_info=remote_info,
                    num_inputs_per_python_subprocess=1,
                    num_python_subprocesses=1
                ),
                **self.params
            )
