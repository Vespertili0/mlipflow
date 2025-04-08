import os
import numpy as np
from wfl.calculators.wfl_fileio_calculator import WFLFileIOCalculator
from wfl.generate.md.abort_base import AbortSimBase
from mace.calculators import MACECalculator
from quippy.potential import Potential
from wfl.autoparallelize import AutoparaInfo, RemoteInfo

from expyre.resources import Resources
import expyre
expyre.config.init(root_dir=os.getcwd())

# calculator classes for remote jobs
class GAPCalc(WFLFileIOCalculator, Potential):
    """

    """
    def __init__(self, keep_files, rundir_prefix, workdir=None, scratchdir=None, **kwargs):
        super().__init__(keep_files, rundir_prefix, workdir=None, scratchdir=None, **kwargs)

class MACECalc(WFLFileIOCalculator, MACECalculator):
    """

    """
    def __init__(self, keep_files, rundir_prefix, workdir=None, scratchdir=None, **kwargs):
        super().__init__(keep_files, rundir_prefix, workdir=None, scratchdir=None, **kwargs)



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
    

#########################################################################################

def prepare_remote(max_time, n_cores, num_inputs_per_queued_job, job_name,
                   pre_cmds=[], post_cmds=[], input_files=[], output_files=[],
                   sys_name='local')->RemoteInfo:
    """
    """
    remote_info = RemoteInfo(resources=Resources(max_time=max_time,
                                                 num_cores=n_cores,
                                                 max_mem_tot="32GB",
                                                 partitions='work'),
                             sys_name=sys_name,
                             num_inputs_per_queued_job=num_inputs_per_queued_job,
                             job_name=job_name,
                             input_files=input_files,
                             output_files=output_files,
                             pre_cmds=pre_cmds,
                             post_cmds=post_cmds,
                             exact_fit=False,
                             partial_node=True,
                             resubmit_killed_jobs=True,
                             ignore_failed_jobs=True,
                             )
    return remote_info