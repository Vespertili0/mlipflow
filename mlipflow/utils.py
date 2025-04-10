import os
from wfl.autoparallelize import RemoteInfo
from expyre.resources import Resources

import expyre
expyre.config.init(root_dir=os.getcwd())

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