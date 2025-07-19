import os
from datetime import datetime
from wfl.autoparallelize import RemoteInfo
from expyre.resources import Resources

import expyre
expyre.config.init(root_dir=os.getcwd())


def time_str_to_seconds(time_str):
    t = datetime.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second


def prepare_remote(max_time: int, n_cores: int, num_inputs_per_queued_job: int, job_name: str,
                   pre_cmds: list[str] = [], post_cmds: list[str] = [], input_files: list[str] = [],
                   output_files: list[str] = [], sys_name: str = 'local') -> RemoteInfo:
    """
    Prepare the remote info for the job submission.
    Args:
        max_time (int): Maximum time for the job in seconds.
        n_cores (int): Number of cores to use.
        num_inputs_per_queued_job (int): Number of inputs per queued job.
        job_name (str): Name of the job.
        pre_cmds (list): List of commands to run before the job.
        post_cmds (list): List of commands to run after the job.
        input_files (list): List of input files for the job.
        output_files (list): List of output files for the job.
        sys_name (str): Name of the system to use as defined in expyre-config.
    
    Returns:
            wfl-RemoteInfo object.
    """
    
    remote_info = RemoteInfo(
        resources=Resources(
            max_time=max_time,
            num_cores=n_cores,
            max_mem_tot="32GB",
            partitions='work'
            ),
        sys_name=sys_name,
        num_inputs_per_queued_job=num_inputs_per_queued_job,
        job_name=job_name,
        input_files=input_files,
        output_files=output_files,
        pre_cmds=pre_cmds,
        post_cmds=post_cmds,
        exact_fit=False,
        partial_node=True,
        resubmit_killed_jobs=False,
        ignore_failed_jobs=True,
    )
    return remote_info