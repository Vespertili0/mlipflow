import os
from datetime import datetime
from wfl.autoparallelize import RemoteInfo
from expyre.resources import Resources
import expyre
expyre.config.init(root_dir=os.getcwd())


def time_str_to_seconds(time_str: str) -> int:
    """
    Convert a time string in format HH:MM:SS to seconds.

    Args:
        time_str (str): Time string in format "HH:MM:SS".

    Returns:
        int: Total seconds.
    """
    t = datetime.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second


def prepare_remote(max_time: int | str, n_cores: int, num_inputs_per_queued_job: int, job_name: str, 
                   env_vars: list[str] = [], pre_cmds: list[str] = [], post_cmds: list[str] = [], 
                   input_files: list[str] = [], output_files: list[str] = [], sys_name: str = 'local') -> RemoteInfo:
    """
    Prepare the remote info for the job submission.

    Args:
        max_time (int | str): Maximum time for the job in seconds or "HH:MM:SS" string.
        n_cores (int): Number of cores to use.
        num_inputs_per_queued_job (int): Number of inputs per queued job.
        job_name (str): Name of the job.
        env_vars (list[str]): List of environment variables to set.
        pre_cmds (list[str]): List of commands to run before the job.
        post_cmds (list[str]): List of commands to run after the job.
        input_files (list[str]): List of input files for the job.
        output_files (list[str]): List of output files for the job.
        sys_name (str): Name of the system to use as defined in expyre-config.

    Returns:
        wfl.autoparallelize.RemoteInfo: The configured RemoteInfo object.
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
        env_vars=env_vars,
        exact_fit=False,
        partial_node=True,
        resubmit_killed_jobs=False,
        ignore_failed_jobs=True,
    )
    return remote_info