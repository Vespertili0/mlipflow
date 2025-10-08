import os
import shutil
import numpy as np
from ase.io import read, write
from wfl.configset import ConfigSet, OutputSpec

#def create_folder_structure(workdir: str, fit_idx: int) -> tuple[str, str, str, str]:
#    """Create folder structure for MLIP training."""
#    ensemble_dir = os.path.join(workdir, 'ENSEMBLE')
#    iter_dir = os.path.join(workdir, f'{fit_idx}_iteration')
#    mlip_dir = os.path.join(iter_dir, 'MLIP')
#    sgen_dir = os.path.join(iter_dir, 'SGEN')
#
#    for folder in [ensemble_dir, sgen_dir, mlip_dir]:
#        os.makedirs(folder, exist_ok=True)
#
#    return ensemble_dir, iter_dir, mlip_dir, sgen_dir

#def setup_iteration(workdir: str, fit_idx: int) -> dict:
#    """Set up the folder structure for a new iteration of the MLIP training."""
#    ensemble_dir, iter_dir, mlip_dir, sgen_dir = create_folder_structure(workdir, fit_idx)
#    files = {}
#
#    f1 = lambda x, y: f"{sgen_dir}/{x}{y}_{fit_idx}.xyz"
#    files["dft_0th"] = f1("dft", '')
#    files["mlip_output"] = f1("sgen", '_mlip')
#    files["dft_output"] = f1("sgen", '_dft')
#    files["eval"] = f1("eval", '')
#    files["eval_train"] = f1("eval", '_train')
#    files["eval_test"] = f1("eval", '_test')
#
#    files["training"] = os.path.join(mlip_dir, f'training_{fit_idx}.xyz')
#    files["ensemble_traj"] = os.path.join(ensemble_dir, 'ensemble.traj')
#    files["ensemble_xyz"] = os.path.join(ensemble_dir, 'ensemble.xyz')
#
#    f2 = lambda x: f"{mlip_dir}/GAP_{fit_idx}.xml{x}"
#    files["desc"] = f1("desc", '')
#    files["fps"] = f1("sgen", '_fps')
#    files["training_desc"] = f1("training_desc", "")
#    files["gap_params"] = f2(".descriptor_dicts.yaml")
#
#    return files

def get_model_name(mlip_dir: str, fit_idx: int, mlip_prefix: str) -> str:
    """Get the model name based on the MLIP prefix and fit index."""
    model_fmt = {'MACE': 'model', 'GAP': 'xml'}.get(mlip_prefix)
    return os.path.join(mlip_dir, f'{mlip_prefix}_{fit_idx}.{model_fmt}')


def update_training_data(training_xyz: str, add_xyz: str, out_file: str) -> None:
    """Update training data by merging two XYZ files."""
    new_training = read(training_xyz, ':') 
    new_training += read(add_xyz, ':')
    update_configset_tag(new_training, out_file, {'data_type': 'train'})


def initialise_ensembles(ensemble_traj: str, files: dict) -> None:
    """Initialize ensemble from trajectory file."""
    if not os.path.exists(ensemble_traj):
        raise FileNotFoundError(f"Ensemble trajectory file {ensemble_traj} not found")
    try:
        shutil.copy2(ensemble_traj, files.get("ensemble_traj"))
        configs = read(files.get("ensemble_traj"), ':')
        write(files.get("ensemble_xyz"), configs)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ensembles: {str(e)}")


def move_mace_model_file(mlip_dir: str, file_prefix: str) -> None:
    """Move the compiled MACE model file to the MLIP directory."""
    shutil.copy2(
        os.path.join(mlip_dir, 'MACE_model', f'{file_prefix}_compiled.model'),
        os.path.join(mlip_dir, f'{file_prefix}.model')
    )


def check_maxforce_and_cleanarrays(in_file: str, out_file: str, mlip_prefix: str, calc: str, max_force: float = 15) -> None:
    """Remove structures from ase-file with forces exceeding threshold."""
    keys = {'md': 'md', 'opt': 'optimize'}
    array_keys = ['numbers', 'positions', 'tags', 'DFT_forces', f'last_op__{keys.get(calc)}_forces']
    info_keys = ['MD_step', 'DFT_energy', f'last_op__{keys.get(calc)}_energy', 'config_type', 'data_type']

    all_configs = [a for a in read(in_file, ':') if 'DFT_energy' in a.info.keys()]
    
    assert all_configs, 'no valid structures from DFT! check data'
    
    selected_configs = [a for a in all_configs if np.max(np.abs(a.arrays['DFT_forces'])) <= max_force] 
    for config in selected_configs: 
        config.arrays = {
            (f'{mlip_prefix}_forces' if key == f'last_op__{keys.get(calc)}_forces' else key): config.arrays.get(key) for key in array_keys
        }
        config.info = {
            (f'{mlip_prefix}_energy' if key == f'last_op__{keys.get(calc)}_energy' else key): config.info.get(key) for key in info_keys
        }

    write(out_file, selected_configs)


def update_configset_tag(in_config, out_file, tag_dict) -> None:
    """Update a wfl.configset with a new tag."""
    configs = ConfigSet(in_config)
    OutputSpec(out_file, tags=tag_dict, overwrite=True).write(configs)


def filter_info_dict(info_dict: dict, keep_info: list) -> dict:
    """Filter the info dictionary to keep only specified keys."""
    return {key: info_dict[key] for key in keep_info if key in info_dict}


def split_success_failed_configs(configs: list, key: str = 'DFT_energy') -> tuple[list, list]:
    """Split configurations into successful and failed based on a key in the info dictionary."""
    success_configs = []
    failed_configs = []
    for config in configs:
        if key not in config.info.keys():
            failed_configs.append(config)
        else:
            success_configs.append(config)

    return success_configs, failed_configs


def merge_clean_chunks(in_files: list, out_file: str) -> None:
    """Merge all chunks into one file and clean up the data.info."""
    success_configs = []
    failed_configs = []
    for file in in_files:
        success, failed = split_success_failed_configs(
            configs=read(file, ':'),
            key='DFT_energy' 
        )
        for config in success:
            config.info = filter_info_dict(
                info_dict=config.info,
                keep_info=['DFT_energy']
            )
        success_configs += success
        failed_configs += failed
    OutputSpec(out_file, tags={'data_type': 'train'}).write(ConfigSet(success_configs))
    name, ext = os.path.splitext(out_file)
    OutputSpec(f'{name}_failed{ext}').write(ConfigSet(failed_configs))


def clean_up(key: str = '_chunk_') -> None:
    """
    Clean up directories and files matching the key.
    
    Args:
        key (str): Substring to identify files/directories to remove.
    Raises:
        RuntimeError: If an error occurs during removal.
    Returns:
        None
    """
    try:
        run_dirs = [rd for rd in os.listdir() if key in rd]
        for rd in run_dirs:
            if os.path.isdir(rd):
                shutil.rmtree(rd)
            elif os.path.isfile(rd):
                os.remove(rd)
    except Exception as e:
        raise RuntimeError(f"Error removing {rd}: {e}")