import numpy as np
from ase.io import read, write
from ase import Atoms
from wfl.configset import ConfigSet, OutputSpec
from wfl.map import map as wfl_map

def update_training_data(training_xyz: str, add_xyz: str, out_file: str) -> None:       #!!! modify to ConfigSet
    """Update training data by merging two XYZ files."""
    new_training = read(training_xyz, ':') 
    new_training += read(add_xyz, ':')
    update_configset_tag(new_training, out_file, {'data_type': 'train'}, tag_type='info')


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


def _rename_configset_tags(at, old_tag, new_tag, tag_type='info') -> Atoms:
    """Function to rename tags in a wfl.configset ConfigSet object using wfl.map-function"""
    assert tag_type in ['array', 'info'], 'invalid tag'
    if tag_type == 'array':
        at.arrays = {
            (new_tag if key == old_tag else key): v for key, v in at.arrays.items()
        }
    elif tag_type == 'info':
        at.info = {
            (new_tag if key == old_tag else key): v for key, v in at.info.items()
        }
    return at


def update_configset_tag(in_config, out_file, tag_dict, tag_type) -> None:
    """Update tags in a wfl.configset ConfigSet object using wfl.map-function."""
    configs = ConfigSet(in_config)
    OutputSpec(out_file, overwrite=True).write(configs)
    output = OutputSpec(out_file, overwrite=True)
    
    for old_tag, new_tag in tag_dict.items():
        wfl_map(
            inputs=configs, 
            outputs=output,
            map_func=_rename_configset_tags,
            args=[old_tag, new_tag, tag_type]
        )


def add_configset_tag(in_config, out_file, tag_dict) -> None:
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
    import os
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


def _clean_atoms_attributes(at: Atoms, keep_info_keys: list, keep_array_keys: list) -> Atoms:
    """
    Helper function to clean atoms.info and atoms.arrays dictionaries.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object to clean.
    keep_info_keys : list
        List of keys to keep in atoms.info.
    keep_array_keys : list
        List of keys to keep in atoms.arrays.
        
    Returns
    -------
    ase.Atoms
        Cleaned Atoms object.
    """
    # Clean info
    at.info = {k: v for k, v in at.info.items() if k in keep_info_keys}
    
    # Clean arrays
    at.arrays = {k: v for k, v in at.arrays.items() if k in keep_array_keys}
    
    return at


def clean_configset_data(inputs, outputs, keep_info_keys=None, keep_array_keys=None):
    """
    Clean the info and array attributes of a configset.
    
    Parameters
    ----------
    inputs : wfl.configset.ConfigSet or list(Atoms) or list(str)
        Input configurations.
    outputs : wfl.configset.OutputSpec or str
        Output configuration.
    keep_info_keys : list, optional
        List of keys to keep in atoms.info. Default is ['slab', 'species'].
    keep_array_keys : list, optional
        List of keys to keep in atoms.arrays. Default is ['numbers', 'positions', 'tags'].
        
    Returns
    -------
    wfl.configset.OutputSpec
        The output specification containing the processed configurations.
    """
    if keep_info_keys is None:
        keep_info_keys = ['slab', 'species']
    if keep_array_keys is None:
        keep_array_keys = ['numbers', 'positions', 'tags']
        
    # Merge defaults with user provided lists to ensure minimum keys are always kept
    default_info = {'slab', 'species'}
    default_arrays = {'numbers', 'positions', 'tags'}
    
    keep_info_keys = list(set(keep_info_keys) | default_info)
    keep_array_keys = list(set(keep_array_keys) | default_arrays)

    return wfl_map(
        inputs=ConfigSet(inputs), 
        outputs=OutputSpec(outputs),
        map_func=_clean_atoms_attributes,
        args=[keep_info_keys, keep_array_keys]
    )
