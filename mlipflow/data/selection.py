import numpy as np
from ase.io import write
from wfl.configset import ConfigSet
from wfl.utils.misc import atoms_to_list

def split_configset_by_force_agreement(in_file, out_file, pair_tuple, main_suffix='train', side_suffix='test') -> None:
    """
    Split a ConfigSet into two parts based on force agreement between two force keys.
    
    Logic:
    - Top 20% (highest MAE) always go into the main_chunk.
    - From the bottom 80%, randomly select another 60% of the total dataset.
    - Remaining ~20% go into the side_chunk.
    
    Parameters
    ----------
    in_file : str
        Input file containing atomic configurations.
    out_file : str
        Base name for output files.
    pair_tuple : tuple[str, str]
        Names of the two force arrays to compare, e.g. ('DFT', 'MACE').
    main_suffix : str, default='train'
        Suffix for the main split output file.
    side_suffix : str, default='test'
        Suffix for the side split output file.
    
    """
    assert len(pair_tuple) == 2, 'need two force keys to compare'

    configs = atoms_to_list(ConfigSet(in_file))
    # get MAE for forces and rank
    force_mae = []
    for at in configs:
        force_mae.append(
            np.mean(np.abs(at.arrays[f'{pair_tuple[0]}forces'] - at.arrays[f'{pair_tuple[1]}forces'])))
    n_total = len(force_mae)

    # find top 20% group using quantile
    cutoff = np.quantile(force_mae, 0.8)
    top_frames = np.where(force_mae >= cutoff)[0]

    # pull extra 60% (of total) from bottom 80% at random
    other_frames = np.where(force_mae < cutoff)[0]
    n_extra = min(int(n_total * 0.6), len(other_frames))
    selected_frames = np.random.choice(
        other_frames,
        size=n_extra,
        replace=False
    )

    # combine and finalise frame_idx
    main_chunk = np.concatenate([top_frames, selected_frames])
    side_chunk = np.setdiff1d(np.arange(n_total), main_chunk)

    # pull atoms by indices
    main_at = [list(configs)[idx] for idx in main_chunk]
    side_at = [list(configs)[idx] for idx in side_chunk]
    
    for atoms, suffix in zip([main_at, side_at], [main_suffix, side_suffix]):
        write(f'{suffix}_{out_file}', atoms)
