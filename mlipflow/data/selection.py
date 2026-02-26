import numpy as np
from ase.io import write
from wfl.configset import ConfigSet, OutputSpec
from wfl.utils.misc import atoms_to_list
from mlipflow.strategies.mlip import MACEModel

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


def select_by_uncertainty(
    train_file: str, 
    pool_file: str, 
    out_file: str, 
    mlip_strategy: MACEModel,
    certainty_threshold: float = 0.8, 
    pca_threshold: int = 10,
    max_gmm_components: int = 30,
    gmm_n_init: int = 5,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> None:
    """
    Select new structures from a pool based on GMM uncertainty.
    
    Logic:
    - Extracts atomic descriptors from training configurations.
    - Fits a PCA on training descriptors and projects them.
    - Fits a PyTorch GMM on the PCA-reduced training descriptors.
    - Evaluates uncertainty on pool configurations.
    - Selects top `n_select` structures with the highest uncertainty.

    Parameters
    ----------
    train_file : str
        Input file containing training atomic configurations.
    pool_file : str
        Input file containing pool atomic configurations.
    out_file : str
        Output file to save the selected configurations.
    n_select : int
        Number of configurations to select.
    desc_key : str, default='SOAP'
        Key for atomic descriptors in atoms.arrays.
    n_pca_components : int, default=10
        Number of PCA components to retain.
    max_gmm_components : int, default=30
        Maximum number of GMM components to search for.
    gmm_n_init : int, default=5
        Number of initialisations per GMM to avoid local minima.
    device : str, default='cuda'
        Device for PyTorch computations ('cuda' or 'cpu').
    """
    import torch
    from mlipflow.data.gmm import (
        find_best_gmm, evaluate_pool_uncertainty, torch_pca_dynamic,
        get_certainty_threshold, select_uncertain_structures
    )
    from mace.calculators import MACECalculator

    mlip_calc = MACECalculator(mlip_strategy.model_file)

    train_configs = atoms_to_list(ConfigSet(train_file))
    pool_configs = atoms_to_list(ConfigSet(pool_file))
    
    # Prepare descriptors
    train_descr = np.array(
        [mlip_calc.get_descriptors(at) for at in train_configs]
    )
    X_train = torch.from_numpy(train_descr).to(pt_device).float()
    pool_descr = np.array(
        [mlip_calc.get_descriptors(at) for at in pool_configs]
    )
    X_pool = torch.from_numpy(pool_descr).to(pt_device).float()
    
    # Run PCA on training data
    X_train_reduced, pca_V, pca_mean, n_paca_components = torch_pca_dynamic(
        X=X_train.reshape(-1, X_train.shape[-1]), 
        threshold=pca_threshold
    )
    
    # Fit GMM & evalute uncertainty cutoff
    best_gmm = find_best_gmm(
        X_reduced=X_train_reduced, 
        max_k=max_gmm_components, 
        n_init=gmm_n_init, 
        device=device
    )
    train_atom_ll = evaluate_pool_uncertainty(best_gmm, pca_V, pca_mean, X_train)
    gmm_threshold = get_certainty_threshold(
        train_atom_ll, 
        certainty_percentile=certainty_threshold
    )
    
    # Identify pool members with uncertainty below threshold
    pool_atom_ll = evaluate_pool_uncertainty(best_gmm, pca_V, pca_mean, X_pool)
    uncertain_frames_tensor = select_uncertain_structures(pool_atom_ll, gmm_threshold)

    # Isolate uncertain frames
    selected_configs = [pool_configs[frame_idx] for frame_idx in uncertain_frames_tensor]   
    
    OutputSpec(out_file).write(ConfigSet(selected_configs))

    return None
