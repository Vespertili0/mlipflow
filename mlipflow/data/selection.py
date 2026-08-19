from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase.io import write
from mace.calculators import MACECalculator
from wfl.configset import ConfigSet, OutputSpec
from wfl.utils.misc import atoms_to_list

from mlipflow.data import setup_logging
from mlipflow.data.gmm import (
    compute_descriptors,
    evaluate_pool_uncertainty,
    get_certainty_threshold,
    select_uncertain_structures,
    torch_pca_dynamic,
    train_gmm,
)
from mlipflow.data.processing import check_maxforce_and_cleanarrays

if TYPE_CHECKING:
    from mlipflow.strategies.mlip import MACEModel

setup_logging()
logger = logging.getLogger(__name__)


def split_configset_by_force_agreement(
    in_file,
    out_file,
    pair_tuple,
    main_suffix="train",
    side_suffix="test",
    clean_data: bool = True,
    max_force: float = 12.0,
    mlip_prefix: str = "MACE",
    calc: str = "opt",
) -> None:
    """
    Split a ConfigSet into two parts based on force agreement between two force keys.

    Logic:
    - Top 20% (highest MAE) always go into the main_chunk.
    - From the bottom 80%, randomly select another 60% of the total dataset.
    - Remaining ~20% go into the side_chunk.

    Parameters
    ----------
    in_file : str | list[Atoms] | ConfigSet
        Input file or atomic configurations.
    out_file : str | dict
        Base name for output files or dict of output file paths.
    pair_tuple : tuple[str, str]
        Names of the two force arrays to compare, e.g. ('DFT', 'MACE').
    main_suffix : str, default='train'
        Suffix for the main split output file.
    side_suffix : str, default='test'
        Suffix for the side split output file.
    clean_data : bool, default=True
        If True, runs check_maxforce_and_cleanarrays prior to force MAE computation.
    max_force : float, default=12.0
        Maximum force threshold for filtering configurations when clean_data is True.
    mlip_prefix : str, default='MACE'
        MLIP prefix used for cleaning arrays when clean_data is True.
    calc : str, default='opt'
        Calculation context used for key cleaning when clean_data is True.

    """
    assert len(pair_tuple) == 2, "need two force keys to compare"

    if clean_data:
        configs = check_maxforce_and_cleanarrays(
            in_file=in_file,
            out_file=None,
            mlip_prefix=mlip_prefix,
            calc=calc,
            max_force=max_force,
        )
    else:
        configs = atoms_to_list(ConfigSet(in_file))
    # get MAE for forces and rank
    force_mae = [
        np.mean(
            np.abs(
                at.arrays[f"{pair_tuple[0]}forces"]
                - at.arrays[f"{pair_tuple[1]}forces"]
            )
        )
        for at in configs
    ]
    n_total = len(force_mae)

    # find top 20% group using quantile
    cutoff = np.quantile(force_mae, 0.8)
    top_frames = np.where(force_mae >= cutoff)[0]

    # pull extra 60% (of total) from bottom 80% at random
    other_frames = np.where(force_mae < cutoff)[0]
    n_extra = min(int(n_total * 0.6), len(other_frames))
    selected_frames = np.random.default_rng().choice(
        other_frames, size=n_extra, replace=False
    )

    # combine and finalise frame_idx
    main_chunk = np.concatenate([top_frames, selected_frames])
    side_chunk = np.setdiff1d(np.arange(n_total), main_chunk)

    # pull atoms by indices
    main_at = [list(configs)[idx] for idx in main_chunk]
    side_at = [list(configs)[idx] for idx in side_chunk]

    for atoms, suffix in zip([main_at, side_at], [main_suffix, side_suffix]):
        if isinstance(out_file, dict):
            write(out_file[suffix], atoms)
        else:
            write(f"{suffix}_{out_file}", atoms)


def select_by_uncertainty(
    train_file: str,
    pool_file: str,
    out_file: str,
    mlip_strategy: MACEModel,
    certainty_threshold: float = 0.8,
    pca_variance_threshold: float = 0.95,
    max_gmm_components: int = 30,
    gmm_n_init: int = 5,
    device: str = "cpu",
    dtype: torch.dtype = None,
) -> None:
    """
    Select new structures from a pool based on GMM uncertainty.

    Logic:
    - Extracts atomic descriptors from training configurations.
    - Fits a PCA on training descriptors and projects them.
    - Fits a PyTorch GMM on the PCA-reduced training descriptors.
    - Evaluates uncertainty on pool configurations using structure-wise
      min-pooling of atomic log-likelihoods.
    - Selects configurations that have a structure uncertainty score higher
      than the threshold defined by the `certainty_threshold` percentile on the
      training data.

    Parameters
    ----------
    train_file : str
        Input file containing training atomic configurations.
    pool_file : str
        Input file containing pool atomic configurations.
    out_file : str
        Output file to save the selected configurations.
    mlip_strategy : MACEModel
        The `MACEModel` defining the strategy and paths for calculation.
    certainty_threshold : float, default=0.8
        Percentile of the training data uncertainty distribution used as the cutoff
        for certainty. Structures with uncertainty above this value are selected.
    pca_variance_threshold : float, default=0.95
        Variance ratio to be captured by the dynamic PCA projection.
    max_gmm_components : int, default=30
        Maximum number of GMM components to search for.
    gmm_n_init : int, default=5
        Number of initialisations per GMM to avoid local minima.
    device : str, default='cpu'
        Device for PyTorch computations ('cuda' or 'cpu').
    dtype : torch.dtype, optional
        PyTorch dtype, defaults to torch.float32.
    """
    if dtype is None:
        dtype = torch.float32

    mlip_calc = MACECalculator(mlip_strategy.model_file)

    train_configs = atoms_to_list(ConfigSet(train_file))
    pool_configs = list(ConfigSet(pool_file))

    # Prepare descriptors
    train_padded, train_flat = compute_descriptors(
        train_configs, mlip_calc, device, dtype
    )
    pool_padded, _ = compute_descriptors(pool_configs, mlip_calc, device, dtype)

    # Run PCA on training data
    X_train_reduced, pca_V, pca_mean, _n_pca_components = torch_pca_dynamic(
        X=train_flat, threshold=pca_variance_threshold
    )

    # Fit GMM & evaluate uncertainty cutoff
    best_gmm = train_gmm(
        X_reduced=X_train_reduced,
        k=None,
        max_k=max_gmm_components,
        n_init=gmm_n_init,
        device=device,
    )
    train_uncertainty = evaluate_pool_uncertainty(
        best_gmm, pca_V, pca_mean, train_padded
    )
    gmm_threshold = get_certainty_threshold(
        train_uncertainty, certainty_percentile=certainty_threshold
    )

    # Identify pool members with uncertainty below threshold (higher score = more uncertain)
    pool_uncertainty = evaluate_pool_uncertainty(best_gmm, pca_V, pca_mean, pool_padded)
    uncertain_frames_tensor = select_uncertain_structures(
        pool_uncertainty, gmm_threshold
    )

    # Isolate uncertain frames
    selected_configs = [
        pool_configs[frame_idx] for frame_idx in uncertain_frames_tensor
    ]
    logger.info(
        f"Selected {len(selected_configs)} configurations based on uncertainty threshold {gmm_threshold}"
    )

    OutputSpec(out_file).write(ConfigSet(selected_configs))

    return
