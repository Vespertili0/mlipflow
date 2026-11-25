from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

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
    from ase import Atoms

    from mlipflow.strategies.mlip import MACEModel

setup_logging()
logger = logging.getLogger(__name__)

_VALID_SPLIT_MODES = ("stratified_random", "random")


def _random_split_indices(
    n_total: int, test_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint ``(train_idx, test_idx)`` arrays via unbiased random permutation."""
    if n_total == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty
    if n_total == 1:
        return np.array([0], dtype=int), np.empty(0, dtype=int)

    n_test = max(1, min(n_total - 1, round(n_total * test_fraction)))
    perm = rng.permutation(n_total)
    return perm[n_test:], perm[:n_test]


def _stratified_random_split_indices(
    configs: list[Atoms],
    test_fraction: float,
    rng: np.random.Generator,
    stratify_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint ``(train_idx, test_idx)`` arrays stratified by ``stratify_key``."""
    n_total = len(configs)
    if n_total <= 1:
        return _random_split_indices(n_total, test_fraction, rng)

    groups: dict[str | None, list[int]] = {}
    for idx, at in enumerate(configs):
        raw_val = at.info.get(stratify_key)
        key = None if raw_val is None else str(raw_val)
        groups.setdefault(key, []).append(idx)

    # Fall back to unstratified random split when stratification is trivial
    if len(groups) <= 1 or all(len(indices) <= 1 for indices in groups.values()):
        logger.debug(
            "Stratification key '%s' has <= 1 multi-frame group; falling back to random split.",
            stratify_key,
        )
        return _random_split_indices(n_total, test_fraction, rng)

    sorted_keys = sorted(
        groups.keys(), key=lambda k: (k is None, "" if k is None else k)
    )
    target_test = max(1, min(n_total - 1, round(n_total * test_fraction)))

    alloc: dict[str | None, int] = {}
    remainders: list[tuple[float, str | None]] = []
    for k in sorted_keys:
        n_g = len(groups[k])
        if n_g <= 1:
            alloc[k] = 0
            continue
        exact = n_g * test_fraction
        base = min(int(np.floor(exact)), n_g - 1)
        alloc[k] = base
        if base < n_g - 1:
            remainders.append((exact - base, k))

    deficit = target_test - sum(alloc.values())
    if deficit > 0:
        remainders.sort(key=lambda item: -item[0])
        for _, k in remainders:
            if deficit <= 0:
                break
            if alloc[k] < len(groups[k]) - 1:
                alloc[k] += 1
                deficit -= 1
    elif deficit < 0:
        for k in sorted_keys:
            if deficit >= 0:
                break
            if alloc[k] > 0:
                alloc[k] -= 1
                deficit += 1

    train_idx: list[int] = []
    test_idx: list[int] = []
    for k in sorted_keys:
        perm = rng.permutation(groups[k])
        n_g_test = alloc[k]
        test_idx.extend(perm[:n_g_test].tolist())
        train_idx.extend(perm[n_g_test:].tolist())

    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def split_configset(
    in_file: str | list[Atoms] | ConfigSet,
    out_file: str | dict[str, str],
    main_suffix: str = "train",
    side_suffix: str = "test",
    split_mode: Literal["stratified_random", "random"] = "stratified_random",
    test_fraction: float = 0.2,
    seed: int | None = 42,
    stratify_key: str = "species",
    clean_data: bool = True,
    max_force: float = 12.0,
    mlip_prefix: str = "MACE",
    calc: str = "opt",
) -> None:
    """Split a ConfigSet into unbiased training and held-out test sets.

    Supports species-stratified random splitting (default) and unstratified
    random splitting with a deterministic random seed, preserving the full
    force-error distribution in the held-out test set for unbiased accuracy
    evaluation and split-conformal UQ calibration.

    Parameters
    ----------
    in_file : str or list of Atoms or ConfigSet
        Input file or atomic configurations.
    out_file : str or dict of str to str
        Base filename for output files (written as ``f"{suffix}_{out_file}"``)
        or a mapping ``{main_suffix: train_path, side_suffix: test_path}``.
    main_suffix : str, default="train"
        Key/prefix for the training split output file.
    side_suffix : str, default="test"
        Key/prefix for the held-out test split output file.
    split_mode : {"stratified_random", "random"}, default="stratified_random"
        Splitting strategy. ``"stratified_random"`` stratifies by ``stratify_key``
        and falls back cleanly to ``"random"`` if the key is absent.
    test_fraction : float, default=0.2
        Fraction of configurations assigned to the held-out test split; must lie
        strictly in ``(0.0, 1.0)``.
    seed : int or None, default=42
        Random seed for deterministic splitting.
    stratify_key : str, default="species"
        ``atoms.info`` key used for stratification when ``split_mode="stratified_random"``.
    clean_data : bool, default=True
        If True, runs ``check_maxforce_and_cleanarrays`` prior to splitting.
    max_force : float, default=12.0
        Maximum force threshold for filtering configurations when ``clean_data`` is True.
    mlip_prefix : str, default="MACE"
        MLIP prefix used for cleaning arrays when ``clean_data`` is True.
    calc : str, default="opt"
        Calculation context used for key cleaning when ``clean_data`` is True.

    Raises
    ------
    ValueError
        If ``split_mode`` is not in ``("stratified_random", "random")`` or
        ``test_fraction`` is not strictly between 0 and 1.
    """
    if split_mode not in _VALID_SPLIT_MODES:
        raise ValueError(
            f"Invalid split_mode={split_mode!r}. Expected one of {_VALID_SPLIT_MODES}."
        )
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(
            f"test_fraction must be strictly between 0 and 1, got {test_fraction!r}."
        )

    if clean_data:
        configs = check_maxforce_and_cleanarrays(
            in_file=in_file,
            out_file=None,
            mlip_prefix=mlip_prefix,
            calc=calc,
            max_force=max_force,
        )
    else:
        configs = list(ConfigSet(in_file))

    rng = np.random.default_rng(seed)
    if split_mode == "stratified_random":
        main_chunk, side_chunk = _stratified_random_split_indices(
            configs=configs,
            test_fraction=test_fraction,
            rng=rng,
            stratify_key=stratify_key,
        )
    else:
        main_chunk, side_chunk = _random_split_indices(
            n_total=len(configs), test_fraction=test_fraction, rng=rng
        )

    main_at = [configs[idx] for idx in main_chunk]
    side_at = [configs[idx] for idx in side_chunk]

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
