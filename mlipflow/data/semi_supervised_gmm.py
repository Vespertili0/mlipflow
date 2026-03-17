import logging
from typing import List, Tuple, Union

import torch
from ase import Atoms
from wfl.configset import ConfigSet
from wfl.utils.misc import atoms_to_list
from mace.calculators import MACECalculator

from mlipflow.strategies.mlip import MACEModel
from mlipflow.data import setup_logging
from mlipflow.data.gmm import (
    TorchGMM, 
    torch_pca_dynamic,
    extract_species_labels,
    compute_descriptors,
    project_with_pca,
    evaluate_structure_cluster_probability,
    train_gmm
)

setup_logging()
logger = logging.getLogger(__name__)

class GMMLabelChecker:
    """
    Iterative semi-supervised labelling pipeline using MACE descriptors and a
    fixed-K Gaussian Mixture Model.

    Pipeline Overview
    -----------------
    1. Initial Fit: Fit PCA and TorchGMM on the labelled configset (K = number of unique species).
    2. First-Pass Labelling: Score pool configurations in the initial PCA space.
    3. High-Certainty Filter: Retain pool configs with structure-level certainty > high_certainty.
    4. Refit: Refit PCA and GMM on the combined dataset (original + high-certainty pool).
    5. Final Validation: Re-score all pool configs and retain those above final_certainty.

    Design Notes
    ------------
    - K is fixed to the number of unique `atoms.info['species']` labels in the train set.
    - BIC optimisation is omitted to preserve the component ↔ species mapping.
    - The PCA is fully refit in Step 4 because labels are re-predicted from scratch.
    - Certainty is calculated using log-sum-exp-stable posteriors via evaluate_structure_cluster_probability.

    Parameters
    ----------
    train_file : str or list of str
        Filepath(s) to labelled reference configurations; must carry ``atoms.info['species']``.
    pool_file : str or list of str
        Filepath(s) to unlabelled configurations to be scored and labelled.
    mlip_strategy : MACEModel
        The `MACEModel` defining the strategy and paths for calculation.
    device : str
        Torch device ('cpu' or 'cuda').
    high_certainty : float
        Min certainty threshold for Step 3 filter (default 0.9).
    final_certainty : float
        Min certainty threshold for Step 5 exclusion (default 0.8).
    pca_threshold : float
        Cumulative variance captured by PCA (default 0.95).
    gmm_iters : int
        EM iterations per GMM fit (default 100).
    """

    def __init__(
        self,
        train_file: Union[str, List[str]],
        pool_file: Union[str, List[str]],
        mlip_strategy: MACEModel,
        device: str = 'cpu',
        high_certainty: float = 0.95,
        final_certainty: float = 0.90,
        pca_threshold: float = 0.95,
        gmm_iters: int = 100,
    ):
        self.train_configs = atoms_to_list(ConfigSet(train_file))
        self.pool_configs = atoms_to_list(ConfigSet(pool_file))
        self.mlip_strategy = mlip_strategy
        self.calc = MACECalculator(model_paths=self.mlip_strategy.model_file, device=device)
        self.device = device
        self.high_certainty = high_certainty
        self.final_certainty = final_certainty
        self.pca_threshold = pca_threshold
        self.gmm_iters = gmm_iters
        self.dtype = torch.float32


    def _step1_initial_fit(self):
        """
        Step 1 – Initial Fit
        Compute MACE descriptors for the labelled configset -> PCA -> fit TorchGMM
        where K = number of unique 'species' labels in the configset (no BIC scan).
        """
        logger.info("=== Step 1: Initial Fit ===")
        species_labels = extract_species_labels(self.train_configs)
        self.K = len(species_labels)
        self.species_labels = species_labels

        self.X_train_padded, X_train_flat = compute_descriptors(
            self.train_configs, self.calc, self.device, self.dtype
        )
        X_train_reduced, self.pca_V_0, self.pca_mean_0, _ = torch_pca_dynamic(
            X_train_flat, threshold=self.pca_threshold
        )
        self.gmm_0 = train_gmm(
            X_train_reduced, 
            k=self.K, 
            n_init=5, 
            n_iters=self.gmm_iters, 
            device=self.device
        )


    def _step2_first_pass_labelling(self):
        """
        Step 2 – First-Pass Labelling
        Project pool descriptors into the *initial* PCA space and predict component labels.
        """
        logger.info("=== Step 2: First-Pass Labelling ===")
        self.X_pool_padded, _ = compute_descriptors(
            self.pool_configs, self.calc, self.device, self.dtype
        )
        self.X_pool_proj_0, self.mask_pool_0 = project_with_pca(
            self.X_pool_padded, self.pca_V_0, self.pca_mean_0
        )
        self.pool_certainty_0 = evaluate_structure_cluster_probability(
            self.gmm_0, self.X_pool_padded, self.pca_V_0, self.pca_mean_0
        )
        logger.info(
            f"Pool certainty (pass 0): "
            f"min={self.pool_certainty_0.min().item():.3f}  "
            f"mean={self.pool_certainty_0.mean().item():.3f}  "
            f"max={self.pool_certainty_0.max().item():.3f}"
        )


    def _step3_high_certainty_filter(self) -> List[int]:
        """
        Step 3 – High-Certainty Filter
        Retain only pool configs where the structure-level max posterior > high_certainty.
        Structure certainty = min over atoms of max_k P(k | atom).
        """
        logger.info(f"=== Step 3: High-Certainty Filter (threshold={self.high_certainty}) ===")
        mask_high = self.pool_certainty_0 > self.high_certainty
        indices = torch.where(mask_high)[0].tolist()
        logger.info(
            f"Retained {len(indices)} / {len(self.pool_configs)} pool configs "
            f"with certainty > {self.high_certainty}"
        )
        return indices


    def _step4_refit(self, high_certainty_idx: List[int]):
        """
        Step 4 – Refit
        Combine original labelled descriptors with high-certainty pool descriptors.
        Refit PCA on the combined set (new latent space captures enriched variance).
        Refit TorchGMM(K) on the new PCA-reduced features.
        """
        logger.info("=== Step 4: Refit (combined PCA + GMM) ===")

        # Flatten train real atoms
        mask_train = torch.any(self.X_train_padded != 0, dim=-1)
        X_train_flat = self.X_train_padded[mask_train]

        if high_certainty_idx:
            # Gather high-certainty pool atoms
            hc_padded = self.X_pool_padded[high_certainty_idx]       # (n_hc, A_max, D)
            mask_hc = torch.any(hc_padded != 0, dim=-1)
            X_hc_flat = hc_padded[mask_hc]
            X_combined_flat = torch.cat([X_train_flat, X_hc_flat], dim=0)
            logger.info(
                f"Combined: {X_train_flat.shape[0]} train atoms + "
                f"{X_hc_flat.shape[0]} high-certainty pool atoms "
                f"= {X_combined_flat.shape[0]} total atoms"
            )
        else:
            logger.warning("No high-certainty pool configs found; refitting on train set only.")
            X_combined_flat = X_train_flat

        # Full PCA refit on combined set
        X_combined_reduced, self.pca_V_1, self.pca_mean_1, _ = torch_pca_dynamic(
            X_combined_flat, threshold=self.pca_threshold
        )
        self.gmm_1 = train_gmm(
            X_combined_reduced, 
            k=self.K, 
            n_init=5, 
            n_iters=self.gmm_iters, 
            device=self.device
        )


    def _step5_final_validation(self) -> Tuple[List[Atoms], torch.Tensor]:
        """
        Step 5 – Final Validation
        Re-project all pool configs into the new PCA space and re-predict labels
        with the refitted GMM. Exclude any config where certainty < final_certainty.

        Returns:
            Tuple of (selected_configs, final_certainty_scores).
        """
        logger.info(f"=== Step 5: Final Validation (threshold={self.final_certainty}) ===")

        X_pool_proj_1, mask_pool_1 = project_with_pca(
            self.X_pool_padded, self.pca_V_1, self.pca_mean_1
        )
        pool_certainty_1 = evaluate_structure_cluster_probability(
            self.gmm_1, self.X_pool_padded, self.pca_V_1, self.pca_mean_1
        )
        logger.info(
            f"Pool certainty (final): "
            f"min={pool_certainty_1.min().item():.3f}  "
            f"mean={pool_certainty_1.mean().item():.3f}  "
            f"max={pool_certainty_1.max().item():.3f}"
        )

        keep_mask = pool_certainty_1 >= self.final_certainty
        keep_idx = torch.where(keep_mask)[0].tolist()
        excluded = len(self.pool_configs) - len(keep_idx)
        logger.info(
            f"Excluded {excluded} configs below final certainty {self.final_certainty}. "
            f"Retaining {len(keep_idx)} configs."
        )

        selected_configs = [self.pool_configs[i] for i in keep_idx]
        # Attach certainty score to each selected config's info dict
        for config, idx in zip(selected_configs, keep_idx):
            config.info['gmm_certainty'] = pool_certainty_1[idx].item()

        return selected_configs, pool_certainty_1[keep_idx]


    def run(self) -> Tuple[List[Atoms], torch.Tensor]:
        """
        Execute the full semi-supervised refinement pipeline.

        Returns:
            Tuple ``(final_configs, certainty_scores)`` where:
            - ``final_configs`` is the list of selected pool Atoms objects, each
              annotated with ``atoms.info['gmm_certainty']``.
            - ``certainty_scores`` is a 1-D tensor of corresponding certainty values.
        """
        self._step1_initial_fit()
        self._step2_first_pass_labelling()
        high_certainty_idx = self._step3_high_certainty_filter()
        self._step4_refit(high_certainty_idx)
        final_configs, certainty_scores = self._step5_final_validation()
        logger.info(
            f"Pipeline complete. {len(final_configs)} configurations selected."
        )
        return final_configs, certainty_scores
