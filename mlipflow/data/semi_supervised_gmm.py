from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import torch
from mace.calculators import MACECalculator
from wfl.configset import ConfigSet

from mlipflow.data import setup_logging
from mlipflow.data.gmm import (
    compute_descriptors,
    evaluate_structure_cluster_probability,
    extract_species_labels,
    project_with_pca,
    torch_pca_dynamic,
    train_gmm,
)

if TYPE_CHECKING:
    import numpy as np
    from ase import Atoms

    from mlipflow.strategies.mlip import MACEModel

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
    atom_indices: Optional slice, list, or array to filter atoms
                  (e.g., slice(48, None) for atoms[48:]).
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
        train_file: str | list[str],
        pool_file: str | list[str],
        mlip_strategy: MACEModel,
        atom_indices: slice | list[int] | np.ndarray | None = None,
        device: str = "cpu",
        high_certainty: float = 0.99,
        final_certainty: float = 0.95,
        pca_threshold: float = 0.98,
        gmm_iters: int = 100,
    ):
        self.train_configs = list(ConfigSet(train_file))
        self.pool_configs = list(ConfigSet(pool_file))
        self.mlip_strategy = mlip_strategy
        self.calc = MACECalculator(
            model_paths=self.mlip_strategy.model_file, device=device
        )
        self.atom_indices = atom_indices
        self.device = device
        self.high_certainty = high_certainty
        self.final_certainty = final_certainty
        self.pca_threshold = pca_threshold
        self.gmm_iters = gmm_iters
        self.dtype = torch.float32

    def _step1_initial_fit(self):
        """
        Step 1 - Initial Fit
        Compute MACE descriptors for the labelled configset -> PCA -> fit TorchGMM
        where K = number of unique 'species' labels in the configset (no BIC scan).
        """
        logger.info("=== Step 1: Initial Fit ===")
        species_labels = extract_species_labels(self.train_configs)
        self.K = len(species_labels)
        self.species_labels = species_labels

        self.X_train_padded, X_train_flat = compute_descriptors(
            self.train_configs, self.calc, self.device, self.dtype, self.atom_indices
        )
        X_train_reduced, self.pca_V_0, self.pca_mean_0, _ = torch_pca_dynamic(
            X_train_flat, threshold=self.pca_threshold
        )
        self.gmm_0 = train_gmm(
            X_train_reduced,
            k=self.K,
            n_init=5,
            n_iters=self.gmm_iters,
            device=self.device,
        )

    def _step2_first_pass_labelling(self):
        """
        Step 2 - First-Pass Labelling
        Project pool descriptors into the *initial* PCA space and predict component labels.
        """
        logger.info("=== Step 2: First-Pass Labelling ===")
        self.X_pool_padded, _ = compute_descriptors(
            self.pool_configs, self.calc, self.device, self.dtype, self.atom_indices
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

    def _step3_high_certainty_filter(self) -> list[int]:
        """
        Step 3 - High-Certainty Filter
        Retain only pool configs where the structure-level max posterior > high_certainty.
        Structure certainty = min over atoms of max_k P(k | atom).
        """
        logger.info(
            f"=== Step 3: High-Certainty Filter (threshold={self.high_certainty}) ==="
        )
        mask_high = self.pool_certainty_0 > self.high_certainty
        indices = torch.where(mask_high)[0].tolist()
        logger.info(
            f"Retained {len(indices)} / {len(self.pool_configs)} pool configs "
            f"with certainty > {self.high_certainty}"
        )
        return indices

    def _step4_refit(self, high_certainty_idx: list[int]):
        """
        Step 4 - Refit
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
            hc_padded = self.X_pool_padded[high_certainty_idx]  # (n_hc, A_max, D)
            mask_hc = torch.any(hc_padded != 0, dim=-1)
            X_hc_flat = hc_padded[mask_hc]
            X_combined_flat = torch.cat([X_train_flat, X_hc_flat], dim=0)
            logger.info(
                f"Combined: {X_train_flat.shape[0]} train atoms + "
                f"{X_hc_flat.shape[0]} high-certainty pool atoms "
                f"= {X_combined_flat.shape[0]} total atoms"
            )
        else:
            logger.warning(
                "No high-certainty pool configs found; refitting on train set only."
            )
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
            device=self.device,
        )

    def _map_components_to_species(self) -> dict:
        """
        Maps GMM components to species labels by evaluating training configurations.

        Returns:
            dict: Mapping from component index (int) to species label (str).
        """
        logger.info("Mapping GMM components to species labels...")

        # Project training configs into final PCA space
        X_train_proj, mask_train = project_with_pca(
            self.X_train_padded, self.pca_V_1, self.pca_mean_1
        )

        # Get log posteriors for all training atoms
        valid_train_proj = X_train_proj[mask_train]
        log_post = self.gmm_1.posterior_log_probs(valid_train_proj)
        preds = torch.argmax(log_post, dim=1)

        # Get true species labels for each training atom
        # (Assuming all atoms in a structure have the same species label as per extract_species_labels)
        true_labels = []
        for _i, config in enumerate(self.train_configs):
            species = config.info["species"]
            num_atoms = len(config)
            true_labels.extend([species] * num_atoms)

        # Count occurences of (species, comp) pairs
        counts = Counter(zip(true_labels, preds.tolist()))

        # For each species, find the dominant component
        species_to_comp = {}
        for species in self.species_labels:
            max_count = -1
            best_comp = -1
            for comp in range(self.K):
                count = counts.get((species, comp), 0)
                if count > max_count:
                    max_count = count
                    best_comp = comp
            species_to_comp[species] = best_comp

        # Reverse mapping: component -> species
        comp_to_species = {v: k for k, v in species_to_comp.items()}

        # Handle any components that weren't the "best" for any species
        # (Though with K fixed to species count, it should be 1-to-1 ideally)
        for comp in range(self.K):
            if comp not in comp_to_species:
                # Fallback to the species that has this component most frequently
                max_count = -1
                best_species = "unknown"
                for species in self.species_labels:
                    count = counts.get((species, comp), 0)
                    if count > max_count:
                        max_count = count
                        best_species = species
                comp_to_species[comp] = best_species

        logger.info(f"Component and species map: {comp_to_species}")
        return comp_to_species

    def _step5_relabel(self) -> tuple[list[Atoms], list[Atoms], torch.Tensor]:
        """
        Step 5 - Relabel
        Project all pool configs into the new PCA space, re-predict labels,
        and split into certain (relabeled) and uncertain (labelled as "unknown") configs.

        Returns:
            Tuple of (certain_configs, uncertain_configs, final_certainty_scores).
        """
        logger.info(f"=== Step 5: Relabel (threshold={self.final_certainty}) ===")

        # Get component -> species mapping
        comp_to_species = self._map_components_to_species()

        # Project and evaluate certainty
        X_pool_proj_1, mask_pool_1 = project_with_pca(
            self.X_pool_padded, self.pca_V_1, self.pca_mean_1
        )
        pool_certainty_1 = evaluate_structure_cluster_probability(
            self.gmm_1, self.X_pool_padded, self.pca_V_1, self.pca_mean_1
        )

        # Predict components for all pool atoms
        valid_pool_proj = X_pool_proj_1[mask_pool_1]
        log_post_pool = self.gmm_1.posterior_log_probs(valid_pool_proj)
        atom_preds = torch.argmax(log_post_pool, dim=1)

        # Reconstruct structure-level predictions (majority vote over atoms)
        num_atoms_list = mask_pool_1.sum(dim=1).int().tolist()
        split_preds = torch.split(atom_preds, num_atoms_list)
        struct_preds = [
            torch.mode(chunk).values.item() if len(chunk) > 0 else -1
            for chunk in split_preds
        ]

        logger.info(
            f"Pool certainty (final): "
            f"min={pool_certainty_1.min().item():.3f}  "
            f"mean={pool_certainty_1.mean().item():.3f}  "
            f"max={pool_certainty_1.max().item():.3f}"
        )

        certain_configs = []
        uncertain_configs = []

        for config, comp, certainty in zip(
            self.pool_configs, struct_preds, pool_certainty_1.tolist()
        ):
            # Attach certainty score
            config.info["gmm_certainty"] = certainty

            if certainty >= self.final_certainty:
                config.info["species"] = comp_to_species.get(comp, "unknown")
                certain_configs.append(config)
            else:
                config.info["species"] = "unknown"
                uncertain_configs.append(config)

        logger.info(
            f"Relabeling complete. "
            f"Retained {len(certain_configs)} certain configs, "
            f"{len(uncertain_configs)} uncertain configs (labelled 'unknown')."
        )

        # Log distribution of assigned labels
        label_counts = Counter(
            config.info.get("species", "missing") for config in self.pool_configs
        )
        logger.info(f"Assigned species distribution: {dict(label_counts)}")

        return certain_configs, uncertain_configs, pool_certainty_1

    def run(self) -> tuple[list[Atoms], list[Atoms], torch.Tensor]:
        """
        Execute the full semi-supervised refinement pipeline.

        Returns:
            Tuple ``(certain_configs, uncertain_configs, certainty_scores)`` where:
            - ``certain_configs`` is the list of high-certainty pool Atoms objects,
              re-labelled with GMM-predicted species.
            - ``uncertain_configs`` is the list of low-certainty pool Atoms objects,
              labelled with ``species='unknown'``.
            - ``certainty_scores`` is a 1-D tensor of final certainty values for all pool configs.
        """
        self._step1_initial_fit()
        self._step2_first_pass_labelling()
        high_certainty_idx = self._step3_high_certainty_filter()
        self._step4_refit(high_certainty_idx)
        certain_configs, uncertain_configs, certainty_scores = self._step5_relabel()

        logger.info(
            f"Pipeline complete. {len(certain_configs)} certain and "
            f"{len(uncertain_configs)} uncertain configurations."
        )
        return certain_configs, uncertain_configs, certainty_scores
