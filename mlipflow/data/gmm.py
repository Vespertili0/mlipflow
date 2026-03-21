from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from mlipflow.data import setup_logging
from mlipflow.utils import find_robust_elbow

if TYPE_CHECKING:
    import ase
    import mace.calculators
    import numpy as np

setup_logging()
logger = logging.getLogger(__name__)


class TorchGMM:
    """
    Gaussian Mixture Model (GMM) implemented in PyTorch for GPU-accelerated fitting and scoring.
    Supports full covariance matrices.
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        device: str = "cuda",
        jitter: float = 1e-4,
        dtype: torch.dtype = torch.float32,
    ):
        self.K = n_components
        self.D = n_features
        self.device = device
        self.jitter = jitter  # for numerical stability of covariance matrices
        self.dtype = dtype

        # Initialise parameters with explicit dtype
        self.mu = torch.randn(self.K, self.D, device=device, dtype=self.dtype)
        self.sigma = torch.stack(
            [torch.eye(self.D, device=device, dtype=self.dtype) for _ in range(self.K)]
        )
        self.pi = torch.ones(self.K, device=device, dtype=self.dtype) / self.K

    def _component_log_pdf(self, x):
        x = x.to(self.dtype)
        # torch.eye creates a (D, D) identity matrix, multiplied by jitter;
        # adding to the diagonals of the covariance matrix, sigma, makes it robust
        sigma_reg = (
            self.sigma
            + torch.eye(self.D, device=self.device, dtype=self.dtype) * self.jitter
        )
        diff = x.unsqueeze(1) - self.mu.unsqueeze(
            0
        )  # Shape: (N, K, D), where N is the number of samples in x

        # Compute Cholesky decomposition for each component's covariance matrix, LxL^T = sigma_reg
        L = torch.linalg.cholesky(sigma_reg)  # Shape: (K, D, D)

        # FIX: Align batch dimensions perfectly for the solver
        # L becomes (1, K, D, D) and diff becomes (N, K, D, 1)
        # Resulting 'y' will be shape (N, K, D, 1)
        y = torch.linalg.solve_triangular(
            L.unsqueeze(0), diff.unsqueeze(-1), upper=False
        )

        # Calculate quadrance (Mahalanobis distance squared)
        quadrance = torch.sum(y**2, dim=-2).squeeze(-1)  # Shape: (N, K)

        # Calculate log determinant and normalisation constant
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)

        # Calculate log normalisation. `math.pi` avoids PyTorch scalar upcasting issues.
        pi_term = self.D * math.log(2 * math.pi)
        log_norm = -0.5 * (pi_term + log_det)

        # Return log probabilities. unsqueeze(0) broadcasts log_norm across N
        return -0.5 * quadrance + log_norm.unsqueeze(0)

    def fit(self, x: torch.Tensor, iters: int = 100):
        """
        Fits the GMM parameters (mu, sigma, pi) to the data using Expectation-Maximisation (EM).

        Args:
            x (torch.Tensor): Training data. Shape: (N_samples, N_features).
            iters (int): Number of EM iterations to run.
        """
        x = x.to(self.dtype)
        # Handle K=1 separately to avoid unnecessary loops
        if self.K == 1:
            self.mu = torch.mean(x, dim=0, keepdim=True)
            diff = x - self.mu
            self.sigma = (torch.mm(diff.t(), diff) / x.shape[0]).unsqueeze(0)
            self.pi = torch.tensor([1.0], device=self.device, dtype=self.dtype)
            return

        # Expectation-Maximisation Loop
        for _ in range(iters):
            # E-Step
            # Get the log likelihoods
            log_probs = self._component_log_pdf(x)
            # Add log(prior) -> equivalent to multiplying in normal space
            weighted_log_probs = log_probs + torch.log(self.pi)
            # The Log-Sum-Exp Trick for the denominator
            log_resp = weighted_log_probs - torch.logsumexp(
                weighted_log_probs, dim=1, keepdim=True
            )
            # Convert back to standard probability space [0, 1]
            resp = torch.exp(log_resp)

            # M-Step
            # Calculate effective number of points in each cluster
            N_k = resp.sum(dim=0) + 1e-6
            # Update Priors (pi)
            self.pi = N_k / x.shape[0]
            # Update Means (mu)
            self.mu = torch.mm(resp.t(), x) / N_k.unsqueeze(1)
            # Update Covariances (sigma)
            diff = x.unsqueeze(1) - self.mu.unsqueeze(0)
            self.sigma = torch.einsum("nki,nkj,nk->kij", diff, diff, resp) / N_k.view(
                -1, 1, 1
            )

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-likelihood of each sample in x under the full mixture model.

        Args:
            x (torch.Tensor): Input data. Shape: (N_samples, N_features).

        Returns:
            torch.Tensor: Log-likelihood of each sample. Shape: (N_samples,).
        """
        x = x.to(self.dtype)
        log_probs = self._component_log_pdf(x)

        return torch.logsumexp(log_probs + torch.log(self.pi), dim=1)

    def posterior_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes log posterior probabilities log P(k | x) for each sample and component,
        using the log-sum-exp trick for numerical stability.

        Args:
            x (torch.Tensor): Input data. Shape: (N_samples, N_features).

        Returns:
            torch.Tensor: Log posterior probabilities. Shape: (N_samples, K).
                          Each row can be exponentiated to obtain true posteriors that sum to 1.
        """
        x = x.to(self.dtype)
        log_probs = self._component_log_pdf(x)  # (N, K)
        weighted = log_probs + torch.log(self.pi)  # (N, K)
        # Subtract log-sum-exp over components → normalised log posteriors
        return weighted - torch.logsumexp(weighted, dim=1, keepdim=True)  # (N, K)

    def calculate_bic(self, x: torch.Tensor) -> float:
        """
        Calculates the Bayesian Information Criterion (BIC) for the current model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            float: The computed BIC value.
        """
        x = x.to(self.dtype)
        N = x.shape[0]
        # Number of free parameters for FULL covariance
        num_params = self.K * (self.D + self.D * (self.D + 1) / 2) + (self.K - 1)

        log_likelihood = self.score(x).sum()
        bic = num_params * math.log(N) - 2 * log_likelihood

        return bic.item()


def train_gmm(
    X_reduced: torch.Tensor,
    k: int | None = None,
    max_k: int = 30,
    n_init: int = 5,
    n_iters: int = 100,
    device: str = "cuda",
) -> TorchGMM:
    """
    Trains a Gaussian Mixture Model on the provided data.

    If `k` is provided, fits a GMM with exactly `k` components, running `n_init`
    restarts to avoid local optima and returning the model with the best log-likelihood.

    If `k` is None, performs a parameter sweep from K=1 to `max_k`, running `n_init`
    restarts per K, and uses the BIC elbow method to automatically select the optimal K.

    Args:
        X_reduced: PCA-reduced features. Shape: (N_samples, n_components).
        k: Fixed number of components. If None, sweeps up to max_k.
        max_k: Maximum K to evaluate if k is None.
        n_init: Number of random initialisations per K.
        n_iters: Maximum EM iterations per fit.
        device: Torch device string.

    Returns:
        The best fitted TorchGMM instance.
    """
    logger.info("Running GMM-Fitting ...")

    # Determine the range of K to test
    k_range = [k] if k is not None else range(1, max_k + 1)

    bic_history = []
    gmm_history = []

    for current_k in k_range:
        best_init_ll = float("-inf")
        best_init_gmm = None

        # Run EM multiple times for this K to avoid local optima
        for _attempt in range(n_init):
            gmm = TorchGMM(
                n_components=current_k, n_features=X_reduced.shape[1], device=device
            )
            gmm.fit(X_reduced, iters=n_iters)

            # Score this specific initialisation
            with torch.no_grad():
                ll = gmm.score(X_reduced).sum().item()

            # Keep the best initialisation for this K
            if ll > best_init_ll:
                best_init_ll = ll
                best_init_gmm = gmm

        # If we are doing a BIC sweep, calculate BIC and store
        if k is None:
            bic = best_init_gmm.calculate_bic(X_reduced)
            bic_history.append((current_k, bic))
            gmm_history.append(best_init_gmm)
            logger.info(f"  GMM-Fit: K={current_k} (Best of {n_init}) | BIC: {bic:.2f}")
        else:
            # We just wanted one specific K, return its best initialisation immediately
            logger.info(f"  GMM-Fit: Fixed K={current_k} (Best of {n_init} attempts)")
            return best_init_gmm

    # If we get here, k was None and we did a full sweep. Find the elbow.
    bics = [b for (k_val, b) in bic_history]

    # We use start_idx=0 since max_k is usually small, so we want to search the entire curve
    optimal_k_idx = find_robust_elbow(bics, start_idx=0) - 1

    # Retrieve the best GMM at the elbow point
    return gmm_history[optimal_k_idx]


def extract_species_labels(configs: list[ase.Atoms]) -> list[str]:
    """
    Reads the 'species' key from each config's info dict and returns a sorted
    list of unique labels.

    Args:
        configs: List of ASE Atoms objects, each carrying ``atoms.info['species']``.

    Returns:
        Sorted list of unique species label strings.

    Raises:
        ValueError: If any config is missing the 'species' info key.
    """
    labels = []
    for at in configs:
        sp = at.info.get("species")
        if sp is None:
            raise ValueError(
                f"Config is missing 'species' in atoms.info. "
                f"Formula: {at.get_chemical_formula()}"
            )
        labels.append(sp)
    unique = sorted(set(labels))
    logger.info(f"Found {len(unique)} species labels: {unique}")
    return unique


def compute_descriptors(
    configs: list[ase.Atoms],
    calc: mace.calculators.MACECalculator,
    device: str,
    dtype: torch.dtype,
    atom_indices: slice | list[int] | np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-atom MACE descriptors and returns them in two forms.

    Args:
        configs: List of ASE Atoms objects.
        calc:    Initialised MACECalculator.
        device:  Torch device string.
        dtype:   Torch dtype.
        atom_indices: Optional slice, list, or array to filter atoms
                      (e.g., slice(48, None) for atoms[48:]).
    """
    if not isinstance(configs, list):
        configs = list(configs)

    logger.info(f"Computing descriptors for {len(configs)} configurations...")
    # Default to a slice that includes everything if nothing is provided
    if atom_indices is None:
        atom_indices = slice(None)

    # Slice the numpy array BEFORE tensor conversion and device transfer
    raw = [
        torch.from_numpy(calc.get_descriptors(at)[atom_indices]).to(
            device=device, dtype=dtype
        )
        for at in configs
    ]

    A_max = max(t.shape[0] for t in raw)
    D = raw[0].shape[1]

    # 1. Build flat tensor directly
    X_flat = torch.cat(raw, dim=0)

    # 2. Build padded tensor
    logger.info(f"Building padded tensor with shape ({len(configs)}, {A_max}, {D})...")
    X_padded = torch.zeros(len(configs), A_max, D, device=device, dtype=dtype)
    for i, t in enumerate(raw):
        X_padded[i, : t.shape[0], :] = t

    return X_padded, X_flat


def project_with_pca(
    X_padded: torch.Tensor, pca_V: torch.Tensor, pca_mean: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project a padded descriptor tensor into a pre-fitted PCA space.

    Args:
        X_padded: Shape ``(N, A_max, D)``.
        pca_V:    PCA projection matrix.
        pca_mean: Per-feature mean.

    Returns:
        Tuple ``(X_proj_padded, mask)``.
    """
    logger.info("Projecting descriptors into PCA space...")

    mask = torch.any(X_padded != 0, dim=-1)
    valid = X_padded[mask]
    centred = valid - pca_mean
    projected = torch.mm(centred, pca_V)

    n_comp = pca_V.shape[1]
    X_proj = torch.zeros(
        X_padded.shape[0],
        X_padded.shape[1],
        n_comp,
        device=X_padded.device,
        dtype=X_padded.dtype,
    )
    X_proj[mask] = projected
    return X_proj, mask


def evaluate_structure_metric(
    gmm: TorchGMM,
    X_padded: torch.Tensor,
    pca_V: torch.Tensor,
    pca_mean: torch.Tensor,
    metric: str = "log_likelihood",
) -> torch.Tensor:
    """
    Unified evaluation of structure-level GMM metrics using min-pooling over atoms
    (the 'weakest link' principle).

    Args:
        gmm:        The trained GMM.
        X_padded:   Padded atom descriptors. Shape: (N_structures, N_atoms_max, N_channels)
        pca_V:      The PCA projection matrix (V) from the training set.
        pca_mean:   The feature mean from the training set.
        metric:     Either 'log_likelihood' (for uncertainty) or 'posterior' (for certainty).

    Returns:
        torch.Tensor: 1D tensor of shape (N_structures,) containing the structure-level score.
    """
    logger.info(f"Evaluating structure-level GMM metrics using {metric}...")
    device = X_padded.device
    dtype = X_padded.dtype

    # Project into PCA space
    X_proj, mask = project_with_pca(X_padded, pca_V, pca_mean)
    n_structures, n_atoms_max = mask.shape
    valid_proj = X_proj[mask]

    # Get atomic scores from the GMM
    with torch.no_grad():
        if metric == "log_likelihood":
            atomic_scores_flat = gmm.score(valid_proj)
        elif metric == "posterior":
            log_post = gmm.posterior_log_probs(valid_proj)
            max_log_post, _ = torch.max(log_post, dim=1)
            atomic_scores_flat = torch.exp(max_log_post)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Scatter valid scores back to their original rectangular positions.
    # We initialise with +inf so padded slots are ignored by the downstream min-pool.
    atomic_scores_2d = torch.full(
        (n_structures, n_atoms_max), float("inf"), device=device, dtype=dtype
    )
    atomic_scores_2d[mask] = atomic_scores_flat

    # Min-pool across atoms for each structure
    structure_score, _ = torch.min(atomic_scores_2d, dim=1)

    return structure_score


def evaluate_pool_uncertainty(best_gmm, pca_V, pca_mean, new_X_padded):
    """
    Evaluates the uncertainty of new molecular structures using a trained GMM.
    Uncertainty is defined as the negative log-likelihood of the weakest atom.

    Args:
        best_gmm (TorchGMM): The GMM trained on the training set PCA features.
        pca_V (torch.Tensor): The PCA projection matrix (V) from the training set.
        pca_mean (torch.Tensor): The feature mean from the training set.
        new_X_padded (torch.Tensor): Padded descriptors of new structures.

    Returns:
        torch.Tensor: 1D tensor of certainty scores (Higher score = More uncertain).
    """
    # Negative log-likelihood: high output = high uncertainty
    return -evaluate_structure_metric(
        best_gmm, new_X_padded, pca_V, pca_mean, metric="log_likelihood"
    )


def evaluate_structure_cluster_probability(gmm, X_padded, pca_V, pca_mean):
    """
    Computes structure-level GMM certainty using log-sum-exp-stable posteriors.
    Certainty is the minimum across atoms of the max component posterior.

    Args:
        gmm:        Fitted TorchGMM instance.
        X_padded:   Padded descriptor tensor.
        pca_V:      PCA projection matrix.
        pca_mean:   PCA feature mean.

    Returns:
        torch.Tensor: 1D tensor of certainty per structure in [0, 1].
    """
    return evaluate_structure_metric(gmm, X_padded, pca_V, pca_mean, metric="posterior")


def get_certainty_threshold(train_uncertainty_scores, certainty_percentile=0.80):
    """
    Calculates the uncertainty threshold corresponding to a specific certainty level
    based on the training data distribution.

    Args:
        train_uncertainty_scores (torch.Tensor): 1D tensor of uncertainty scores for the TRAINING set.
        certainty_percentile (float): The target certainty level (e.g., 0.80 for 80%).

    Returns:
        float: The threshold score. Structures with a score HIGHER than this are below the certainty percentile.
    """
    # Sort training scores in ascending order (most certain to least certain)
    sorted_train_scores, _ = torch.sort(train_uncertainty_scores)

    # Find the index that corresponds to the percentile
    threshold_idx = int(certainty_percentile * len(sorted_train_scores))

    # Ensure index is within bounds
    threshold_idx = min(threshold_idx, len(sorted_train_scores) - 1)

    threshold_value = sorted_train_scores[threshold_idx].item()
    logger.info(
        f"Certainty threshold for {certainty_percentile * 100}% quantile: {threshold_value}"
    )

    return threshold_value


def select_uncertain_structures(pool_uncertainty_scores, threshold):
    """
    Selects indices of structures from the pool that have an uncertainty score
    higher than the defined threshold (i.e., they are 'below the certainty limit').

    Args:
        pool_uncertainty_scores (torch.Tensor): 1D tensor of scores for the NEW pool.
        threshold (float): The threshold calculated from get_certainty_threshold.

    Returns:
        torch.Tensor: 1D tensor containing the indices of the selected structures.
    """
    # Find all indices where the pool score is strictly greater than the threshold
    # (Greater score = higher uncertainty = lower certainty)
    return torch.where(pool_uncertainty_scores > threshold)[0]


def torch_pca_dynamic(X, threshold=0.95):
    logger.info(f"Running PCA with threshold={threshold}...")
    # 1. Center the data
    mean = torch.mean(X, dim=0)
    X_centered = X - mean

    # 2. SVD
    _U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    # 3. Variance Calculation
    explained_variance = (S**2) / (X.shape[0] - 1)
    total_var = torch.sum(explained_variance)

    # Handle edge case: Data has no variance
    if total_var < 1e-10:
        logger.warning("Data has near-zero variance. Using 1 component.")
        return X_centered[:, :1], Vh[:1].T, mean, 1

    explained_variance_ratio = explained_variance / total_var
    cumulative_variance = torch.cumsum(explained_variance_ratio, dim=0)

    # 4. Find n_components with a fallback
    # Look for the first index where threshold is met
    mask = cumulative_variance >= threshold
    indices = torch.where(mask)[0]

    if len(indices) > 0:
        n_components = indices[0].item() + 1
    else:
        # Fallback: take all components if threshold is never reached
        n_components = X.shape[1]
        logger.warning(
            f"Threshold {threshold} not met. Using all {n_components} components."
        )

    actual_variance = cumulative_variance[n_components - 1].item()
    logger.info(
        f"Captured {actual_variance * 100:.2f}% variance with {n_components} components."
    )

    # 5. Project data
    V = Vh[:n_components].T
    X_reduced = torch.mm(X_centered, V)

    return X_reduced, V, mean, n_components
