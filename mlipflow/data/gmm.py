import math, logging
import torch
from typing import Tuple, List, Optional
from mlipflow.utils import find_robust_elbow
from mlipflow.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TorchGMM:
    """
    Gaussian Mixture Model (GMM) implemented in PyTorch for GPU-accelerated fitting and scoring.
    Supports full covariance matrices.
    """
    def __init__(self, n_components: int, n_features: int, device: str = 'cuda', jitter: float = 1e-4, dtype: torch.dtype = torch.float32):
        self.K = n_components
        self.D = n_features
        self.device = device
        self.jitter = jitter
        self.dtype = dtype
        
        # Initialize parameters with explicit dtype
        self.mu = torch.randn(self.K, self.D, device=device, dtype=self.dtype)
        self.sigma = torch.stack([torch.eye(self.D, device=device, dtype=self.dtype) for _ in range(self.K)])
        self.pi = torch.ones(self.K, device=device, dtype=self.dtype) / self.K

    def _gaussian_log_prob(self, x):
        x = x.to(self.dtype)
        # Add jitter for numerical stability
        sigma_reg = self.sigma + torch.eye(self.D, device=self.device, dtype=self.dtype) * self.jitter
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0) # Shape: (N, K, D)
        
        L = torch.linalg.cholesky(sigma_reg) # Shape: (K, D, D)
        
        # FIX: Align batch dimensions perfectly for the solver
        # L becomes (1, K, D, D) and diff becomes (N, K, D, 1)
        # Resulting 'y' will be shape (N, K, D, 1)
        y = torch.linalg.solve_triangular(
            L.unsqueeze(0), 
            diff.unsqueeze(-1), 
            upper=False
        )
        
        # Calculate quadrance (Mahalanobis distance squared)
        quadrance = torch.sum(y**2, dim=-2).squeeze(-1) # Shape: (N, K)
        
        # Calculate log determinant and normalization constant
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
        
        # Calculate log normalisation. `math.pi` avoids PyTorch scalar upcasting issues.
        pi_term = self.D * math.log(2 * math.pi)
        log_norm = -0.5 * (pi_term + log_det)
        
        # Return log probabilities. unsqueeze(0) broadcasts log_norm across N
        return -0.5 * quadrance + log_norm.unsqueeze(0)


    def fit(self, x: torch.Tensor, iters: int = 100):
        """
        Fits the GMM parameters (mu, sigma, pi) to the data using Expectation-Maximization (EM).

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

        # Expectation-Maximization Loop
        for _ in range(iters):
            # E-Step
            log_probs = self._gaussian_log_prob(x)
            weighted_log_probs = log_probs + torch.log(self.pi)
            log_resp = weighted_log_probs - torch.logsumexp(weighted_log_probs, dim=1, keepdim=True)
            resp = torch.exp(log_resp)
            
            # M-Step
            N_k = resp.sum(dim=0) + 1e-6
            self.pi = N_k / x.shape[0]
            self.mu = torch.mm(resp.t(), x) / N_k.unsqueeze(1)
            
            diff = x.unsqueeze(1) - self.mu.unsqueeze(0)
            self.sigma = torch.einsum('nki,nkj,nk->kij', diff, diff, resp) / N_k.view(-1, 1, 1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-likelihood of each sample in x under the full mixture model.

        Args:
            x (torch.Tensor): Input data. Shape: (N_samples, N_features).
            
        Returns:
            torch.Tensor: Log-likelihood of each sample. Shape: (N_samples,).
        """
        x = x.to(self.dtype)
        log_probs = self._gaussian_log_prob(x)
        
        return torch.logsumexp(log_probs + torch.log(self.pi), dim=1)


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


def find_best_gmm(X_reduced: torch.Tensor, max_k: int = 30, n_init: int = 5, device: str = 'cuda') -> Tuple[TorchGMM, List[Tuple[int, float]]]:
    """
    Finds the best Gaussian Mixture Model (GMM) by comparing Bayes Information Criterion (BIC) 
    across different numbers of components and using the Elbow method.

    Args:
        X_reduced (torch.Tensor): Training data, typically PCA-reduced features. Shape: (N_samples, N_features).
        max_k (int, optional): Maximum number of components to evaluate. Defaults to 10.
        n_init (int, optional): Number of initializations to perform for each K to avoid local optima. Defaults to 5.
        device (str, optional): Device to perform computations on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        Tuple[TorchGMM, List[Tuple[int, float]]]: 
            - The best TorchGMM instance based on the BIC elbow curve.
            - A history list of tuples containing (K, BIC) evaluated.
    """
    best_overall_gmm = None
    bic_history = []
    gmm_history = []
    
    for k in range(1, max_k + 1):
        best_k_ll = float('-inf')
        best_k_gmm = None
        
        # Run EM multiple times for this K
        for attempt in range(n_init):
            gmm = TorchGMM(n_components=k, n_features=X_reduced.shape[1], device=device)
            gmm.fit(X_reduced, iters=100)
            
            # Score this specific initialization
            with torch.no_grad():
                ll = gmm.score(X_reduced).sum().item()
            
            # Keep the best initialization for this K
            if ll > best_k_ll:
                best_k_ll = ll
                best_k_gmm = gmm
                
        # Now calculate BIC using the BEST initialization for this K
        bic = best_k_gmm.calculate_bic(X_reduced)
        bic_history.append((k, bic))
        gmm_history.append(best_k_gmm)
        print(f"  Fit K={k} (Best of {n_init} attempts) | BIC: {bic:.2f}")
        
    # Find the elbow of the BIC curve
    bics = [b for (k, b) in bic_history]
    
    # We use start_idx=0 since max_k is usually small, so we want to search the entire curve
    optimal_k_idx = find_robust_elbow(bics, start_idx=0) - 1
    
    # Retrieve the best GMM at the elbow point
    best_overall_gmm = gmm_history[optimal_k_idx]
            
    return best_overall_gmm


def evaluate_pool_uncertainty(best_gmm, pca_V, pca_mean, new_X):
    """
    Evaluates the uncertainty of new molecular structures using a trained GMM.
    
    Args:
        best_gmm (TorchGMM): The GMM trained on the training set PCA features.
        pca_V (torch.Tensor): The PCA projection matrix (V) from the training set.
        pca_mean (torch.Tensor): The feature mean from the training set.
        new_X (torch.Tensor): Descriptors of new structures. Shape: (N_structures, N_atoms, N_channels)
        
    Returns:
        torch.Tensor: 1D tensor of shape (N_structures,) containing the uncertainty 
                      score for each structure. (Higher score = More uncertain).
    """
    device = pca_mean.device
    dtype = pca_mean.dtype
    
    # 1. Create mask internally: True for real atoms (non-zero features), False for padding
    new_mask = torch.any(new_X != 0, dim=-1) # Shape: (N_structures, N_atoms_max)
    n_structures, n_atoms_max = new_mask.shape
    
    # 2. Extract valid atoms using the mask
    # Shape becomes: (Total_Valid_Atoms, N_channels)
    valid_atoms = new_X[new_mask].to(device=device, dtype=dtype)
    
    # 3. Project into the PCA space using TRAINING mean and TRAINING V
    # This is critical: we must use the exact same space the GMM was trained on
    centered_atoms = valid_atoms - pca_mean
    projected_atoms = torch.mm(centered_atoms, pca_V)
    
    # 4. Get atomic Log-Likelihoods from the GMM
    with torch.no_grad(): # No need to track gradients for inference
        atomic_ll_flat = best_gmm.score(projected_atoms)
        
    # 5. Aggregate back to Structure-Level (Min-Pooling) efficiently without loops
    # Initialize with +inf so that padded atoms are ignored in the standard min()
    atomic_ll = torch.full((n_structures, n_atoms_max), float('inf'), device=device, dtype=dtype)
    
    # Scatter valid scores back to their original rectangular positions
    atomic_ll[new_mask] = atomic_ll_flat
    
    # Min-pool across atoms for each structure
    structure_min_ll, _ = torch.min(atomic_ll, dim=1)
    
    # Uncertainty is defined by the "weakest link" (the most unlikely atom).
    # We take the negative so that High Output = High Uncertainty.
    structure_uncertainty = -structure_min_ll
        
    return structure_uncertainty


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
    selected_indices = torch.where(pool_uncertainty_scores > threshold)[0]
    
    return selected_indices


def torch_pca_dynamic(X, threshold=0.95):
    # 1. Center the data
    mean = torch.mean(X, dim=0)
    X_centered = X - mean
    
    # 2. SVD
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    # 3. Variance Calculation
    explained_variance = (S**2) / (X.shape[0] - 1)
    total_var = torch.sum(explained_variance)
    
    # Handle edge case: Data has no variance
    if total_var < 1e-10:
        print("Warning: Data has near-zero variance. Using 1 component.")
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
        print(f"Threshold {threshold} not met. Using all {n_components} components.")
    
    actual_variance = cumulative_variance[n_components-1].item()
    print(f"Captured {actual_variance*100:.2f}% variance with {n_components} components.")
    
    # 5. Project data
    V = Vh[:n_components].T 
    X_reduced = torch.mm(X_centered, V)
    
    return X_reduced, V, mean, n_components