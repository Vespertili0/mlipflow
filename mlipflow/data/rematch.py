from __future__ import annotations

import logging

import torch

from mlipflow.data import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _compute_batched_rematch_block(
    X_a: torch.Tensor,
    X_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    gamma: float = 0.1,
    zeta: int = 1,
    sinkhorn_iters: int = 100,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute the raw (unnormalised) ReMATCH cross-similarity block between two
    batches of atomic configurations using entropy-regularised optimal transport.

    This is the inner kernel operating on a single ``(B_a, B_b)`` tile of the
    full ``(B, B)`` similarity matrix.

    Parameters
    ----------
    X_a : torch.Tensor
        Padded local descriptors for the first batch.
        Shape: ``(B_a, A_max_a, D)``.
    X_b : torch.Tensor
        Padded local descriptors for the second batch.
        Shape: ``(B_b, A_max_b, D)``.
    mask_a : torch.Tensor
        Boolean validity mask for ``X_a``.  Shape: ``(B_a, A_max_a)``.
    mask_b : torch.Tensor
        Boolean validity mask for ``X_b``.  Shape: ``(B_b, A_max_b)``.
    gamma : float, optional
        Entropic regularisation strength.  Default ``0.1``.
    zeta : int, optional
        Sharpness power applied to dot-product similarities.  Default ``1``.
    sinkhorn_iters : int, optional
        Number of Sinkhorn alternating projection iterations.  Default ``30``.
    eps : float, optional
        Additive stabilisation constant to prevent division by zero.
        Default ``1e-12``.

    Returns
    -------
    torch.Tensor
        Raw cross-similarity scores.  Shape: ``(B_a, B_b)``.
    """
    dtype = X_a.dtype
    B_a, A_max_a, _D = X_a.shape
    B_b, A_max_b, _ = X_b.shape

    # 1. Pairwise atomic dot-product similarities: (B_a, B_b, A_max_a, A_max_b)
    S = torch.einsum("aid,bjd->abij", X_a, X_b)
    if zeta != 1:
        S = torch.pow(torch.clamp(S, min=0.0), zeta)

    # 2. 4D validity mask to neutralise padding entries
    mask_4d = mask_a.unsqueeze(1).unsqueeze(3) & mask_b.unsqueeze(0).unsqueeze(2)
    mask_4d_float = mask_4d.to(dtype)

    # 3. Uniform marginal weight vectors normalised over real atoms only
    N_a = torch.clamp(mask_a.sum(dim=1, keepdim=True).to(dtype), min=1.0)
    N_b = torch.clamp(mask_b.sum(dim=1, keepdim=True).to(dtype), min=1.0)

    r = (mask_a.to(dtype) / N_a).unsqueeze(1).expand(B_a, B_b, A_max_a)
    c = (mask_b.to(dtype) / N_b).unsqueeze(0).expand(B_a, B_b, A_max_b)

    # 4. Gibbs kernel with padding elements zeroed out
    # Stabilize S by subtracting the max value per pair
    # Stabilize S
    S_max = S.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    K = torch.exp((S - S_max) / gamma) * mask_4d_float

    u = torch.ones_like(r)
    v = torch.ones_like(c)

    # 5. Sinkhorn alternating projections
    for _ in range(sinkhorn_iters):
        Kv = torch.matmul(K, v.unsqueeze(-1)).squeeze(-1)
        u = r / (Kv + eps)

        KTu = torch.matmul(K.transpose(-1, -2), u.unsqueeze(-1)).squeeze(-1)
        v = c / (KTu + eps)

    # 6. Optimal transport plan → raw Frobenius inner product with S
    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    return torch.sum(P * S, dim=(-1, -2))


def compute_rematch_matrix(
    X_padded: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 0.1,
    zeta: int = 1,
    sinkhorn_iters: int = 100,
    eps: float = 1e-12,
    block_size: int = 512,
) -> torch.Tensor:
    """
    Compute the normalised global ReMATCH cross-similarity matrix for an
    ensemble of atomic configurations.

    For small batches (``B <= block_size``) the full ``(B, B)`` matrix is
    computed in one pass.  For larger batches the output is assembled
    tile-by-tile to cap peak GPU memory at
    ``O(block_size² x A_max²)`` instead of ``O(B² x A_max²)``.

    If the computed diagonal deviates from 1.0 under ``torch.float32``
    (indicating numerical instability), the entire computation is
    automatically re-run in ``torch.float64`` with a logged warning.

    Parameters
    ----------
    X_padded : torch.Tensor
        Padded local atomic descriptors.  Shape: ``(B, A_max, D)``.
    mask : torch.Tensor
        Boolean validity mask.  Shape: ``(B, A_max)``.
        ``True`` for real atoms, ``False`` for padding.
    gamma : float, optional
        Entropic regularisation strength.  Default ``0.1``.
    zeta : int, optional
        Sharpness power applied to dot-product similarities.  Default ``1``.
    sinkhorn_iters : int, optional
        Number of Sinkhorn alternating projection iterations.  Default ``30``.
    eps : float, optional
        Additive stabilisation constant.  Default ``1e-12``.
    block_size : int, optional
        Maximum tile width for blockwise computation.  Default ``512``.

    Returns
    -------
    torch.Tensor
        Symmetric normalised similarity matrix.  Shape: ``(B, B)``.
        All values lie in ``[0.0, 1.0]`` with the diagonal exactly ``1.0``
        (up to floating-point precision).
    """
    result = _compute_rematch_matrix_impl(
        X_padded, mask, gamma, zeta, sinkhorn_iters, eps, block_size
    )

    # Precision fallback: if float32 produced unstable diagonals, retry in float64
    diag = torch.diagonal(result, dim1=0, dim2=1)
    if X_padded.dtype == torch.float32 and diag.min().item() < 0.999:
        logger.warning(
            "ReMATCH diagonal instability detected under float32 "
            "(min diag=%.6f). Retrying in float64.",
            diag.min().item(),
        )
        result = _compute_rematch_matrix_impl(
            X_padded.to(torch.float64),
            mask,
            gamma,
            zeta,
            sinkhorn_iters,
            eps,
            block_size,
        )

    return result


def _compute_rematch_matrix_impl(
    X_padded: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    zeta: int,
    sinkhorn_iters: int,
    eps: float,
    block_size: int,
) -> torch.Tensor:
    """
    Internal implementation that computes the full normalised ReMATCH matrix,
    optionally using blockwise tiling.

    Parameters
    ----------
    X_padded : torch.Tensor
        Padded descriptors.  Shape: ``(B, A_max, D)``.
    mask : torch.Tensor
        Validity mask.  Shape: ``(B, A_max)``.
    gamma : float
        Entropic regularisation strength.
    zeta : int
        Sharpness power.
    sinkhorn_iters : int
        Number of Sinkhorn iterations.
    eps : float
        Stabilisation constant.
    block_size : int
        Tile width for blockwise computation.

    Returns
    -------
    torch.Tensor
        Normalised similarity matrix.  Shape: ``(B, B)``.
    """
    B = X_padded.shape[0]
    dtype = X_padded.dtype
    device = X_padded.device
    kernel_kwargs = {
        "gamma": gamma,
        "zeta": zeta,
        "sinkhorn_iters": sinkhorn_iters,
        "eps": eps,
    }

    raw = torch.zeros(B, B, dtype=dtype, device=device)

    # Tile the computation when the batch exceeds block_size
    for i_start in range(0, B, block_size):
        i_end = min(i_start + block_size, B)
        X_i = X_padded[i_start:i_end]
        m_i = mask[i_start:i_end]

        for j_start in range(0, B, block_size):
            j_end = min(j_start + block_size, B)
            X_j = X_padded[j_start:j_end]
            m_j = mask[j_start:j_end]

            raw[i_start:i_end, j_start:j_end] = _compute_batched_rematch_block(
                X_i, X_j, m_i, m_j, **kernel_kwargs
            )

    # Enforce symmetry on the raw matrix
    raw = 0.5 * (raw + raw.transpose(0, 1))

    # Symmetric normalisation: K_norm(a,b) = K_raw(a,b) / sqrt(K_raw(a,a) * K_raw(b,b))
    diag = torch.diagonal(raw, dim1=0, dim2=1)
    v_sqrt = torch.sqrt(torch.clamp(diag, min=eps))
    normalisation = torch.outer(v_sqrt, v_sqrt)

    return raw / (normalisation + eps)
