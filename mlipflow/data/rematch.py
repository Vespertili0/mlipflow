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


def _get_energy(atoms, prefix: str) -> float:
    """Extract total energy from an ``Atoms`` object.

    The function walks a priority-ordered list of ``atoms.info`` keys,
    returning the first valid float it finds.  Returns ``inf`` as a
    defensive fallback so that ``min()`` comparisons never raise.

    Parameters
    ----------
    atoms : ase.Atoms
        Configuration to inspect.
    prefix : str
        MLIP prefix string (e.g. ``"MACE_"``).

    Returns
    -------
    float
        Total energy in eV, or ``inf`` if none found.
    """
    for key in [
        f"{prefix}energy",
        f"{prefix}_energy",
        "energy",
        "DFT_energy",
        "last_op__md_energy",
        "last_op__optimize_energy",
    ]:
        if key in atoms.info:
            try:
                return float(atoms.info[key])
            except (ValueError, TypeError):
                pass
    return float("inf")


def _get_sort_key(atoms, prefix: str) -> tuple:
    """Build a deterministic sort key for an ``Atoms`` object.

    The key is a tuple ``(energy, positions_flat, cell_flat)`` so that
    survivors are ordered first by energy, then by geometry for
    tie-breaking.  This guarantees output ordering is independent of
    input ordering.

    Parameters
    ----------
    atoms : ase.Atoms
        Configuration to build the key for.
    prefix : str
        MLIP prefix string (e.g. ``"MACE_"``).

    Returns
    -------
    tuple
        ``(energy, positions_tuple, cell_tuple)``.
    """
    energy = _get_energy(atoms, prefix)
    pos_flat = tuple(atoms.get_positions().ravel().tolist())
    cell_flat = tuple(atoms.get_cell().array.ravel().tolist())
    return (energy, pos_flat, cell_flat)


def species_chunked_louvain_clustering(
    atoms_list: list,
    mlip_calc,
    prefix: str,
    device: str,
    *,
    energy_tol: float = 0.05,
    network_threshold: float = 0.90,
    resolution: float = 1.0,
    gamma: float = 0.1,
    zeta: int = 1,
    block_size: int = 512,
    atom_slice=None,
) -> list:
    """Cluster configurations into PES basins and return basin representatives.

    Configurations are first bucketed by species.  Within each species
    bucket that contains at least two structures, a sparse topo-energetic
    adjacency graph is constructed and partitioned using the Louvain
    algorithm.  Each community is collapsed to its lowest-energy member.

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        Pre-cleaned configurations.  Each must carry a total energy in its
        ``info`` dict and ideally a ``"species"`` label.
    mlip_calc : mace.calculators.MACECalculator
        Initialised MACE calculator for descriptor extraction.
    prefix : str
        MLIP info-key prefix (e.g. ``"MACE_"``).
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``).
    energy_tol : float, optional
        Thermal energy scale :math:`\\tau_E` in eV.  Pairs with
        :math:`|\\Delta E| > \\tau_E` are pruned (Stage 1) and surviving
        edges decay exponentially with this scale (Stage 2).
        Default ``0.05``.
    network_threshold : float, optional
        Structural similarity lower-bound :math:`\\tau_S`.  ReMATCH
        scores below this value are pruned.  Default ``0.90``.
    resolution : float, optional
        Louvain resolution parameter :math:`\\gamma`.  Values :math:`\\ge
        1.0` penalise large communities, protecting shallow micro-basins.
        Default ``1.0``.
    gamma : float, optional
        ReMATCH entropic regularisation strength.  Default ``0.1``.
    zeta : int, optional
        ReMATCH sharpness power.  Default ``1``.
    block_size : int, optional
        Tile width for blockwise ReMATCH computation.  Default ``512``.
    atom_slice : slice | list[int] | None, optional
        Atom index filter for descriptor computation.  Default ``None``.

    Returns
    -------
    list[ase.Atoms]
        Deterministically sorted list of basin representative
        configurations.
    """
    import collections

    import networkx as nx
    import numpy as np

    from mlipflow.data.gmm import compute_descriptors

    # ── 1. Bucket by species ──────────────────────────────────────────
    species_buckets: dict[str, list] = collections.defaultdict(list)
    for atoms in atoms_list:
        species = atoms.info.get("species", "unknown")
        species_buckets[species].append(atoms)

    logger.info(
        "Species bucketing: %d species from %d configurations.",
        len(species_buckets),
        len(atoms_list),
    )

    survivors: list = []

    # ── 2. Process each bucket ────────────────────────────────────────
    for species, bucket in species_buckets.items():
        if len(bucket) < 2:
            logger.info(
                "Species '%s': %d config(s) — bypassing clustering.",
                species,
                len(bucket),
            )
            survivors.extend(bucket)
            continue

        logger.info("Species '%s': clustering %d configurations.", species, len(bucket))

        # ── 2a. Descriptors & ReMATCH ────────────────────────────────
        X_padded, _X_flat = compute_descriptors(
            bucket, mlip_calc, device, torch.float32, atom_indices=atom_slice
        )
        mask = torch.any(X_padded != 0, dim=-1)

        with torch.no_grad():
            sim_mat = compute_rematch_matrix(
                X_padded, mask, gamma=gamma, zeta=zeta, block_size=block_size
            )

        # Move to CPU/numpy for networkx
        S = sim_mat.cpu().numpy()
        n = len(bucket)

        # ── 2b. Energy difference matrix ──────────────────────────────
        energies = np.array([_get_energy(at, prefix) for at in bucket])
        delta_E = np.abs(energies[:, None] - energies[None, :])

        # ── 2c. Two-stage topo-energetic edge weighting ───────────────
        #  Stage 1: Threshold gate (pruning)
        gate = (network_threshold <= S) & (delta_E <= energy_tol)
        np.fill_diagonal(gate, False)  # no self-loops

        #  Stage 2: Continuous weight function on surviving edges
        W = np.zeros((n, n), dtype=np.float64)
        surviving = gate.nonzero()
        if surviving[0].size > 0:
            s_vals = S[surviving]
            de_vals = delta_E[surviving]
            structural_weight = (s_vals - network_threshold) / (1.0 - network_threshold)
            thermal_decay = np.exp(-de_vals / energy_tol)
            W[surviving] = structural_weight * thermal_decay

        # ── 2d. Graph construction & Louvain ──────────────────────────
        G = nx.from_numpy_array(W)
        communities = nx.community.louvain_communities(
            G, weight="weight", resolution=resolution, seed=42
        )

        logger.info(
            "Species '%s': Louvain found %d communities from %d configs.",
            species,
            len(communities),
            n,
        )

        # ── 2e. Representative selection ──────────────────────────────
        for community in communities:
            best_idx = min(
                community, key=lambda idx: (_get_sort_key(bucket[idx], prefix), idx)
            )
            survivors.append(bucket[best_idx])

    # ── 3. Deterministic sort ─────────────────────────────────────────
    survivors.sort(key=lambda at: _get_sort_key(at, prefix))

    logger.info(
        "Louvain basin collapse complete: %d -> %d survivors.",
        len(atoms_list),
        len(survivors),
    )

    return survivors
