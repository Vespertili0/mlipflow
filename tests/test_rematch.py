from __future__ import annotations

import pytest
import torch

from mlipflow.data.rematch import compute_rematch_matrix


@pytest.fixture
def device():
    """Use CPU for deterministic CI tests."""
    return "cpu"


@pytest.fixture
def rng():
    """Seeded generator for reproducible random descriptors."""
    return torch.Generator().manual_seed(42)


class TestDiagonalIdentity:
    """Identical structures must produce a diagonal of exactly 1.0."""

    def test_identical_structures_diagonal_one(self, device, rng):
        B, A, D = 4, 6, 16
        X = torch.randn(B, A, D, generator=rng, device=device)
        mask = torch.ones(B, A, dtype=torch.bool, device=device)

        sim = compute_rematch_matrix(X, mask)
        diag = torch.diagonal(sim, dim1=0, dim2=1)

        torch.testing.assert_close(
            diag, torch.ones(B, device=device), atol=1e-5, rtol=0
        )

    def test_single_structure(self, device, rng):
        """Edge case: B=1 must yield a 1x1 matrix of value 1.0."""
        X = torch.randn(1, 5, 10, generator=rng, device=device)
        mask = torch.ones(1, 5, dtype=torch.bool, device=device)

        sim = compute_rematch_matrix(X, mask)

        assert sim.shape == (1, 1)
        assert sim.item() == pytest.approx(1.0, abs=1e-5)


class TestSymmetry:
    """ReMATCH kernel must be symmetric: K(a,b) == K(b,a)."""

    def test_symmetry(self, device, rng):
        B, A, D = 5, 8, 12
        X = torch.randn(B, A, D, generator=rng, device=device)
        mask = torch.ones(B, A, dtype=torch.bool, device=device)

        sim = compute_rematch_matrix(X, mask)

        torch.testing.assert_close(sim, sim.T, atol=1e-5, rtol=0)

    def test_symmetry_with_variable_padding(self, device, rng):
        """Symmetry must hold even when structures have different atom counts."""
        B, A_max, D = 4, 10, 8
        X = torch.randn(B, A_max, D, generator=rng, device=device)
        mask = torch.ones(B, A_max, dtype=torch.bool, device=device)
        # Vary real atom count per structure
        mask[0, 5:] = False
        mask[1, 7:] = False
        mask[2, 3:] = False

        sim = compute_rematch_matrix(X, mask)

        torch.testing.assert_close(sim, sim.T, atol=1e-5, rtol=0)


class TestPaddingNeutrality:
    """Adding zero-padded atoms must not alter similarity scores."""

    def test_padding_neutrality(self, device, rng):
        B, A, D = 3, 5, 10
        X_compact = torch.randn(B, A, D, generator=rng, device=device)
        mask_compact = torch.ones(B, A, dtype=torch.bool, device=device)

        # Pad with 4 extra zero-filled slots
        pad_width = 4
        X_padded = torch.zeros(B, A + pad_width, D, device=device)
        X_padded[:, :A, :] = X_compact
        mask_padded = torch.zeros(B, A + pad_width, dtype=torch.bool, device=device)
        mask_padded[:, :A] = True

        sim_compact = compute_rematch_matrix(X_compact, mask_compact)
        sim_padded = compute_rematch_matrix(X_padded, mask_padded)

        torch.testing.assert_close(sim_compact, sim_padded, atol=1e-5, rtol=0)


class TestGammaInvariance:
    """Self-similarity must remain 1.0 regardless of gamma."""

    @pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0, 10.0])
    def test_gamma_self_similarity_invariant(self, gamma, device, rng):
        B, A, D = 3, 6, 8
        X = torch.randn(B, A, D, generator=rng, device=device)
        mask = torch.ones(B, A, dtype=torch.bool, device=device)

        sim = compute_rematch_matrix(X, mask, gamma=gamma)
        diag = torch.diagonal(sim, dim1=0, dim2=1)

        torch.testing.assert_close(
            diag, torch.ones(B, device=device), atol=1e-4, rtol=0
        )


class TestDissimilarStructures:
    """Orthogonal descriptor vectors should produce near-zero similarity."""

    def test_dissimilar_structures_low_score(self, device):
        D = 16
        # Two structures with orthogonal descriptor directions
        X = torch.zeros(2, 1, D, device=device)
        X[0, 0, 0] = 1.0  # unit vector along dim 0
        X[1, 0, D // 2] = 1.0  # unit vector along dim D//2
        mask = torch.ones(2, 1, dtype=torch.bool, device=device)

        sim = compute_rematch_matrix(X, mask)

        # Off-diagonal should be near zero
        assert sim[0, 1].item() < 0.1
        assert sim[1, 0].item() < 0.1
        # Diagonal should remain 1.0
        assert sim[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert sim[1, 1].item() == pytest.approx(1.0, abs=1e-5)


class TestBlockwiseEquivalence:
    """Blockwise computation must match the full-batch result."""

    def test_blockwise_matches_full(self, device, rng):
        B, A, D = 8, 5, 10
        X = torch.randn(B, A, D, generator=rng, device=device)
        mask = torch.ones(B, A, dtype=torch.bool, device=device)

        sim_full = compute_rematch_matrix(X, mask, block_size=B)
        sim_blocked = compute_rematch_matrix(X, mask, block_size=3)

        torch.testing.assert_close(sim_full, sim_blocked, atol=1e-5, rtol=0)

    def test_blockwise_single_element_blocks(self, device, rng):
        """Extreme case: block_size=1 forces per-pair tiling."""
        B, A, D = 4, 3, 6
        X = torch.randn(B, A, D, generator=rng, device=device)
        mask = torch.ones(B, A, dtype=torch.bool, device=device)

        sim_full = compute_rematch_matrix(X, mask, block_size=B)
        sim_blocked = compute_rematch_matrix(X, mask, block_size=1)

        torch.testing.assert_close(sim_full, sim_blocked, atol=1e-5, rtol=0)


class TestZetaPower:
    """Non-unity zeta should still produce valid, normalised output."""

    def test_zeta_two_valid(self, device, rng):
        B, A, D = 3, 4, 8
        X = torch.randn(B, A, D, generator=rng, device=device)
        mask = torch.ones(B, A, dtype=torch.bool, device=device)

        sim = compute_rematch_matrix(X, mask, zeta=2)
        diag = torch.diagonal(sim, dim1=0, dim2=1)

        torch.testing.assert_close(
            diag, torch.ones(B, device=device), atol=1e-4, rtol=0
        )
        assert (sim >= -1e-6).all()
        assert (sim <= 1.0 + 1e-6).all()
