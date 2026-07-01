from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from ase import Atoms

from mlipflow.data.rematch import (
    _get_energy,
    _get_sort_key,
    species_chunked_louvain_clustering,
)
from mlipflow.graphflow.nodes import EnsembleState, run_rematch_basin_collapse

# ── Helpers ───────────────────────────────────────────────────────────


def _make_atoms(energy: float, species: str = "unknown", position=None):
    """Create an Atoms object with an energy label and optional species."""
    pos = position if position is not None else [[0, 0, 0]]
    at = Atoms("H", positions=pos)
    at.info["energy"] = energy
    at.info["species"] = species
    return at


# ══════════════════════════════════════════════════════════════════════
#  Node-level validation tests (orchestrator in nodes.py)
# ══════════════════════════════════════════════════════════════════════


class TestBasinCollapseValidation:
    """Validation and edge-case tests for run_rematch_basin_collapse."""

    def test_missing_configs_raises(self):
        """Empty configs list must raise ValueError."""
        state = EnsembleState(configs=[])
        with pytest.raises(ValueError, match="No configurations provided"):
            run_rematch_basin_collapse(state)

    def test_missing_mlip_strategy_raises(self):
        """Missing mlip_strategy key must raise KeyError."""
        state = EnsembleState(configs=["input.xyz"])
        with pytest.raises(KeyError, match="mlip_strategy"):
            run_rematch_basin_collapse(state)

    @patch("mlipflow.graphflow.nodes.ConfigSet")
    def test_single_config_skips(self, mock_configset):
        """A single configuration should skip collapse and return unchanged."""
        at = Atoms("H", positions=[[0, 0, 0]])
        mock_configset.return_value = [at]

        mlip_mock = MagicMock()
        mlip_mock.model_file = "dummy.model"

        state = EnsembleState(configs=["input.xyz"], mlip_strategy=mlip_mock)

        with patch("mace.calculators.MACECalculator"):
            result = run_rematch_basin_collapse(state)

        assert result["configs"] == ["input.xyz"]


# ══════════════════════════════════════════════════════════════════════
#  Node-level integration test (delegates to clustering)
# ══════════════════════════════════════════════════════════════════════


class TestBasinCollapseOrchestration:
    """Verify the orchestrator delegates correctly and writes output."""

    @patch("mlipflow.graphflow.nodes.OutputSpec")
    @patch("mlipflow.graphflow.nodes.resolve_step_path")
    @patch("mlipflow.data.rematch.species_chunked_louvain_clustering")
    @patch("mace.calculators.MACECalculator")
    @patch("mlipflow.graphflow.nodes.ConfigSet")
    def test_delegates_to_clustering(
        self,
        mock_configset,
        mock_mace_calc,
        mock_clustering,
        mock_resolve,
        mock_output_spec,
    ):
        """The node must delegate to species_chunked_louvain_clustering."""
        mock_resolve.return_value = "collapsed_basin_configs.xyz"

        atoms_list = [_make_atoms(0.0), _make_atoms(1.0)]
        mock_configset.return_value = atoms_list

        # Clustering returns one survivor
        survivor = atoms_list[0]
        mock_clustering.return_value = [survivor]

        mock_writer = MagicMock()
        mock_output_spec.return_value = mock_writer

        mlip_mock = MagicMock()
        mlip_mock.model_file = "dummy.model"
        mlip_mock.mlip_prefix = "MACE_"

        state = EnsembleState(configs=["input.xyz"], mlip_strategy=mlip_mock)

        result = run_rematch_basin_collapse(state)

        assert result["configs"] == ["collapsed_basin_configs.xyz"]
        assert result["step_counter"] == 2
        mock_clustering.assert_called_once()
        mock_writer.write.assert_called_once()


# ══════════════════════════════════════════════════════════════════════
#  Core clustering function tests (in rematch.py)
# ══════════════════════════════════════════════════════════════════════


class TestSpeciesChunkedLouvainClustering:
    """Tests for the core species_chunked_louvain_clustering function."""

    @patch("mlipflow.data.rematch.compute_rematch_matrix")
    @patch("mlipflow.data.gmm.compute_descriptors")
    def test_groups_identical_structures(self, mock_compute_desc, mock_rematch):
        """Three identical + one distinct should collapse to 2."""
        atoms_list = []
        for i in range(4):
            at = _make_atoms(float(i))
            atoms_list.append(at)

        X_padded = torch.randn(4, 3, 8)
        mock_compute_desc.return_value = (X_padded, torch.randn(12, 8))

        # Similarity: 0,1,2 are identical (sim > 0.90), 3 is distinct
        sim = torch.eye(4)
        sim[0, 1] = sim[1, 0] = 0.995
        sim[0, 2] = sim[2, 0] = 0.995
        sim[1, 2] = sim[2, 1] = 0.995
        sim[3, :3] = sim[:3, 3] = 0.5
        mock_rematch.return_value = sim

        mlip_calc = MagicMock()

        survivors = species_chunked_louvain_clustering(
            atoms_list,
            mlip_calc,
            "MACE_",
            "cpu",
            energy_tol=5.0,  # High tolerance so energy gate doesn't prune
            network_threshold=0.90,
        )

        assert len(survivors) == 2

    @patch("mlipflow.data.rematch.compute_rematch_matrix")
    @patch("mlipflow.data.gmm.compute_descriptors")
    def test_keeps_lowest_energy(self, mock_compute_desc, mock_rematch):
        """Among duplicates, the lowest-energy configuration is retained."""
        energies = [5.0, 1.0, 3.0]
        atoms_list = [_make_atoms(e) for e in energies]

        X_padded = torch.randn(3, 2, 6)
        mock_compute_desc.return_value = (X_padded, torch.randn(6, 6))

        sim = torch.ones(3, 3)
        mock_rematch.return_value = sim

        mlip_calc = MagicMock()

        survivors = species_chunked_louvain_clustering(
            atoms_list,
            mlip_calc,
            "MACE_",
            "cpu",
            energy_tol=10.0,
            network_threshold=0.90,
        )

        assert len(survivors) == 1
        assert survivors[0].info["energy"] == 1.0

    @patch("mlipflow.data.rematch.compute_rematch_matrix")
    @patch("mlipflow.data.gmm.compute_descriptors")
    def test_keeps_lowest_energy_info_dict(self, mock_compute_desc, mock_rematch):
        """Energy lookup via info dict keys (no calculator attached)."""
        energies = [5.0, 1.0, 3.0]
        atoms_list = []
        for e in energies:
            at = Atoms("H", positions=[[0, 0, 0]])
            at.info["MACE_energy"] = e
            atoms_list.append(at)

        X_padded = torch.randn(3, 2, 6)
        mock_compute_desc.return_value = (X_padded, torch.randn(6, 6))

        sim = torch.ones(3, 3)
        mock_rematch.return_value = sim

        mlip_calc = MagicMock()

        survivors = species_chunked_louvain_clustering(
            atoms_list,
            mlip_calc,
            "MACE_",
            "cpu",
            energy_tol=10.0,
            network_threshold=0.90,
        )

        assert len(survivors) == 1
        assert survivors[0].info["MACE_energy"] == 1.0


# ══════════════════════════════════════════════════════════════════════
#  Species bucketing tests
# ══════════════════════════════════════════════════════════════════════


class TestSpeciesBucketing:
    """Tests for species-level chunking behaviour."""

    def test_singleton_bucket_bypasses_clustering(self):
        """A species with only 1 config must bypass clustering entirely."""
        at = _make_atoms(2.0, species="lone_species")

        # No mocking of compute_descriptors needed — should never be called
        mlip_calc = MagicMock()

        survivors = species_chunked_louvain_clustering([at], mlip_calc, "MACE_", "cpu")

        assert len(survivors) == 1
        assert survivors[0] is at

    @patch("mlipflow.data.rematch.compute_rematch_matrix")
    @patch("mlipflow.data.gmm.compute_descriptors")
    def test_multiple_species_processed_independently(
        self, mock_compute_desc, mock_rematch
    ):
        """Configs from different species must never be compared."""
        # Species A: 2 configs (identical, should collapse to 1)
        a1 = _make_atoms(1.0, species="A")
        a2 = _make_atoms(2.0, species="A")

        # Species B: 1 config (singleton, bypasses)
        b1 = _make_atoms(0.5, species="B")

        X_padded = torch.randn(2, 3, 8)
        mock_compute_desc.return_value = (X_padded, torch.randn(6, 8))

        # Both A configs are highly similar
        sim = torch.ones(2, 2)
        mock_rematch.return_value = sim

        mlip_calc = MagicMock()

        survivors = species_chunked_louvain_clustering(
            [a1, a2, b1],
            mlip_calc,
            "MACE_",
            "cpu",
            energy_tol=5.0,
            network_threshold=0.90,
        )

        # A collapses to 1, B kept as singleton → 2 total
        assert len(survivors) == 2
        # compute_descriptors called once (for species A only)
        mock_compute_desc.assert_called_once()


# ══════════════════════════════════════════════════════════════════════
#  Energy gating tests
# ══════════════════════════════════════════════════════════════════════


class TestEnergyGating:
    """Tests for the Stage 1 energy tolerance gate."""

    @patch("mlipflow.data.rematch.compute_rematch_matrix")
    @patch("mlipflow.data.gmm.compute_descriptors")
    def test_high_energy_diff_prevents_merging(self, mock_compute_desc, mock_rematch):
        """Configs with large energy difference must not merge."""
        # Two structurally identical configs but energy gap = 10.0 eV
        atoms_list = [_make_atoms(0.0), _make_atoms(10.0)]

        X_padded = torch.randn(2, 3, 8)
        mock_compute_desc.return_value = (X_padded, torch.randn(6, 8))

        sim = torch.ones(2, 2)
        mock_rematch.return_value = sim

        mlip_calc = MagicMock()

        survivors = species_chunked_louvain_clustering(
            atoms_list,
            mlip_calc,
            "MACE_",
            "cpu",
            energy_tol=0.05,  # Tight tolerance
            network_threshold=0.90,
        )

        # Energy gap >> energy_tol → both must survive as separate basins
        assert len(survivors) == 2


# ══════════════════════════════════════════════════════════════════════
#  Deterministic sorting tests
# ══════════════════════════════════════════════════════════════════════


class TestDeterministicSorting:
    """Tests for deterministic output ordering."""

    @patch("mlipflow.data.rematch.compute_rematch_matrix")
    @patch("mlipflow.data.gmm.compute_descriptors")
    def test_output_sorted_by_energy(self, mock_compute_desc, mock_rematch):
        """Survivors must be sorted by energy regardless of input order."""
        # Input order: high, low, mid
        atoms_list = [_make_atoms(3.0), _make_atoms(1.0), _make_atoms(2.0)]

        X_padded = torch.randn(3, 3, 8)
        mock_compute_desc.return_value = (X_padded, torch.randn(9, 8))

        # All dissimilar → each is its own basin
        sim = torch.eye(3) * 0.5
        sim.fill_diagonal_(1.0)
        mock_rematch.return_value = sim

        mlip_calc = MagicMock()

        survivors = species_chunked_louvain_clustering(
            atoms_list,
            mlip_calc,
            "MACE_",
            "cpu",
            energy_tol=0.01,
            network_threshold=0.90,
        )

        assert len(survivors) == 3
        energies = [s.info["energy"] for s in survivors]
        assert energies == sorted(energies)


# ══════════════════════════════════════════════════════════════════════
#  Helper function tests
# ══════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Tests for _get_energy and _get_sort_key."""

    def test_get_energy_from_info(self):
        """Energy should be retrieved from info dict."""
        at = Atoms("H", positions=[[0, 0, 0]])
        at.info["MACE_energy"] = 42.0
        assert _get_energy(at, "MACE_") == 42.0

    def test_get_energy_fallback_to_inf(self):
        """Missing energy should return inf."""
        at = Atoms("H", positions=[[0, 0, 0]])
        assert _get_energy(at, "MACE_") == float("inf")

    def test_get_energy_prefix_variants(self):
        """All prefix variants should be discovered."""
        at = Atoms("H", positions=[[0, 0, 0]])
        at.info["DFT_energy"] = -10.0
        assert _get_energy(at, "MACE_") == -10.0

    def test_get_sort_key_returns_tuple(self):
        """Sort key must be a 3-element tuple."""
        at = _make_atoms(1.0)
        key = _get_sort_key(at, "MACE_")
        assert isinstance(key, tuple)
        assert len(key) == 3
        assert key[0] == 1.0

    def test_get_sort_key_deterministic(self):
        """Identical atoms must produce identical sort keys."""
        a = _make_atoms(1.0)
        b = _make_atoms(1.0)
        assert _get_sort_key(a, "MACE_") == _get_sort_key(b, "MACE_")
