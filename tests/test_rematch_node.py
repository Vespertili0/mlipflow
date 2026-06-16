from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from ase import Atoms

from mlipflow.graphflow.nodes import EnsembleState, run_rematch_basin_collapse


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

        # Patch MACECalculator to prevent actual model loading
        with patch("mace.calculators.MACECalculator"):
            result = run_rematch_basin_collapse(state)

        assert result["configs"] == ["input.xyz"]


class TestBasinCollapseGrouping:
    """Tests for the grouping and energy-selection logic."""

    @patch("mlipflow.graphflow.nodes.OutputSpec")
    @patch("mlipflow.data.rematch.compute_rematch_matrix")
    @patch("mlipflow.data.gmm.compute_descriptors")
    @patch("mace.calculators.MACECalculator")
    @patch("mlipflow.graphflow.nodes.ConfigSet")
    @patch("mlipflow.graphflow.nodes.resolve_step_path")
    def test_basin_collapse_groups_identical(
        self,
        mock_resolve,
        mock_configset,
        mock_mace_calc,
        mock_compute_desc,
        mock_rematch,
        mock_output_spec,
    ):
        """Three identical + one distinct structure should collapse to 2."""
        mock_resolve.return_value = "collapsed_basin_configs.xyz"

        # Build 4 atoms objects: indices 0,1,2 are "identical", index 3 is distinct
        atoms_list = []
        for i in range(4):
            at = Atoms("H", positions=[[0, 0, 0]])
            at.info["energy"] = float(i)  # energies: 0, 1, 2, 3
            from ase.calculators.singlepoint import SinglePointCalculator

            at.calc = SinglePointCalculator(at, energy=float(i))
            atoms_list.append(at)

        mock_configset.return_value = atoms_list

        # Fake descriptors
        X_padded = torch.randn(4, 3, 8)
        mock_compute_desc.return_value = (X_padded, torch.randn(12, 8))

        # Similarity matrix: 0,1,2 are identical (sim > 0.99), 3 is distinct
        sim = torch.eye(4)
        sim[0, 1] = sim[1, 0] = 0.995
        sim[0, 2] = sim[2, 0] = 0.995
        sim[1, 2] = sim[2, 1] = 0.995
        sim[3, :3] = sim[:3, 3] = 0.5
        mock_rematch.return_value = sim

        # Mock OutputSpec write
        mock_writer = MagicMock()
        mock_output_spec.return_value = mock_writer

        mlip_mock = MagicMock()
        mlip_mock.model_file = "dummy.model"

        state = EnsembleState(configs=["input.xyz"], mlip_strategy=mlip_mock)

        result = run_rematch_basin_collapse(state)

        assert result["configs"] == ["collapsed_basin_configs.xyz"]
        assert result["step_counter"] == 2
        mock_writer.write.assert_called_once()

        # The last ConfigSet call is for the output
        final_configset_call = mock_configset.call_args_list[-1]
        collapsed = final_configset_call[0][0]
        assert len(collapsed) == 2

    @patch("mlipflow.graphflow.nodes.OutputSpec")
    @patch("mlipflow.data.rematch.compute_rematch_matrix")
    @patch("mlipflow.data.gmm.compute_descriptors")
    @patch("mace.calculators.MACECalculator")
    @patch("mlipflow.graphflow.nodes.ConfigSet")
    @patch("mlipflow.graphflow.nodes.resolve_step_path")
    def test_basin_collapse_keeps_lowest_energy(
        self,
        mock_resolve,
        mock_configset,
        mock_mace_calc,
        mock_compute_desc,
        mock_rematch,
        mock_output_spec,
    ):
        """Among duplicates, the configuration with the lowest energy is retained."""
        mock_resolve.return_value = "collapsed_basin_configs.xyz"

        atoms_list = []
        energies = [5.0, 1.0, 3.0]  # Index 1 has lowest energy
        for e in energies:
            at = Atoms("H", positions=[[0, 0, 0]])
            from ase.calculators.singlepoint import SinglePointCalculator

            at.calc = SinglePointCalculator(at, energy=e)
            atoms_list.append(at)

        mock_configset.return_value = atoms_list

        X_padded = torch.randn(3, 2, 6)
        mock_compute_desc.return_value = (X_padded, torch.randn(6, 6))

        # All three are in the same basin
        sim = torch.ones(3, 3)
        mock_rematch.return_value = sim

        mock_writer = MagicMock()
        mock_output_spec.return_value = mock_writer

        mlip_mock = MagicMock()
        mlip_mock.model_file = "dummy.model"

        state = EnsembleState(configs=["input.xyz"], mlip_strategy=mlip_mock)

        result = run_rematch_basin_collapse(state)

        assert result["step_counter"] == 2

        # Should collapse to 1 config — the one with energy 1.0 (index 1)
        final_configset_call = mock_configset.call_args_list[-1]
        collapsed = final_configset_call[0][0]
        assert len(collapsed) == 1
        assert collapsed[0].get_potential_energy() == 1.0
