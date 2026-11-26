from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ase import Atoms
from ase.io import read, write

from mlipflow.graphflow.nodes import (
    EnsembleState,
    run_config_fps_selection,
    run_config_uncertainty_selection,
)
from mlipflow.strategies.mlip import MACEModel


def test_run_config_fps_selection(tmp_path):
    """
    Test run_config_fps_selection using real data and dummy info fields.
    """
    # Setup paths
    test_dir = Path(__file__).resolve().parent
    src_xyz = str(Path(test_dir) / "data" / "test_data.xyz")
    config_file = tmp_path / "test_data.xyz"
    shutil.copy2(src_xyz, config_file)

    # Add dummy energy info to the configurations
    atoms = read(config_file, ":")
    rng = np.random.default_rng(42)
    new_atoms_list = []
    for at in atoms:
        # Filter to keep only Cu atoms to avoid descriptor mismatch with simple Z=29 descriptor
        # The wfl/quippy wrapper expects descriptor arrays to match atom count if applied globally
        cu_only_at = at[at.numbers == 29]
        cu_only_at.info["dummy_energy"] = rng.random()
        new_atoms_list.append(cu_only_at)
    write(config_file, new_atoms_list)

    # Change directory to tmp_path to capture outputs
    cwd = Path.cwd()
    os.chdir(tmp_path)

    try:
        # Setup EnsembleState
        # Valid descriptors for Cu (Z=29)
        descriptor_string = "soap cutoff=4.0 l_max=4 n_max=4 atom_sigma=0.5 n_Z=1 Z={29} n_species=1 species_Z={29}"

        calculation_kwargs = {
            "fps_selection": {
                "descriptor_string": descriptor_string,
                "descriptor_key": "SOAP",
                "info_field": "dummy_energy",
                "n_optimal": 5,
                "seed": 42,
            }
        }

        # Mock strategies as they are required keys in EnsembleState TypedDict
        qchem_mock = MagicMock()
        mlip_mock = MagicMock()

        state = EnsembleState(
            configs=[str(config_file)],
            qchem_strategy=qchem_mock,
            mlip_strategy=mlip_mock,
            calculation_kwargs=calculation_kwargs,
        )

        # Run selection
        result = run_config_fps_selection(state)

        # Verify output
        output_configs = result["configs"]
        assert len(output_configs) == 1
        output_file = output_configs[0]

        assert Path(output_file).exists()

        # Verify content
        selected_atoms = read(output_file, ":")
        assert len(selected_atoms) == 5

        # Verify descriptors are present (ConfigurationSelector adds them)
        assert "SOAP" in selected_atoms[0].info

    finally:
        os.chdir(cwd)


def test_run_config_uncertainty_selection_kwargs_and_idempotency(tmp_path):
    """Verify run_config_uncertainty_selection forwards all LLPR kwargs and enforces disk idempotency."""
    mlip_strategy = MACEModel(mlip_name="pot_v0", run_mode="local")
    metrics = {29: {"mae": 0.03, "p90": 0.06, "max_err": 0.12}}

    sel_kwargs = {
        "force_tol_eVA": 0.12,
        "min_select_count": 3,
        "atom_indices": slice(48, None),
        "exclude_fixed": False,
        "delta": 0.50,
        "device": "cpu",
        "element_error_target": 0.08,
        "element_error_metric": "p90",
        "use_inline_uq": True,
    }

    state = EnsembleState(
        configs=["pool.xyz"],
        iteration=1,
        step_counter=9,
        mlip_strategy=mlip_strategy,
        uq_file="custom_v0_llpr.pt",
        element_metrics=metrics,
        calculation_kwargs={"uncertainty_selection": sel_kwargs},
    )

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with patch(
            "mlipflow.graphflow.nodes.select_by_uncertainty",
            return_value=[Atoms("H"), Atoms("H")],
        ) as mock_select:
            res = run_config_uncertainty_selection(state)

            mock_select.assert_called_once()
            call_kw = mock_select.call_args.kwargs
            assert call_kw["pool_file"] == ["pool.xyz"]
            assert call_kw["mlip_strategy"] is mlip_strategy
            assert call_kw["uq_file"] == "custom_v0_llpr.pt"
            assert call_kw["force_tol_eVA"] == 0.12
            assert call_kw["min_select_count"] == 3
            assert call_kw["atom_indices"] == slice(48, None)
            assert call_kw["exclude_fixed"] is False
            assert call_kw["delta"] == 0.50
            assert call_kw["device"] == "cpu"
            assert call_kw["element_metrics"] == metrics
            assert call_kw["element_error_target"] == 0.08
            assert call_kw["element_error_metric"] == "p90"
            assert call_kw["use_inline_uq"] is True

            assert res["step_counter"] == 10
            assert res["selected_count"] == 2
            out_path = Path(res["configs"][0])
            assert "selected_configs" in out_path.name

        # Disk idempotency guard: non-empty output skips select_by_uncertainty completely
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("existing selected content", encoding="utf-8")

        with patch("mlipflow.graphflow.nodes.select_by_uncertainty") as mock_skip:
            res_skip = run_config_uncertainty_selection(state)
            mock_skip.assert_not_called()
            assert res_skip["configs"] == [str(out_path)]
            assert res_skip["step_counter"] == 10
    finally:
        os.chdir(cwd)
