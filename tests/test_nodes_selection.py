from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator

from mlipflow.data.llpr import PerElementLLPR
from mlipflow.graphflow.nodes import (
    EnsembleState,
    run_config_fps_selection,
    run_config_uncertainty_selection,
    run_mlip_sp,
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

        # Disk idempotency guard: non-empty output skips select_by_uncertainty and restores selected_count
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write(
            out_path,
            [Atoms("H", positions=[[0, 0, 0]]), Atoms("H", positions=[[1, 0, 0]])],
        )

        with patch("mlipflow.graphflow.nodes.select_by_uncertainty") as mock_skip:
            res_skip = run_config_uncertainty_selection(state)
            mock_skip.assert_not_called()
            assert res_skip["configs"] == [str(out_path)]
            assert res_skip["selected_count"] == 2
            assert res_skip["step_counter"] == 10

        # When use_inline_uq is omitted from calculation_kwargs, workflow node defaults to True
        state_default_inline = EnsembleState(
            configs=["pool.xyz"],
            iteration=2,
            step_counter=1,
            mlip_strategy=mlip_strategy,
            uq_file="custom_v0_llpr.pt",
            calculation_kwargs={"uncertainty_selection": {"force_tol_eVA": 0.10}},
        )
        with patch(
            "mlipflow.graphflow.nodes.select_by_uncertainty", return_value=[Atoms("H")]
        ) as mock_default:
            run_config_uncertainty_selection(state_default_inline)
            assert mock_default.call_args.kwargs["use_inline_uq"] is True
    finally:
        os.chdir(cwd)


def test_run_mlip_sp_to_uncertainty_selection_inline_prefilter(tmp_path):
    """End-to-end run_mlip_sp -> run_config_uncertainty_selection skips descriptors on below-threshold frames."""
    data_dir = Path(__file__).resolve().parent / "data"
    model_path = data_dir / "mace_test.model"
    src_xyz = data_dir / "test_data.xyz"
    all_frames = read(src_xyz, ":")
    atom_slice = slice(48, None)

    mace_calc = MACECalculator(
        model_paths=str(model_path), device="cpu", default_dtype="float32"
    )
    train_frames = all_frames[:6]
    val_frames = [at.copy() for at in all_frames[6:10]]
    rng = np.random.default_rng(31)
    for at in val_frames:
        at_c = at.copy()
        at_c.calc = mace_calc
        f_mace = at_c.get_forces()
        at.arrays["MACE_forces"] = f_mace
        at.arrays["DFT_forces"] = f_mace + rng.normal(0.0, 0.03, size=f_mace.shape)

    llpr = PerElementLLPR(invariants_only=True, device="cpu")
    llpr.fit(train_frames, mace_calc, atom_indices=atom_slice, exclude_fixed=False)
    llpr.calibrate(
        val_frames,
        mace_calc,
        atom_indices=atom_slice,
        exclude_fixed=False,
        deltas=(0.50, 0.20, 0.10),
        tune_lambdas=(1e-4,),
    )
    uq_file = tmp_path / "mace_test_llpr.pt"
    llpr.save(uq_file)

    eq_frames = [train_frames[0].copy(), train_frames[1].copy()]
    for at in eq_frames:
        at.info["cluster"] = "eq"

    ood_frame = train_frames[0].copy()
    pos = ood_frame.get_positions()
    pos[48:] += np.random.default_rng(123).normal(
        0.0, 0.25, size=(len(ood_frame) - 48, 3)
    )
    ood_frame.set_positions(pos)
    ood_frame.info["cluster"] = "ood"

    _, eq_scores = llpr.score_configs(
        eq_frames, mace_calc, atom_indices=atom_slice, exclude_fixed=False, delta=0.50
    )
    _, ood_score = llpr.score_atoms(
        ood_frame, mace_calc, atom_indices=atom_slice, exclude_fixed=False, delta=0.50
    )
    assert ood_score > float(np.max(eq_scores))
    tol = 0.5 * (float(np.max(eq_scores)) + float(ood_score))

    # Pool has 2 below-tol equilibrium frames + 2 identical passing OOD frames
    pool_xyz = tmp_path / "merged_pool.xyz"
    write(pool_xyz, [*eq_frames, ood_frame.copy(), ood_frame.copy()])

    mlip_strategy = MACEModel(
        mlip_name=str(data_dir / "mace_test"), run_mode="local", uq_file=str(uq_file)
    )

    state = EnsembleState(
        configs=[str(pool_xyz)],
        iteration=1,
        step_counter=8,
        mlip_strategy=mlip_strategy,
        uq_file=str(uq_file),
        calculation_kwargs={
            "mlip_sp": {"dispersion": False},
            "uncertainty_selection": {
                "force_tol_eVA": tol,
                "atom_indices": atom_slice,
                "exclude_fixed": False,
                "delta": 0.50,
            },
        },
    )

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        # 1. run_mlip_sp inherits atom_indices, exclude_fixed, and uq_file -> writes inline UQ + provenance
        sp_state = run_mlip_sp(state)
        sp_frames = read(sp_state["configs"][0], ":")
        assert len(sp_frames) == 4
        for at in sp_frames:
            assert "MACE_max_uq" in at.info
            assert "MACE_max_uq_p90" in at.info
            assert "MACE_uq_delta" in at.info
            assert at.info["MACE_uq_model"] == uq_file.name

        # 2. run_config_uncertainty_selection uses inline pre-filter (default use_inline_uq=True),
        # skipping get_descriptors on the 2 equilibrium frames and running it ONLY on the 2 OOD frames!
        desc_call_count = 0
        orig_get_desc = MACECalculator.get_descriptors

        def _spy_get_descriptors(self, *args, **kwargs):
            nonlocal desc_call_count
            desc_call_count += 1
            return orig_get_desc(self, *args, **kwargs)

        with patch.object(MACECalculator, "get_descriptors", new=_spy_get_descriptors):
            sel_state = run_config_uncertainty_selection(sp_state)

        assert desc_call_count == 2
        selected = read(sel_state["configs"][0], ":")
        # Sequential Batch LLPR collapses the 2 identical OOD copies into 1 non-redundant selection
        assert len(selected) == 1
        assert sel_state["selected_count"] == 1
        assert selected[0].info["cluster"] == "ood"
    finally:
        os.chdir(cwd)
