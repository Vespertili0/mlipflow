from __future__ import annotations

import inspect
import os
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator

import mlipflow.data.selection as selection_mod
from mlipflow.data.llpr import PerElementLLPR
from mlipflow.data.selection import select_by_uncertainty, split_configset
from mlipflow.strategies.mlip import MACEModel


def test_split_configset_stratified_real_data(tmp_path):
    """Test stratified_random splitting on real multi-species MD dataset."""
    test_dir = Path(__file__).resolve().parent
    src_xyz = test_dir / "data" / "test_data_II.xyz"
    atoms = read(src_xyz, ":")
    for idx, at in enumerate(atoms):
        at.info["frame_id"] = idx
    in_xyz = tmp_path / "md_pool.xyz"
    write(in_xyz, atoms)

    out_1 = {
        "train": str(tmp_path / "train_1.xyz"),
        "test": str(tmp_path / "test_1.xyz"),
    }
    out_2 = {
        "train": str(tmp_path / "train_2.xyz"),
        "test": str(tmp_path / "test_2.xyz"),
    }
    out_diff_seed = {
        "train": str(tmp_path / "train_3.xyz"),
        "test": str(tmp_path / "test_3.xyz"),
    }

    split_configset(
        in_file=str(in_xyz),
        out_file=out_1,
        split_mode="stratified_random",
        test_fraction=0.2,
        seed=42,
        clean_data=False,
    )
    split_configset(
        in_file=str(in_xyz),
        out_file=out_2,
        split_mode="stratified_random",
        test_fraction=0.2,
        seed=42,
        clean_data=False,
    )
    split_configset(
        in_file=str(in_xyz),
        out_file=out_diff_seed,
        split_mode="stratified_random",
        test_fraction=0.2,
        seed=99,
        clean_data=False,
    )

    train_1 = read(out_1["train"], ":")
    test_1 = read(out_1["test"], ":")
    test_2 = read(out_2["test"], ":")
    test_diff = read(out_diff_seed["test"], ":")

    assert len(train_1) == 272
    assert len(test_1) == 68

    train_ids = {at.info["frame_id"] for at in train_1}
    test_ids = {at.info["frame_id"] for at in test_1}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids | test_ids) == len(atoms)

    # Reproducible for fixed seed, distinct for a different seed
    assert [at.info["frame_id"] for at in test_1] == [
        at.info["frame_id"] for at in test_2
    ]
    assert [at.info["frame_id"] for at in test_1] != [
        at.info["frame_id"] for at in test_diff
    ]

    # Species proportions preserved within ±1 of exact 20% per stratum
    total_counts = Counter(at.info["species"] for at in atoms)
    test_counts = Counter(at.info["species"] for at in test_1)
    for species, total_n in total_counts.items():
        expected = total_n * 0.2
        assert abs(test_counts[species] - expected) <= 1.0


def test_split_configset_singleton_species_kept_in_train(tmp_path):
    """Test that singleton species in real test_data.xyz remain in training set."""
    test_dir = Path(__file__).resolve().parent
    src_xyz = test_dir / "data" / "test_data.xyz"
    atoms = read(src_xyz, ":")

    out_files = {
        "train": str(tmp_path / "train.xyz"),
        "test": str(tmp_path / "test.xyz"),
    }
    split_configset(
        in_file=str(src_xyz),
        out_file=out_files,
        split_mode="stratified_random",
        test_fraction=0.2,
        seed=42,
        clean_data=False,
    )

    train_atoms = read(out_files["train"], ":")
    test_atoms = read(out_files["test"], ":")

    assert len(train_atoms) == 14
    assert len(test_atoms) == 3
    assert len(train_atoms) + len(test_atoms) == len(atoms)

    # FA has only 1 frame in test_data.xyz and must stay in train
    assert any(at.info.get("species") == "FA" for at in train_atoms)
    assert all(at.info.get("species") != "FA" for at in test_atoms)


@pytest.mark.parametrize("split_mode", ["random", "stratified_random"])
def test_split_configset_preserves_force_error_tail(tmp_path, split_mode):
    """Verify unbiased splitting preserves the upper force-error tail in dft_test."""
    atoms_list = []
    n_total = 100
    for i in range(n_total):
        at = Atoms("H", positions=[[0, 0, 0]])
        at.arrays["DFT_forces"] = np.array([[0.0, 0.0, 0.0]])
        error = 10.0 if i >= 80 else 0.1
        at.arrays["MACE_forces"] = np.array([[error, 0.0, 0.0]])
        at.info["DFT_energy"] = 0.0
        at.info["species"] = "sp_A" if i % 2 == 0 else "sp_B"
        at.info["index"] = i
        atoms_list.append(at)

    in_file = tmp_path / "tail_check.xyz"
    write(in_file, atoms_list)

    out_a = {
        "train": str(tmp_path / f"train_a_{split_mode}.xyz"),
        "test": str(tmp_path / f"test_a_{split_mode}.xyz"),
    }
    out_b = {
        "train": str(tmp_path / f"train_b_{split_mode}.xyz"),
        "test": str(tmp_path / f"test_b_{split_mode}.xyz"),
    }

    split_configset(
        in_file=str(in_file),
        out_file=out_a,
        split_mode=split_mode,
        test_fraction=0.2,
        seed=123,
    )
    split_configset(
        in_file=str(in_file),
        out_file=out_b,
        split_mode=split_mode,
        test_fraction=0.2,
        seed=123,
    )

    train_atoms = read(out_a["train"], ":")
    test_atoms = read(out_a["test"], ":")
    test_atoms_b = read(out_b["test"], ":")

    assert len(train_atoms) == 80
    assert len(test_atoms) == 20

    train_indices = {at.info["index"] for at in train_atoms}
    test_indices = {at.info["index"] for at in test_atoms}
    assert train_indices.isdisjoint(test_indices)
    assert len(train_indices | test_indices) == n_total

    # Upper force-error tail (indices 80..99) must NOT be truncated out of test
    tail_in_test = [idx for idx in test_indices if idx >= 80]
    assert len(tail_in_test) > 0

    # Fixed-seed reproducibility
    assert [at.info["index"] for at in test_atoms] == [
        at.info["index"] for at in test_atoms_b
    ]


def test_split_configset_missing_species_and_validation(tmp_path):
    """Test fallback when species key is missing and validation of invalid parameters."""
    atoms_list = []
    for i in range(20):
        at = Atoms("H", positions=[[0, 0, 0]])
        at.info["DFT_energy"] = 0.0
        at.info["index"] = i
        atoms_list.append(at)

    in_file = tmp_path / "no_species.xyz"
    write(in_file, atoms_list)

    # Legacy string out_file format still works in tmp_path
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        split_configset(
            in_file=str(in_file),
            out_file="split.xyz",
            split_mode="stratified_random",
            test_fraction=0.25,
            seed=7,
        )
        assert Path("train_split.xyz").exists()
        assert Path("test_split.xyz").exists()
        assert len(read("train_split.xyz", ":")) == 15
        assert len(read("test_split.xyz", ":")) == 5
    finally:
        os.chdir(cwd)

    with pytest.raises(ValueError, match="split_mode"):
        split_configset(
            in_file=str(in_file),
            out_file={
                "train": str(tmp_path / "t.xyz"),
                "test": str(tmp_path / "v.xyz"),
            },
            split_mode="force_agreement",
        )

    with pytest.raises(ValueError, match="test_fraction"):
        split_configset(
            in_file=str(in_file),
            out_file={
                "train": str(tmp_path / "t.xyz"),
                "test": str(tmp_path / "v.xyz"),
            },
            test_fraction=1.0,
        )


def test_selection_module_has_no_gmm_dependency():
    """Verify mlipflow.data.selection does not import TorchGMM or mlipflow.data.gmm."""
    source = inspect.getsource(selection_mod)
    assert "TorchGMM" not in source
    assert "mlipflow.data.gmm" not in source


def test_select_by_uncertainty_real_mace_redundancy_suppression(tmp_path):
    """End-to-end Sequential Batch LLPR selection on real mace_test.model + Cu(100) slab data."""
    test_dir = Path(__file__).resolve().parent
    model_path = test_dir / "data" / "mace_test.model"
    src_xyz = test_dir / "data" / "test_data.xyz"

    all_frames = read(src_xyz, ":")
    atom_slice = slice(48, None)

    mace_calc = MACECalculator(
        model_paths=str(model_path), device="cpu", default_dtype="float32"
    )

    train_frames = all_frames[:12]
    val_frames = [at.copy() for at in all_frames[12:]]
    rng = np.random.default_rng(7)
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
    )

    uq_file = tmp_path / "mace_test_llpr.pt"
    llpr.save(uq_file)
    orig_pt_bytes = uq_file.read_bytes()

    # Build pool: 3 equilibrium train frames + 4 exact copies of rattled OOD A + 4 exact copies of rattled OOD B
    eq_frames = [at.copy() for at in train_frames[:3]]
    for at in eq_frames:
        at.info["cluster"] = "eq"

    ood_a = train_frames[0].copy()
    pos_a = ood_a.get_positions()
    pos_a[48:] += np.random.default_rng(101).normal(
        0.0, 0.25, size=(len(ood_a) - 48, 3)
    )
    ood_a.set_positions(pos_a)
    ood_a.info["cluster"] = "A"

    ood_b = train_frames[1].copy()
    pos_b = ood_b.get_positions()
    pos_b[48:] += np.random.default_rng(202).normal(
        0.0, 0.25, size=(len(ood_b) - 48, 3)
    )
    ood_b.set_positions(pos_b)
    ood_b.info["cluster"] = "B"

    pool = (
        eq_frames + [ood_a.copy() for _ in range(4)] + [ood_b.copy() for _ in range(4)]
    )
    pool_xyz = tmp_path / "pool.xyz"
    write(pool_xyz, pool)

    # Pick tolerance cleanly between equilibrium max and rattled OOD min
    _, eq_scores = llpr.score_configs(
        eq_frames, mace_calc, atom_indices=atom_slice, exclude_fixed=False, delta=0.50
    )
    _, ood_scores = llpr.score_configs(
        [ood_a, ood_b],
        mace_calc,
        atom_indices=atom_slice,
        exclude_fixed=False,
        delta=0.50,
    )
    assert float(np.min(ood_scores)) > float(np.max(eq_scores))
    tol = 0.5 * (float(np.max(eq_scores)) + float(np.min(ood_scores)))

    mace_model = MACEModel(
        mlip_name=str(test_dir / "data" / "mace_test"),
        run_mode="local",
        uq_file=str(uq_file),
    )
    out_file = tmp_path / "selected.xyz"

    selected = select_by_uncertainty(
        pool_file=str(pool_xyz),
        out_file=str(out_file),
        mlip_strategy=mace_model,
        uq_file=str(uq_file),
        force_tol_eVA=tol,
        atom_indices=atom_slice,
        exclude_fixed=False,
        delta=0.50,
        device="cpu",
    )

    assert out_file.exists()
    selected_on_disk = read(out_file, ":")
    assert len(selected) == len(selected_on_disk)

    # Duplicate copies of A and B must be collapsed (at most 1 of A and 1 of B)
    clusters = [at.info["cluster"] for at in selected_on_disk]
    assert clusters.count("A") == 1
    assert clusters.count("B") == 1
    assert "eq" not in clusters

    # Inline annotations must be written on selected frames with inactive atoms zeroed
    for at in selected_on_disk:
        assert "MACE_max_uq" in at.info
        assert float(at.info["MACE_max_uq"]) > tol
        assert "MACE_atom_uq" in at.arrays
        assert np.all(at.arrays["MACE_atom_uq"][:48] == 0.0)
        assert np.all(at.arrays["MACE_atom_uq"][48:] > 0.0)

    # Saved .pt artifact must remain untouched
    assert uq_file.read_bytes() == orig_pt_bytes


def _build_synthetic_llpr_fixture(tmp_path):
    """Create a fitted/calibrated PerElementLLPR .pt artifact and matching spy calculator."""
    D = 6

    class _SpyCalc:
        def __init__(self):
            self.descriptor_calls = 0

        def get_descriptors(
            self, atoms: Atoms, invariants_only: bool = True
        ) -> np.ndarray:
            _ = invariants_only
            self.descriptor_calls += 1
            return np.asarray(atoms.arrays["fake_desc"], dtype=np.float64)

    def _make_at(descs: np.ndarray, tag: str, err: float | None = None) -> Atoms:
        at = Atoms("CuO", positions=[[0, 0, 0], [0, 0, 1.5]])
        at.arrays["fake_desc"] = np.asarray(descs, dtype=np.float64)
        at.info["tag"] = tag
        if err is not None:
            at.arrays["MACE_forces"] = np.zeros((2, 3), dtype=np.float64)
            at.arrays["DFT_forces"] = np.array([[err, 0.0, 0.0], [err, 0.0, 0.0]])
        return at

    spy = _SpyCalc()
    # Balanced orthonormal basis spanning dims 0 and 1 (exact equal variance G[0,0] == G[1,1])
    basis = np.eye(D, dtype=np.float64)
    train_ats = [
        _make_at(np.vstack([basis[i % 2], basis[i % 2]]), "train") for i in range(20)
    ]
    val_ats = [
        _make_at(np.vstack([basis[i % 2], basis[i % 2]]), "val", err=0.02)
        for i in range(20)
    ]

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_ats, spy)
    llpr.calibrate(val_ats, spy, tune_lambdas=(1e-4,), deltas=(0.50, 0.20, 0.10))
    uq_file = tmp_path / "synth_llpr.pt"
    llpr.save(uq_file)
    spy.descriptor_calls = 0
    return uq_file, spy, _make_at, D


def test_select_by_uncertainty_inline_uq_prefilter_and_provenance(tmp_path, caplog):
    """Verify default use_inline_uq=False, use_inline_uq=True pre-filter, and provenance mismatch fallback."""
    uq_file, spy, _make_at, D = _build_synthetic_llpr_fixture(tmp_path)
    mace_model = MACEModel(
        mlip_name=str(tmp_path / "synth"), run_mode="local", uq_file=str(uq_file)
    )
    Path(mace_model.model_file).write_text("placeholder", encoding="utf-8")

    in_d = np.zeros((2, D), dtype=np.float64)
    in_d[:, 0] = 0.5
    ood_d = np.zeros((2, D), dtype=np.float64)
    ood_d[:, 2] = 2.0

    pool = [
        _make_at(in_d, "eq0"),
        _make_at(in_d, "eq1"),
        _make_at(in_d, "eq2"),
        _make_at(ood_d, "ood0"),
        _make_at(ood_d, "ood1"),
    ]
    for at in pool[:3]:
        at.info["MACE_max_uq"] = 0.01
        at.info["MACE_uq_delta"] = 0.50
        at.info["MACE_uq_model"] = uq_file.name
    for at in pool[3:]:
        at.info["MACE_max_uq"] = 0.90
        at.info["MACE_uq_delta"] = 0.50
        at.info["MACE_uq_model"] = uq_file.name

    with patch("mlipflow.data.selection.MACECalculator", return_value=spy):
        # 1. use_inline_uq=True with matching provenance skips below-tol frames
        spy.descriptor_calls = 0
        sel_fast = select_by_uncertainty(
            pool_file=pool,
            out_file=tmp_path / "fast.xyz",
            mlip_strategy=mace_model,
            uq_file=uq_file,
            force_tol_eVA=0.10,
            delta=0.50,
            use_inline_uq=True,
        )
        assert spy.descriptor_calls == 2
        assert [at.info["tag"] for at in sel_fast] == ["ood0"]

        # 2. Default use_inline_uq=False always scores full pool freshly
        spy.descriptor_calls = 0
        sel_default = select_by_uncertainty(
            pool_file=pool,
            out_file=tmp_path / "default.xyz",
            mlip_strategy=mace_model,
            uq_file=uq_file,
            force_tol_eVA=0.10,
            delta=0.50,
        )
        assert spy.descriptor_calls == 5
        assert [at.info["tag"] for at in sel_default] == ["ood0"]

        # 3. Provenance mismatch under use_inline_uq=True logs warning & falls back to full pool
        pool[0].info["MACE_uq_delta"] = 0.10  # mismatches requested delta=0.50
        spy.descriptor_calls = 0
        caplog.clear()
        with caplog.at_level("WARNING"):
            sel_mismatch = select_by_uncertainty(
                pool_file=pool,
                out_file=tmp_path / "mismatch.xyz",
                mlip_strategy=mace_model,
                uq_file=uq_file,
                force_tol_eVA=0.10,
                delta=0.50,
                use_inline_uq=True,
            )
        assert spy.descriptor_calls == 5
        assert [at.info["tag"] for at in sel_mismatch] == ["ood0"]
        assert any(
            "provenance" in rec.message.lower() or "mismatch" in rec.message.lower()
            for rec in caplog.records
        )


def test_select_by_uncertainty_zero_passing_and_fallback_matrix(tmp_path, caplog):
    """Verify n_passing == 0 empty return, .pt element_metrics bootstrap, warning on missing metrics, and greedy top-K fallback."""
    uq_file, spy, _make_at, D = _build_synthetic_llpr_fixture(tmp_path)
    mace_model = MACEModel(
        mlip_name=str(tmp_path / "synth"), run_mode="local", uq_file=str(uq_file)
    )
    Path(mace_model.model_file).write_text("placeholder", encoding="utf-8")

    # All frames are in-distribution (well below force_tol_eVA=5.0)
    d_a = np.zeros((2, D), dtype=np.float64)
    d_a[:, 0] = 0.8
    d_b = np.zeros((2, D), dtype=np.float64)
    d_b[:, 1] = 0.79
    pool = [
        _make_at(d_a, "a0"),
        _make_at(d_a, "a1"),  # duplicate of a0
        _make_at(d_b, "b0"),  # orthogonal to a0
    ]

    with patch("mlipflow.data.selection.MACECalculator", return_value=spy):
        # 1. min_select_count=0 -> empty 0-byte file
        empty_out = tmp_path / "empty.xyz"
        res_empty = select_by_uncertainty(
            pool_file=pool,
            out_file=empty_out,
            mlip_strategy=mace_model,
            uq_file=uq_file,
            force_tol_eVA=5.0,
            min_select_count=0,
        )
        assert res_empty == []
        assert empty_out.exists()
        assert empty_out.stat().st_size == 0

        # 2. min_select_count=2 but element_metrics all <= target -> empty
        res_converged = select_by_uncertainty(
            pool_file=pool,
            out_file=tmp_path / "converged.xyz",
            mlip_strategy=mace_model,
            uq_file=uq_file,
            force_tol_eVA=5.0,
            min_select_count=2,
            element_metrics={29: {"p90": 0.04}, 8: {"p90": 0.04}},
            element_error_target=0.05,
        )
        assert res_converged == []

        # 3. min_select_count=2 with explicit element_metrics exceeding target -> greedy diverse [a0, b0]
        res_fallback = select_by_uncertainty(
            pool_file=pool,
            out_file=tmp_path / "fallback.xyz",
            mlip_strategy=mace_model,
            uq_file=uq_file,
            force_tol_eVA=5.0,
            min_select_count=2,
            element_metrics={29: {"p90": 0.02}, 8: {"p90": 0.08}},
            element_error_target=0.05,
            element_error_metric="p90",
        )
        assert [at.info["tag"] for at in res_fallback] == ["a0", "b0"]

        # 4. Bootstrapped standalone call (element_metrics=None) reads element_metrics from .pt (p90 == 0.02 > 0.01)
        res_bootstrapped = select_by_uncertainty(
            pool_file=pool,
            out_file=tmp_path / "bootstrapped.xyz",
            mlip_strategy=mace_model,
            uq_file=uq_file,
            force_tol_eVA=5.0,
            min_select_count=2,
            element_metrics=None,
            element_error_target=0.01,
        )
        assert [at.info["tag"] for at in res_bootstrapped] == ["a0", "b0"]

        # 5. Legacy .pt without element_metrics + element_metrics=None -> warns & returns empty
        legacy_payload = torch.load(str(uq_file), map_location="cpu", weights_only=True)
        legacy_payload.pop("element_metrics", None)
        legacy_uq = tmp_path / "legacy_llpr.pt"
        torch.save(legacy_payload, str(legacy_uq))

        caplog.clear()
        with caplog.at_level("WARNING"):
            res_missing_metrics = select_by_uncertainty(
                pool_file=pool,
                out_file=tmp_path / "missing_metrics.xyz",
                mlip_strategy=mace_model,
                uq_file=legacy_uq,
                force_tol_eVA=5.0,
                min_select_count=2,
                element_metrics=None,
                element_error_target=0.01,
            )
        assert res_missing_metrics == []
        assert any("element_metrics" in rec.message for rec in caplog.records)
