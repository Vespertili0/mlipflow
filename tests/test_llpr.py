from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from mace.calculators import MACECalculator

from mlipflow.data import PerElementLLPR as ReExportedPerElementLLPR
from mlipflow.data.llpr import PerElementLLPR


class _FakeMACECalculator:
    """Lightweight deterministic calculator returning pre-stored per-atom descriptors."""

    def __init__(self, feature_dim: int = 8):
        self.feature_dim = feature_dim

    def get_descriptors(self, atoms: Atoms, invariants_only: bool = True) -> np.ndarray:
        _ = invariants_only
        if "fake_desc" in atoms.arrays:
            return np.asarray(atoms.arrays["fake_desc"], dtype=np.float64)
        # Deterministic fallback from positions & atomic numbers
        pos = atoms.get_positions()
        nums = atoms.get_atomic_numbers()[:, None]
        base = np.hstack([pos, nums, pos**2])
        if base.shape[1] < self.feature_dim:
            pad = np.zeros((len(atoms), self.feature_dim - base.shape[1]))
            base = np.hstack([base, pad])
        return base[:, : self.feature_dim].astype(np.float64)


def _make_synthetic_atoms(
    symbols: list[str],
    descs: np.ndarray,
    err_norms: np.ndarray | None = None,
    fixed_indices: list[int] | None = None,
) -> Atoms:
    n = len(symbols)
    at = Atoms(symbols=symbols, positions=np.zeros((n, 3)))
    at.arrays["fake_desc"] = np.asarray(descs, dtype=np.float64)
    if fixed_indices:
        at.set_constraint(FixAtoms(indices=fixed_indices))
    if err_norms is not None:
        f_pred = np.zeros((n, 3), dtype=np.float64)
        f_ref = np.zeros((n, 3), dtype=np.float64)
        f_ref[:, 0] = np.asarray(err_norms, dtype=np.float64)
        at.arrays["MACE_forces"] = f_pred
        at.arrays["DFT_forces"] = f_ref
    return at


def test_reexport_from_mlipflow_data():
    """Verify PerElementLLPR is re-exported from mlipflow.data."""
    assert ReExportedPerElementLLPR is PerElementLLPR


def test_llpr_fit_score_determinism_and_constraints():
    """Verify float64 Gram/Cholesky fit, strict determinism, and FixAtoms/atom_indices exclusion."""
    rng = np.random.default_rng(7)
    D = 8
    calc = _FakeMACECalculator(feature_dim=D)

    train_configs = []
    for _ in range(15):
        # 2 fixed Cu + 2 active Cu + 2 active O
        descs = rng.normal(0.0, 1.0, size=(6, D))
        at = _make_synthetic_atoms(
            ["Cu", "Cu", "Cu", "Cu", "O", "O"], descs, fixed_indices=[0, 1]
        )
        train_configs.append(at)

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_configs, calc)

    assert llpr.feature_dim == D
    assert llpr.atom_counts == {29: 30, 8: 30}
    assert llpr.gram_matrices[29].dtype == torch.float64
    assert llpr.cholesky_factors[8].dtype == torch.float64

    test_at = _make_synthetic_atoms(
        ["Cu", "Cu", "Cu", "Cu", "O", "O"],
        rng.normal(0.0, 1.0, size=(6, D)),
        fixed_indices=[0, 1],
    )

    atom_sc_1, struct_sc_1 = llpr.score_atoms(test_at, calc, calibrated=False)
    atom_sc_2, struct_sc_2 = llpr.score_atoms(test_at, calc, calibrated=False)

    np.testing.assert_array_equal(atom_sc_1, atom_sc_2)
    assert struct_sc_1 == struct_sc_2

    # Fixed atoms must have score 0.0; active atoms must be strictly positive & finite
    assert np.all(atom_sc_1[:2] == 0.0)
    assert np.all(atom_sc_1[2:] > 0.0)
    assert np.all(np.isfinite(atom_sc_1[2:]))
    assert struct_sc_1 == pytest.approx(float(np.max(atom_sc_1[2:])))

    # Explicit atom_indices slice restricts scoring further
    sliced_scores, sliced_struct = llpr.score_atoms(
        test_at, calc, atom_indices=slice(4, None), calibrated=False
    )
    assert np.all(sliced_scores[:4] == 0.0)
    assert np.all(sliced_scores[4:] > 0.0)
    assert sliced_struct == pytest.approx(float(np.max(sliced_scores[4:])))


def test_llpr_unseen_element_and_calibration_fallback():
    """Verify unseen Z returns inf and uncalibrated-in-val Z falls back to global q_atom."""
    rng = np.random.default_rng(11)
    D = 6
    calc = _FakeMACECalculator(feature_dim=D)

    # Train contains Cu (29) and O (8)
    train_configs = [
        _make_synthetic_atoms(["Cu", "O"], rng.normal(size=(2, D))) for _ in range(12)
    ]
    # Val contains ONLY Cu (29) -> O (8) must fall back to global q_atom
    val_configs = [
        _make_synthetic_atoms(
            ["Cu", "Cu"], rng.normal(size=(2, D)), err_norms=np.array([0.05, 0.08])
        )
        for _ in range(10)
    ]

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_configs, calc)
    llpr.calibrate(val_configs, calc, tune_lambdas=(1e-4,))

    assert 29 in llpr.calibrations[0.20]["q_atom_by_z"]
    assert 8 not in llpr.calibrations[0.20]["q_atom_by_z"]

    # Score frame containing seen-in-fit-only O (8) and completely unseen H (1)
    probe = _make_synthetic_atoms(["Cu", "O", "H"], rng.normal(size=(3, D)))
    raw_scores, raw_struct = llpr.score_atoms(probe, calc, calibrated=False)
    assert np.isfinite(raw_scores[0])
    assert np.isfinite(raw_scores[1])
    assert np.isinf(raw_scores[2])
    assert np.isinf(raw_struct)

    cal_scores, cal_struct = llpr.score_atoms(probe, calc, calibrated=True, delta=0.20)
    q_global = llpr.calibrations[0.20]["q_atom"]
    assert cal_scores[1] == pytest.approx(q_global * raw_scores[1])
    assert np.isinf(cal_scores[2])
    assert np.isinf(cal_struct)


def test_llpr_per_element_calibration_applied_before_max_pooling():
    """Prove q_atom_by_z is applied BEFORE worst-atom max-pooling across elements."""
    D = 4
    calc = _FakeMACECalculator(feature_dim=D)

    # Identical Gram basis for both Cu and O so raw scores depend solely on feature norm
    basis = np.eye(D, dtype=np.float64)
    train_configs = [
        _make_synthetic_atoms(["Cu", "O"], np.vstack([basis[i % D], basis[i % D]]))
        for i in range(16)
    ]

    # Validation: Cu has tiny error (0.01), O has huge error (10.0) at identical unit descriptors
    val_configs = [
        _make_synthetic_atoms(
            ["Cu", "O"],
            np.vstack([basis[i % D], basis[i % D]]),
            err_norms=np.array([0.01, 10.0]),
        )
        for i in range(20)
    ]

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_configs, calc)
    llpr.calibrate(val_configs, calc, tune_lambdas=(1e-4,), deltas=(0.50, 0.20, 0.10))

    # Probe where Cu has LARGER raw leverage (2.0 * e0) than O (0.5 * e0)
    probe_desc = np.vstack([2.0 * basis[0], 0.5 * basis[0]])
    probe = _make_synthetic_atoms(["Cu", "O"], probe_desc)

    raw_scores, raw_max = llpr.score_atoms(probe, calc, calibrated=False)
    assert raw_scores[0] > raw_scores[1]
    assert raw_max == pytest.approx(raw_scores[0])

    # After per-element calibration, O dominates Cu BEFORE max-pooling
    cal_scores, struct_score = llpr.score_atoms(
        probe, calc, calibrated=True, delta=0.10
    )
    assert cal_scores[1] > cal_scores[0]
    q_struct_10 = llpr.calibrations[0.10]["q_struct"]
    assert struct_score == pytest.approx(q_struct_10 * cal_scores[1])


def test_llpr_dual_conformal_coverage_and_diagnostics():
    """Verify empirical coverage matches 1 - delta for delta=0.50, 0.20, and 0.10 + diagnostics."""
    rng = np.random.default_rng(2026)
    D = 12
    calc = _FakeMACECalculator(feature_dim=D)

    # Low-rank training manifold (strong variance along first 4 dims, tiny along rest)
    def sample_desc(n_atoms: int, ood_scale: float = 0.05) -> np.ndarray:
        x = np.zeros((n_atoms, D), dtype=np.float64)
        x[:, :4] = rng.normal(0.0, 1.0, size=(n_atoms, 4))
        x[:, 4:] = rng.normal(0.0, ood_scale, size=(n_atoms, D - 4))
        return x

    symbols = ["Cu", "Cu", "C", "O", "H"]
    elem_scale = {"Cu": 0.05, "C": 0.20, "O": 0.40, "H": 0.15}

    train_configs = [
        _make_synthetic_atoms(symbols, sample_desc(len(symbols), ood_scale=0.05))
        for _ in range(40)
    ]

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_configs, calc)

    # Generate i.i.d. calibration (200) and independent held-out test (400) sets
    # where force errors correlate with raw LLPR leverage + multiplicative noise
    def build_eval_set(n_configs: int) -> list[Atoms]:
        configs = []
        for _ in range(n_configs):
            ood = float(rng.uniform(0.03, 0.25))
            d = sample_desc(len(symbols), ood_scale=ood)
            tmp_at = _make_synthetic_atoms(symbols, d)
            raw, _ = llpr.score_atoms(tmp_at, calc, calibrated=False)
            noise = rng.lognormal(mean=0.0, sigma=0.35, size=len(symbols))
            scales = np.array([elem_scale[s] for s in symbols])
            errs = scales * raw * noise
            configs.append(_make_synthetic_atoms(symbols, d, err_norms=errs))
        return configs

    val_configs = build_eval_set(200)
    test_configs = build_eval_set(400)

    diag = llpr.calibrate(val_configs, calc, deltas=(0.50, 0.20, 0.10))

    # Verify bounded lambda grid selection & structured diagnostic schema
    assert diag["best_lambda"] in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    assert set(diag["per_element"].keys()) == {1, 6, 8, 29}
    for z, stats in diag["per_element"].items():
        assert stats["n_atoms"] > 0
        assert stats["mae"] > 0.0
        assert stats["p90"] >= stats["mae"]
        assert stats["max_err"] >= stats["p90"]
        assert z in diag["spearman_by_z"]
        assert set(stats["q_atom"].keys()) == {0.50, 0.20, 0.10}

    # Verify finite-sample calibration coverage >= 1 - delta on val set
    for delta in (0.50, 0.20, 0.10):
        assert diag["coverage"][delta]["atom"] >= (1.0 - delta) - 1e-9
        assert diag["coverage"][delta]["struct"] >= (1.0 - delta) - 1e-9

    # Verify empirical coverage on independent held-out test set matches 1 - delta
    true_atom_errs = np.concatenate(
        [
            np.linalg.norm(at.arrays["DFT_forces"] - at.arrays["MACE_forces"], axis=1)
            for at in test_configs
        ]
    )
    true_struct_errs = np.array(
        [
            float(
                np.max(
                    np.linalg.norm(
                        at.arrays["DFT_forces"] - at.arrays["MACE_forces"], axis=1
                    )
                )
            )
            for at in test_configs
        ]
    )

    struct_bounds_by_delta = {}
    for delta in (0.50, 0.20, 0.10):
        atom_bounds_list, struct_bounds = llpr.score_configs(
            test_configs, calc, calibrated=True, delta=delta
        )
        atom_bounds = np.concatenate(atom_bounds_list)
        atom_cov = float(np.mean(true_atom_errs <= atom_bounds))
        struct_cov = float(np.mean(true_struct_errs <= struct_bounds))

        target = 1.0 - delta
        assert abs(atom_cov - target) <= 0.08
        assert abs(struct_cov - target) <= 0.08
        struct_bounds_by_delta[delta] = struct_bounds

    # Monotonicity across coverage levels: 90% >= 80% >= 50%
    assert np.all(struct_bounds_by_delta[0.10] >= struct_bounds_by_delta[0.20] - 1e-12)
    assert np.all(struct_bounds_by_delta[0.20] >= struct_bounds_by_delta[0.50] - 1e-12)

    # Default calibrated scoring uses default_delta=0.20
    _, default_struct_bounds = llpr.score_configs(test_configs, calc, calibrated=True)
    np.testing.assert_allclose(default_struct_bounds, struct_bounds_by_delta[0.20])


def test_llpr_save_and_load_weights_only(tmp_path):
    """Verify pure tensor/primitive serialization round-trip with torch.load(weights_only=True)."""
    rng = np.random.default_rng(99)
    D = 8
    calc = _FakeMACECalculator(feature_dim=D)

    train_configs = [
        _make_synthetic_atoms(["Cu", "O", "H"], rng.normal(size=(3, D)))
        for _ in range(15)
    ]
    val_configs = [
        _make_synthetic_atoms(
            ["Cu", "O", "H"],
            rng.normal(size=(3, D)),
            err_norms=rng.uniform(0.01, 0.2, size=3),
        )
        for _ in range(20)
    ]

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_configs, calc)
    llpr.calibrate(val_configs, calc, deltas=(0.50, 0.20, 0.10))

    uq_path = tmp_path / "model_v0_llpr.pt"
    llpr.save(uq_path)
    assert uq_path.exists()

    # Explicitly verify raw torch.load(..., weights_only=True) succeeds
    raw_payload = torch.load(str(uq_path), map_location="cpu", weights_only=True)
    assert isinstance(raw_payload, dict)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    probe = _make_synthetic_atoms(["Cu", "O", "H"], rng.normal(size=(3, D)))
    for dev in devices:
        reloaded = PerElementLLPR.load(uq_path, device=dev)
        assert reloaded.lambda_reg == llpr.lambda_reg
        assert reloaded.default_delta == llpr.default_delta
        assert set(reloaded.calibrations.keys()) == {0.50, 0.20, 0.10}

        for calibrated, delta in [
            (False, None),
            (True, 0.50),
            (True, 0.20),
            (True, 0.10),
        ]:
            a1, s1 = llpr.score_atoms(probe, calc, calibrated=calibrated, delta=delta)
            a2, s2 = reloaded.score_atoms(
                probe, calc, calibrated=calibrated, delta=delta
            )
            np.testing.assert_allclose(a1, a2, rtol=1e-12, atol=1e-12)
            assert s1 == pytest.approx(s2, rel=1e-12, abs=1e-12)


def test_llpr_validation_errors():
    """Verify clear error handling for unfitted/empty/invalid inputs."""
    calc = _FakeMACECalculator(feature_dim=4)
    llpr = PerElementLLPR(device="cpu")

    with pytest.raises(ValueError, match="train_configs"):
        llpr.fit([], calc)

    probe = _make_synthetic_atoms(["H"], np.ones((1, 4)))
    with pytest.raises(RuntimeError, match="fit"):
        llpr.score_atoms(probe, calc)

    llpr.fit([probe], calc)
    with pytest.raises(ValueError, match="val_configs"):
        llpr.calibrate([], calc)

    with pytest.raises(KeyError, match="DFT_forces"):
        llpr.calibrate([probe], calc)

    with pytest.raises(KeyError, match="delta"):
        llpr.get_calibration(0.33)


def test_llpr_real_mace_model_and_slab_data(tmp_path):
    """End-to-end integration test with real mace_test.model and Cu(100) test_data.xyz."""
    data_dir = Path(__file__).resolve().parent / "data"
    model_path = data_dir / "mace_test.model"
    xyz_path = data_dir / "test_data.xyz"

    all_frames = read(xyz_path, ":")
    fixed_indices = list(range(48))
    for at in all_frames:
        at.set_constraint(FixAtoms(indices=fixed_indices))

    mace_calc = MACECalculator(
        model_paths=str(model_path), device="cpu", default_dtype="float32"
    )

    train_frames = all_frames[:10]
    val_frames = [at.copy() for at in all_frames[10:]]

    # Attach real MACE_forces + realistic synthetic DFT_forces residuals on val_frames
    rng = np.random.default_rng(123)
    for at in val_frames:
        at_copy = at.copy()
        at_copy.calc = mace_calc
        f_mace = at_copy.get_forces()
        at.arrays["MACE_forces"] = f_mace
        at.arrays["DFT_forces"] = f_mace + rng.normal(0.0, 0.05, size=f_mace.shape)

    train_xyz = tmp_path / "train.xyz"
    val_xyz = tmp_path / "val.xyz"
    write(train_xyz, train_frames)
    write(val_xyz, val_frames)

    # Fit & calibrate directly from file paths (ConfigSet-native API)
    llpr = PerElementLLPR(invariants_only=True, device="cpu")
    llpr.fit(train_xyz, mace_calc)
    assert set(llpr.atom_counts.keys()) == {1, 6, 8, 29}

    diag = llpr.calibrate(val_xyz, mace_calc, deltas=(0.50, 0.20, 0.10))
    assert set(diag["coverage"].keys()) == {0.50, 0.20, 0.10}

    # Save & reload with weights_only=True
    uq_file = tmp_path / "mace_test_llpr.pt"
    llpr.save(uq_file)
    reloaded = PerElementLLPR.load(uq_file, device="cpu")

    train_atom_scores, train_struct_scores = reloaded.score_configs(
        train_frames, mace_calc, calibrated=True, delta=0.20
    )
    for scores in train_atom_scores:
        assert np.all(scores[:48] == 0.0)
        assert np.all(scores[48:] > 0.0)
        assert np.all(np.isfinite(scores[48:]))

    # Rattled OOD (+0.15 A on active atoms) should shift UQ upward relative to train
    rattled_frames = []
    for at in train_frames[:5]:
        at_ood = at.copy()
        pos = at_ood.get_positions()
        pos[48:] += rng.normal(0.0, 0.15, size=(len(at_ood) - 48, 3))
        at_ood.set_positions(pos)
        rattled_frames.append(at_ood)

    _, rattled_struct_scores = reloaded.score_configs(
        rattled_frames, mace_calc, calibrated=True, delta=0.20
    )
    assert float(np.median(rattled_struct_scores)) > float(
        np.median(train_struct_scores)
    )


class _ConstantForceCalculator:
    """Minimal ASE-like calculator returning deterministic forces for testing."""

    def __init__(self, forces_map: dict[int, np.ndarray]):
        self.forces_map = forces_map
        self.call_count = 0

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        self.call_count += 1
        tag = int(atoms.info["frame_id"])
        return self.forces_map[tag]


def test_llpr_calibrate_pred_forces_key_none_ignores_stale_forces(tmp_path):
    """Verify pred_forces_key=None forces fresh force_calc evaluation over stale MACE_forces."""
    rng = np.random.default_rng(314)
    D = 6
    desc_calc = _FakeMACECalculator(feature_dim=D)

    train_configs = [
        _make_synthetic_atoms(["Cu", "O"], rng.normal(size=(2, D))) for _ in range(12)
    ]
    val_configs = []
    fresh_forces: dict[int, np.ndarray] = {}
    for i in range(10):
        at = _make_synthetic_atoms(["Cu", "O"], rng.normal(size=(2, D)))
        at.info["frame_id"] = i
        f_dft = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        # Stale MACE_forces -> huge bogus error (10.0 eV/A)
        at.arrays["MACE_forces"] = np.zeros((2, 3), dtype=np.float64)
        at.arrays["DFT_forces"] = f_dft
        # Fresh v_k forces -> small true error (0.05 eV/A)
        fresh_forces[i] = f_dft - np.array(
            [[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64
        )
        val_configs.append(at)

    fresh_calc = _ConstantForceCalculator(fresh_forces)

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_configs, desc_calc)
    diag = llpr.calibrate(
        val_configs,
        desc_calc,
        force_calc=fresh_calc,
        pred_forces_key=None,
        tune_lambdas=(1e-4,),
        deltas=(0.50, 0.20, 0.10),
    )

    assert fresh_calc.call_count == len(val_configs)
    np.testing.assert_allclose(diag["val_atom_errors"], 0.05, rtol=1e-12, atol=1e-12)
    assert llpr.element_metrics == diag["per_element"]
    assert set(llpr.element_metrics.keys()) == {8, 29}
    assert llpr.element_metrics[8]["mae"] == pytest.approx(0.05)

    # Save & reload preserves element_metrics with weights_only=True
    uq_path = tmp_path / "metrics_llpr.pt"
    llpr.save(uq_path)
    raw = torch.load(str(uq_path), map_location="cpu", weights_only=True)
    assert "element_metrics" in raw

    reloaded = PerElementLLPR.load(uq_path, device="cpu")
    assert reloaded.element_metrics[8]["mae"] == pytest.approx(0.05)
    assert reloaded.element_metrics[29]["p90"] == pytest.approx(0.05)

    # Legacy payload without element_metrics still loads cleanly
    raw.pop("element_metrics")
    legacy_path = tmp_path / "legacy_llpr.pt"
    torch.save(raw, str(legacy_path))
    reloaded_legacy = PerElementLLPR.load(legacy_path, device="cpu")
    assert reloaded_legacy.element_metrics == {}


def test_llpr_select_sequential_batch_suppresses_duplicates_and_preserves_state():
    """Verify greedy Gram/Cholesky updates suppress duplicate OOD clusters and leave llpr immutable."""
    D = 6
    calc = _FakeMACECalculator(feature_dim=D)

    # Training spans dims 0 and 1 cleanly for Cu (29) and O (8)
    rng = np.random.default_rng(7)
    train_configs = []
    for _ in range(30):
        descs = np.zeros((2, D), dtype=np.float64)
        descs[:, :2] = rng.normal(0.0, 1.0, size=(2, 2))
        train_configs.append(_make_synthetic_atoms(["Cu", "O"], descs))

    # Calibrate with small in-distribution errors (~0.02 eV/A)
    val_configs = []
    for _ in range(20):
        descs = np.zeros((2, D), dtype=np.float64)
        descs[:, :2] = rng.normal(0.0, 1.0, size=(2, 2))
        val_configs.append(
            _make_synthetic_atoms(["Cu", "O"], descs, err_norms=np.array([0.02, 0.02]))
        )

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_configs, calc)
    llpr.calibrate(val_configs, calc, tune_lambdas=(1e-4,), deltas=(0.50, 0.20, 0.10))

    # Snapshot fitted state to assert strict immutability
    orig_grams = {z: g.clone() for z, g in llpr.gram_matrices.items()}
    orig_chols = {z: c.clone() for z, c in llpr.cholesky_factors.items()}

    # Build pool:
    # 0: in-distribution (below tol)
    # 1, 2, 3: 3 identical copies of OOD direction A (dim 2)
    # 4, 5: 2 identical copies of OOD direction B (dim 3, orthogonal to A)
    def _vec(dim_idx: int, mag: float = 2.0) -> np.ndarray:
        arr = np.zeros((2, D), dtype=np.float64)
        arr[:, dim_idx] = mag
        return arr

    in_dist = np.zeros((2, D), dtype=np.float64)
    in_dist[:, 0] = 0.5
    pool_descs = [
        in_dist,
        _vec(2, 2.0),
        _vec(2, 2.0),
        _vec(2, 2.0),
        _vec(3, 1.8),
        _vec(3, 1.8),
    ]
    pool_nums = [np.array([29, 8], dtype=int) for _ in pool_descs]

    selected = llpr.select_sequential_batch(
        candidate_descs=pool_descs,
        candidate_numbers=pool_nums,
        force_tol_eVA=0.10,
        delta=0.50,
    )

    # Exactly 1 representative from cluster A (idx 1) and 1 from cluster B (idx 4)
    assert selected == [1, 4]

    # Fitted LLPR state must be completely untouched
    for z, orig_g in orig_grams.items():
        assert torch.equal(llpr.gram_matrices[z], orig_g)
        assert torch.equal(llpr.cholesky_factors[z], orig_chols[z])


def test_llpr_select_sequential_batch_unseen_element_and_top_k_fallback():
    """Verify unseen-element inf handling and greedy diverse top-K fallback."""
    D = 4
    calc = _FakeMACECalculator(feature_dim=D)
    eye = np.eye(D, dtype=np.float64)

    # Trained ONLY on Cu (29)
    train_configs = [
        _make_synthetic_atoms(["Cu"], eye[i % D : i % D + 1]) for i in range(16)
    ]
    val_configs = [
        _make_synthetic_atoms(
            ["Cu"], eye[i % D : i % D + 1], err_norms=np.array([0.02])
        )
        for i in range(16)
    ]

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_configs, calc)
    llpr.calibrate(val_configs, calc, tune_lambdas=(1e-4,), deltas=(0.50, 0.20, 0.10))

    # Pool with duplicate unseen-element H (Z=1) environments along e0 + orthogonal H along e1
    pool_descs = [
        eye[0:1],  # idx 0: Cu in-dist
        eye[0:1],  # idx 1: unseen H along e0
        eye[0:1],  # idx 2: duplicate unseen H along e0
        eye[1:2],  # idx 3: orthogonal unseen H along e1
    ]
    pool_nums = [np.array([29]), np.array([1]), np.array([1]), np.array([1])]

    selected = llpr.select_sequential_batch(
        candidate_descs=pool_descs,
        candidate_numbers=pool_nums,
        force_tol_eVA=0.10,
        delta=0.50,
    )
    # idx 1 (H along e0) picked first (inf), idx 2 suppressed, idx 3 (orthogonal H along e1) picked
    assert selected == [1, 3]

    # Top-K fallback mode on all-below-threshold pool picks diverse directions rather than duplicates
    fallback_descs = [
        0.5 * eye[0:1],  # idx 0: cluster e0
        0.5 * eye[0:1],  # idx 1: duplicate cluster e0
        0.49 * eye[1:2],  # idx 2: cluster e1 (slightly lower initial score than idx 1)
    ]
    fallback_nums = [np.array([29]) for _ in fallback_descs]
    fallback_selected = llpr.select_sequential_batch(
        candidate_descs=fallback_descs,
        candidate_numbers=fallback_nums,
        force_tol_eVA=10.0,
        delta=0.50,
        top_k=2,
    )
    # Greedy update after picking idx 0 lowers idx 1 below idx 2 -> picks [0, 2]
    assert fallback_selected == [0, 2]


def test_llpr_score_atoms_multi_delta_single_pass_and_missing_delta():
    """Verify score_atoms_multi_delta calls get_descriptors once, matches score_atoms, and returns None for uncalibrated deltas."""
    rng = np.random.default_rng(55)
    D = 6

    class _CountingCalc(_FakeMACECalculator):
        def __init__(self, feature_dim: int = 6):
            super().__init__(feature_dim=feature_dim)
            self.calls = 0

        def get_descriptors(
            self, atoms: Atoms, invariants_only: bool = True
        ) -> np.ndarray:
            self.calls += 1
            return super().get_descriptors(atoms, invariants_only=invariants_only)

    calc = _CountingCalc(feature_dim=D)
    train_configs = [
        _make_synthetic_atoms(
            ["Cu", "Cu", "O"], rng.normal(size=(3, D)), fixed_indices=[0]
        )
        for _ in range(15)
    ]
    val_configs = [
        _make_synthetic_atoms(
            ["Cu", "Cu", "O"],
            rng.normal(size=(3, D)),
            err_norms=rng.uniform(0.02, 0.15, size=3),
            fixed_indices=[0],
        )
        for _ in range(20)
    ]

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_configs, calc)
    # Calibrate ONLY at 0.50 and 0.20 (deliberately omit 0.10 to test missing optional delta)
    llpr.calibrate(val_configs, calc, tune_lambdas=(1e-4,), deltas=(0.50, 0.20))

    probe = _make_synthetic_atoms(
        ["Cu", "Cu", "O"], rng.normal(size=(3, D)), fixed_indices=[0]
    )

    calc.calls = 0
    multi = llpr.score_atoms_multi_delta(probe, calc, deltas=(0.50, 0.20, 0.10))
    assert calc.calls == 1
    assert set(multi.keys()) == {0.50, 0.20, 0.10}
    assert multi[0.10] is None

    for delta in (0.50, 0.20):
        ref_atom, ref_struct = llpr.score_atoms(
            probe, calc, calibrated=True, delta=delta
        )
        got_atom, got_struct = multi[delta]
        np.testing.assert_allclose(got_atom, ref_atom, rtol=1e-12, atol=1e-12)
        assert got_struct == pytest.approx(ref_struct, rel=1e-12, abs=1e-12)
        assert got_atom[0] == 0.0
