from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.mixing import SumCalculator
from ase.io import read, write
from mace.calculators import MACECalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from wfl.utils.parallel import construct_calculator_picklesafe

from mlipflow.adapters.wflio import ForceCheck, MACELLPRCalculator, select_config
from mlipflow.core.single_point import run_single_point
from mlipflow.data.llpr import PerElementLLPR
from mlipflow.strategies.mlip import MACEModel


def test_force_check():
    abort = ForceCheck(threshold=1.0, n_failed_steps=2)

    # Mock atoms with calculator
    at = Atoms("H")
    calc = MagicMock()
    at.calc = calc

    # Case 1: Forces below threshold
    calc.get_forces.return_value = np.array([[0.5, 0.0, 0.0]])
    assert bool(abort.atoms_ok(at)) is True

    # Case 2: Forces above threshold
    calc.get_forces.return_value = np.array([[1.5, 0.0, 0.0]])
    assert bool(abort.atoms_ok(at)) is False


def test_force_check_uq_threshold_and_inf_nan():
    """Verify ForceCheck UQ abort handles finite, inf, NaN, results fallback, and backward compat."""
    at = Atoms("H")
    calc = MagicMock()
    calc.get_forces.return_value = np.array([[0.2, 0.0, 0.0]])
    calc.results = {}
    at.calc = calc

    # Backward compatibility: uq_threshold=None ignores high UQ
    at.info["MACE_max_uq_p90"] = 99.0
    abort_legacy = ForceCheck(threshold=1.0, n_failed_steps=1, uq_threshold=None)
    assert bool(abort_legacy.atoms_ok(at)) is True

    # With uq_threshold set:
    abort_uq = ForceCheck(threshold=1.0, n_failed_steps=1, uq_threshold=0.25)

    # Below threshold -> ok
    at.info["MACE_max_uq_p90"] = 0.10
    assert bool(abort_uq.atoms_ok(at)) is True

    # Above threshold -> abort
    at.info["MACE_max_uq_p90"] = 0.30
    assert bool(abort_uq.atoms_ok(at)) is False

    # Unseen element (inf) -> abort
    at.info["MACE_max_uq_p90"] = float("inf")
    assert bool(abort_uq.atoms_ok(at)) is False

    # NaN -> abort
    at.info["MACE_max_uq_p90"] = float("nan")
    assert bool(abort_uq.atoms_ok(at)) is False

    # Fallback to at.calc.results["max_uq_p90"] when info key absent
    del at.info["MACE_max_uq_p90"]
    calc.results = {"max_uq_p90": 0.40}
    assert bool(abort_uq.atoms_ok(at)) is False

    calc.results = {"max_uq_p90": 0.15}
    assert bool(abort_uq.atoms_ok(at)) is True

    # Raw force violation still aborts even when UQ is ok
    calc.get_forces.return_value = np.array([[2.0, 0.0, 0.0]])
    assert bool(abort_uq.atoms_ok(at)) is False


def test_select_config():
    # Helper to create list of dummy objects
    traj = list(range(20))

    # Case < 10
    short_traj = traj[:5]
    assert select_config(short_traj) == [4]

    # Case == 500
    long_traj = list(range(500))
    assert select_config(long_traj) == [499]

    # Case else (e.g. 20)
    assert select_config(traj) == [10]


def test_macellpr_calculator_pickle_cache_hit_and_missing_p90(tmp_path):
    """Verify lazy init, pickle safety, consecutive cache-hit UQ attachment, single-pass MD, and missing p90."""
    D = 6

    class _DummyMACE:
        def __init__(self, **_kwargs):
            self.results = {}
            self.calc_calls = 0
            self.desc_calls = 0

        def calculate(self, atoms, properties, system_changes):
            _ = properties, system_changes
            self.calc_calls += 1
            self.results = {
                "energy": -10.0,
                "forces": np.ones((len(atoms), 3), dtype=np.float64),
                "stress": np.zeros(6, dtype=np.float64),
            }

        def get_descriptors(self, atoms, invariants_only: bool = True):
            _ = invariants_only
            self.desc_calls += 1
            return np.asarray(atoms.arrays["fake_desc"], dtype=np.float64)

    class _DummyD3:
        def __init__(self, **_kwargs):
            self.results = {}
            self.calc_calls = 0

        def calculate(self, atoms, properties, system_changes):
            _ = properties, system_changes
            self.calc_calls += 1
            self.results = {
                "energy": -0.5,
                "forces": np.full((len(atoms), 3), 0.25, dtype=np.float64),
                "stress": np.full(6, 0.1, dtype=np.float64),
            }

    # Build a synthetic calibrated .pt artifact with deltas=(0.50, 0.20, 0.10)
    def _make_at(descs: np.ndarray, err: float | None = None) -> Atoms:
        at = Atoms(
            "CuO", positions=[[0, 0, 0], [0, 0, 1.5]], pbc=True, cell=[10, 10, 10]
        )
        at.arrays["fake_desc"] = np.asarray(descs, dtype=np.float64)
        if err is not None:
            at.arrays["MACE_forces"] = np.zeros((2, 3), dtype=np.float64)
            at.arrays["DFT_forces"] = np.array([[err, 0.0, 0.0], [err, 0.0, 0.0]])
        return at

    basis = np.eye(D, dtype=np.float64)
    dummy_for_fit = _DummyMACE()
    train_ats = [_make_at(np.vstack([basis[i % 2], basis[i % 2]])) for i in range(20)]
    val_ats = [
        _make_at(np.vstack([basis[i % 2], basis[i % 2]]), err=0.04) for i in range(20)
    ]

    llpr = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr.fit(train_ats, dummy_for_fit)
    llpr.calibrate(
        val_ats, dummy_for_fit, tune_lambdas=(1e-4,), deltas=(0.50, 0.20, 0.10)
    )
    uq_full = tmp_path / "pot_v0_llpr.pt"
    llpr.save(uq_full)

    # Also save a custom .pt without delta=0.10
    llpr_no_p90 = PerElementLLPR(lambda_reg=1e-4, device="cpu")
    llpr_no_p90.fit(train_ats, dummy_for_fit)
    llpr_no_p90.calibrate(val_ats, dummy_for_fit, tune_lambdas=(1e-4,), deltas=(0.50,))
    uq_no_p90 = tmp_path / "pot_no_p90_llpr.pt"
    llpr_no_p90.save(uq_no_p90)

    created_mace: list[_DummyMACE] = []
    created_d3: list[_DummyD3] = []

    def _make_mace(**kw):
        inst = _DummyMACE(**kw)
        created_mace.append(inst)
        return inst

    def _make_d3(**kw):
        inst = _DummyD3(**kw)
        created_d3.append(inst)
        return inst

    with (
        patch("mlipflow.adapters.wflio.MACECalculator", side_effect=_make_mace),
        patch("mlipflow.adapters.wflio.TorchDFTD3Calculator", side_effect=_make_d3),
    ):
        calc = MACELLPRCalculator(
            model_paths=str(tmp_path / "pot_v0.model"),
            uq_file=str(uq_full),
            dispersion=True,
            dtype="float32",
            device="cpu",
        )
        # Lazy init: nothing instantiated yet
        assert created_mace == []
        assert created_d3 == []
        assert calc._llpr is None

        # Pickle round-trip BEFORE calculate()
        calc = pickle.loads(pickle.dumps(calc))

        at1 = _make_at(np.vstack([basis[0], basis[1]]))
        at1.calc = calc

        # Sequential get_forces() -> get_potential_energy() on same geometry runs only ONCE
        f1 = at1.get_forces()
        e1 = at1.get_potential_energy()
        assert e1 == pytest.approx(-10.5)
        np.testing.assert_allclose(f1, 1.25)
        assert at1.calc is calc
        assert created_mace[0].calc_calls == 1
        assert created_d3[0].calc_calls == 1
        assert created_mace[0].desc_calls == 1

        # Verify all 5 inline UQ annotations are present on at1
        assert "MACE_atom_uq" in at1.arrays
        assert "MACE_max_uq" in at1.info
        assert "MACE_max_uq_p90" in at1.info
        assert at1.info["MACE_max_uq_p90"] >= at1.info["MACE_max_uq"] - 1e-12
        assert at1.info["MACE_uq_delta"] == pytest.approx(0.50)
        assert at1.info["MACE_uq_model"] == uq_full.name

        # Identical consecutive frame hits ASE cache (no recompute) AND still gets UQ attached (non-aliased)
        at2 = _make_at(np.vstack([basis[0], basis[1]]))
        at2.calc = calc
        e2 = at2.get_potential_energy()
        assert e2 == pytest.approx(e1)
        assert created_mace[0].calc_calls == 1
        assert created_mace[0].desc_calls == 1
        assert "MACE_atom_uq" in at2.arrays
        assert at2.arrays["MACE_atom_uq"] is not at1.arrays["MACE_atom_uq"]
        np.testing.assert_allclose(
            at2.arrays["MACE_atom_uq"], at1.arrays["MACE_atom_uq"]
        )
        assert at2.info["MACE_max_uq"] == pytest.approx(at1.info["MACE_max_uq"])
        assert at2.info["MACE_max_uq_p90"] == pytest.approx(at1.info["MACE_max_uq_p90"])
        assert at2.info["MACE_uq_delta"] == pytest.approx(0.50)
        assert at2.info["MACE_uq_model"] == uq_full.name

        # get_descriptors delegates cleanly
        desc_out = calc.get_descriptors(at1)
        assert desc_out.shape == (2, D)

        # Pickle round-trip AFTER calculate() drops live objects and lazily re-initializes
        calc_restored = pickle.loads(pickle.dumps(calc))
        assert calc_restored._mace_calc is None
        assert calc_restored._dispersion_calc is None
        assert calc_restored._llpr is None

        at3 = _make_at(np.vstack([basis[0], basis[1]]))
        at3.calc = calc_restored
        assert at3.get_potential_energy() == pytest.approx(-10.5)
        assert "MACE_max_uq" in at3.info

        # Custom .pt missing p90_delta=0.10 gracefully omits MACE_max_uq_p90 without crashing
        calc_no_p90 = MACELLPRCalculator(
            model_paths=str(tmp_path / "pot_v0.model"),
            uq_file=str(uq_no_p90),
            dispersion=False,
        )
        at4 = _make_at(np.vstack([basis[0], basis[1]]))
        at4.calc = calc_no_p90
        assert at4.get_potential_energy() == pytest.approx(-10.0)
        assert "MACE_atom_uq" in at4.arrays
        assert "MACE_max_uq" in at4.info
        assert "MACE_max_uq_p90" not in at4.info


def test_macellpr_calculator_real_mace_equivalence_and_single_point(tmp_path):
    """Real mace_test.model + Cu(100) test: SumCalculator equivalence, single-pass run_single_point, and construct_calculator_picklesafe."""
    data_dir = Path(__file__).resolve().parent / "data"
    model_path = data_dir / "mace_test.model"
    src_xyz = data_dir / "test_data.xyz"
    all_frames = read(src_xyz, ":")
    atom_slice = slice(48, None)

    ref_mace = MACECalculator(
        model_paths=str(model_path), device="cpu", default_dtype="float32"
    )
    ref_d3 = TorchDFTD3Calculator(
        damping="bj", old=False, device="cpu", dtype=torch.float32
    )
    ref_sum = SumCalculator(calcs=[ref_mace, ref_d3])

    # Fit & calibrate PerElementLLPR on a lean slice of test_data.xyz
    train_frames = all_frames[:6]
    val_frames = [at.copy() for at in all_frames[6:10]]
    rng = np.random.default_rng(77)
    for at in val_frames:
        at_c = at.copy()
        at_c.calc = ref_mace
        f_mace = at_c.get_forces()
        at.arrays["MACE_forces"] = f_mace
        at.arrays["DFT_forces"] = f_mace + rng.normal(0.0, 0.03, size=f_mace.shape)

    llpr = PerElementLLPR(invariants_only=True, device="cpu")
    llpr.fit(train_frames, ref_mace, atom_indices=atom_slice, exclude_fixed=False)
    llpr.calibrate(
        val_frames,
        ref_mace,
        atom_indices=atom_slice,
        exclude_fixed=False,
        deltas=(0.50, 0.20, 0.10),
        tune_lambdas=(1e-4,),
    )
    uq_file = tmp_path / "mace_test_llpr.pt"
    llpr.save(uq_file)

    # 1. Numerical equivalence with dispersion=True vs SumCalculator([MACE, D3])
    probe = all_frames[0].copy()
    probe_ref = probe.copy()
    probe_ref.calc = ref_sum
    e_ref_disp = probe_ref.get_potential_energy()
    f_ref_disp = probe_ref.get_forces()

    calc_disp = MACELLPRCalculator(
        model_paths=str(model_path),
        uq_file=str(uq_file),
        dispersion=True,
        dtype="float32",
        device="cpu",
        atom_indices=atom_slice,
        exclude_fixed=False,
    )
    probe_disp = probe.copy()
    probe_disp.calc = calc_disp
    e_llpr_disp = probe_disp.get_potential_energy()
    f_llpr_disp = probe_disp.get_forces()
    assert abs(e_llpr_disp - e_ref_disp) < 1e-5
    assert float(np.max(np.abs(f_llpr_disp - f_ref_disp))) < 1e-5

    # 2. Numerical equivalence with dispersion=False vs raw MACECalculator
    probe_ref_nodisp = probe.copy()
    probe_ref_nodisp.calc = ref_mace
    e_ref_nodisp = probe_ref_nodisp.get_potential_energy()
    f_ref_nodisp = probe_ref_nodisp.get_forces()

    calc_nodisp = MACELLPRCalculator(
        model_paths=str(model_path),
        uq_file=str(uq_file),
        dispersion=False,
        dtype="float32",
        device="cpu",
        atom_indices=atom_slice,
        exclude_fixed=False,
    )
    probe_nodisp = probe.copy()
    probe_nodisp.calc = calc_nodisp
    assert abs(probe_nodisp.get_potential_energy() - e_ref_nodisp) < 1e-5
    assert float(np.max(np.abs(probe_nodisp.get_forces() - f_ref_nodisp))) < 1e-5

    # 3. Single-pass run_single_point with MACEModel.get_calculator(dispersion=True)
    # including identical consecutive frames + construct_calculator_picklesafe check
    mace_model = MACEModel(
        mlip_name=str(data_dir / "mace_test"), run_mode="local", uq_file=str(uq_file)
    )
    calc_recipe = mace_model.get_calculator(
        job_name="mSP_", dispersion=True, atom_indices=atom_slice, exclude_fixed=False
    )
    # Verify pickle-safety of the recipe and construct_calculator_picklesafe
    unpickled_recipe = pickle.loads(pickle.dumps(calc_recipe))
    constructed = construct_calculator_picklesafe(unpickled_recipe)
    assert isinstance(constructed, MACELLPRCalculator)

    sp_in = tmp_path / "sp_in.xyz"
    sp_out = tmp_path / "mace_sp.xyz"
    write(sp_in, [all_frames[0].copy(), all_frames[0].copy(), all_frames[1].copy()])

    run_single_point(
        in_file=str(sp_in),
        out_file=str(sp_out),
        output_prefix=mace_model.mlip_prefix,
        calculator=calc_recipe,
    )

    out_frames = read(sp_out, ":")
    assert len(out_frames) == 3
    src_frames_sp = [all_frames[0], all_frames[0], all_frames[1]]
    for at, src_at in zip(out_frames, src_frames_sp, strict=True):
        exp_atom50, exp_max50 = llpr.score_atoms(
            src_at, ref_mace, atom_indices=atom_slice, exclude_fixed=False, delta=0.50
        )
        _, exp_max10 = llpr.score_atoms(
            src_at, ref_mace, atom_indices=atom_slice, exclude_fixed=False, delta=0.10
        )
        assert "MACE_energy" in at.info
        assert "MACE_forces" in at.arrays
        assert "MACE_atom_uq" in at.arrays
        assert "MACE_max_uq" in at.info
        assert "MACE_max_uq_p90" in at.info
        assert at.info["MACE_uq_delta"] == pytest.approx(0.50)
        assert at.info["MACE_uq_model"] == uq_file.name
        assert np.all(at.arrays["MACE_atom_uq"][:48] == 0.0)
        assert np.all(at.arrays["MACE_atom_uq"][48:] > 0.0)
        np.testing.assert_allclose(at.arrays["MACE_atom_uq"], exp_atom50, rtol=1e-5)
        assert at.info["MACE_max_uq"] == pytest.approx(exp_max50, rel=1e-5)
        assert at.info["MACE_max_uq_p90"] == pytest.approx(exp_max10, rel=1e-5)
    # Identical consecutive frames must produce identical energies and UQ
    assert out_frames[0].info["MACE_energy"] == pytest.approx(
        out_frames[1].info["MACE_energy"]
    )
    assert out_frames[0].info["MACE_max_uq"] == pytest.approx(
        out_frames[1].info["MACE_max_uq"]
    )

    # 4. Real subprocess smoke test from a separate working directory (simulating expyre subdir)
    remote_subdir = tmp_path / "expyre_remote_subdir"
    remote_subdir.mkdir()
    recipe_pkl = tmp_path / "calc_recipe.pkl"
    recipe_pkl.write_bytes(pickle.dumps(calc_recipe))
    probe_xyz = tmp_path / "probe.xyz"
    write(probe_xyz, all_frames[0].copy())

    child_code = (
        "import json, pickle; "
        "from ase.io import read; "
        "from wfl.utils.parallel import construct_calculator_picklesafe; "
        f"recipe = pickle.loads(open({str(recipe_pkl)!r}, 'rb').read()); "
        "calc = construct_calculator_picklesafe(recipe); "
        f"at = read({str(probe_xyz)!r}); "
        "at.calc = calc; "
        "e = float(at.get_potential_energy()); "
        "print(json.dumps({'energy': e, 'max_uq': float(at.info['MACE_max_uq']), 'uq_model': at.info['MACE_uq_model']}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=remote_subdir,
        check=True,
        capture_output=True,
        text=True,
    )
    child_res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert child_res["energy"] == pytest.approx(e_llpr_disp, abs=1e-5)
    assert child_res["max_uq"] == pytest.approx(
        out_frames[0].info["MACE_max_uq"], abs=1e-5
    )
    assert child_res["uq_model"] == uq_file.name
