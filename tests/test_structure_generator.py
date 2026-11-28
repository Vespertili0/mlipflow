from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from mace.calculators import MACECalculator

from mlipflow.data.llpr import PerElementLLPR
from mlipflow.strategies.dft import EMTCalc
from mlipflow.strategies.mlip import MACEModel
from mlipflow.strategies.structure_generators import MDGen, OPTGen


def test_optgen_run(tmp_path):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(test_dir, "data", "test_data.xyz")
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace = MACEModel(mlip_name=mlip_name, run_mode="local")

    # Mock get_calculator to return EMT calculator to avoid MACE execution errors
    mace.get_calculator = MagicMock(side_effect=EMTCalc().get_calculator)
    mace.remote_info = None

    out_file = tmp_path / "opt_test.xyz"
    # Use traj_subselect=None to ensure output is written even if not converged
    OPTGen(
        params={"fmax": 5.0, "steps": 2}, traj_subselect=None
    ).generate_new_structures(
        in_file=in_file,
        out_file=str(out_file),
        calculator=mace.get_calculator(job_name="mMP_"),
        remote_info=mace.remote_info,
    )
    assert out_file.exists()


def test_mdgen_uq_threshold_aborts_rattled_ood_early(tmp_path):
    """Verify MDGen(..., uq_threshold=...) aborts early on a rattled OOD Cu(100) frame."""
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
    rng = np.random.default_rng(19)
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

    eq_frame = train_frames[0].copy()
    eq_frame.set_constraint(FixAtoms(indices=list(range(48))))

    ood_frame = train_frames[0].copy()
    ood_frame.set_constraint(FixAtoms(indices=list(range(48))))
    pos = ood_frame.get_positions()
    pos[48:] += np.random.default_rng(99).normal(
        0.0, 0.25, size=(len(ood_frame) - 48, 3)
    )
    ood_frame.set_positions(pos)

    _, eq_p90 = llpr.score_atoms(
        eq_frame, mace_calc, atom_indices=atom_slice, exclude_fixed=False, delta=0.10
    )
    _, ood_p90 = llpr.score_atoms(
        ood_frame, mace_calc, atom_indices=atom_slice, exclude_fixed=False, delta=0.10
    )
    assert ood_p90 > eq_p90
    uq_thr = 0.5 * (eq_p90 + ood_p90)

    mace = MACEModel(
        mlip_name=str(data_dir / "mace_test"), run_mode="local", uq_file=str(uq_file)
    )
    calc_recipe = mace.get_calculator(
        job_name="mMD_",
        dispersion=False,
        atom_indices=atom_slice,
        exclude_fixed=False,
        compute_uq=True,
    )

    md_gen = MDGen(
        uncertainty_thrs=float("inf"),  # Isolate UQ-based abort from raw force abort
        n_failed_steps=1,
        uq_threshold=uq_thr,
        params={"steps": 6, "dt": 0.5, "temperature": 50.0, "traj_step_interval": 1},
    )

    ood_in = tmp_path / "ood_in.xyz"
    ood_out = tmp_path / "ood_md.xyz"
    write(ood_in, [ood_frame])
    md_gen.generate_new_structures(
        in_file=str(ood_in),
        out_file=str(ood_out),
        calculator=calc_recipe,
        remote_info=None,
    )
    ood_traj = read(ood_out, ":")
    # Rattled OOD structure triggers immediate ForceCheck UQ abort well before 6 steps
    assert len(ood_traj) < 6
    assert "MACE_max_uq_p90" in ood_traj[0].info
    assert float(ood_traj[0].info["MACE_max_uq_p90"]) > uq_thr
