from __future__ import annotations

import os
import pickle
from pathlib import Path

from mace.calculators import MACECalculator

from mlipflow.adapters.wflio import MACELLPRCalculator
from mlipflow.strategies.mlip import GAPModel, MACEModel


def test_mace_model_initialization():
    """Test MACEModel initialization."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace_model = MACEModel(mlip_name=mlip_name, run_mode="local")
    assert mace_model.run_mode == "local"


def test_mace_model_get_calculator_local():
    """Test MACEModel's get_calculator method in local mode (pickle-safe MACELLPRCalculator)."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace_model = MACEModel(mlip_name=mlip_name, run_mode="local")
    calculator = mace_model.get_calculator(job_name="test_job")
    assert calculator is not None
    assert isinstance(calculator, tuple)
    # Default dispersion=True returns pickle-safe MACELLPRCalculator (compute_uq=False when .pt absent)
    assert calculator[0] == MACELLPRCalculator
    assert calculator[2]["dispersion"] is True
    assert calculator[2]["compute_uq"] is False
    assert calculator[2]["uq_file"] is None
    assert Path(calculator[2]["model_paths"]).is_absolute()
    assert pickle.loads(pickle.dumps(calculator))[0] == MACELLPRCalculator
    assert mace_model.remote_info is None


def test_mace_model_get_calculator_remote():
    """Test MACEModel's get_calculator method in remote mode (pickle-safe for expyre)."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace_model = MACEModel(mlip_name=mlip_name, run_mode="remote")
    calculator = mace_model.get_calculator(job_name="test_job")
    assert calculator is not None
    assert isinstance(calculator, tuple)
    assert calculator[0] == MACELLPRCalculator
    assert calculator[2]["dispersion"] is True
    assert calculator[2]["compute_uq"] is False
    assert pickle.loads(pickle.dumps(calculator))[0] == MACELLPRCalculator
    assert mace_model.remote_info is not None
    assert mace_model.remote_info.job_name == "test_job"


def test_mace_model_get_calculator_no_dispersion():
    """Test MACEModel's get_calculator method without dispersion or UQ artifact."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace_model = MACEModel(mlip_name=mlip_name, run_mode="local")
    calculator = mace_model.get_calculator(job_name="test_job", dispersion=False)
    assert calculator is not None
    assert isinstance(calculator, tuple)
    # dispersion=False and no .pt -> MACECalculator directly
    assert calculator[0] == MACECalculator


def test_mace_model_get_calculator_with_llpr_artifact(tmp_path):
    """Test get_calculator returns MACELLPRCalculator with resolved absolute paths when .pt exists."""
    model_file = tmp_path / "pot_v1.model"
    uq_file = tmp_path / "pot_v1_llpr.pt"
    model_file.write_text("dummy model", encoding="utf-8")
    uq_file.write_text("dummy uq", encoding="utf-8")

    mace = MACEModel(mlip_name=str(tmp_path / "pot_v1"), run_mode="local")

    # 1. With valid .pt and compute_uq=True (both dispersion=True and dispersion=False)
    for disp in (True, False):
        recipe = mace.get_calculator(
            job_name="mSP_",
            dispersion=disp,
            device="cpu",
            atom_indices=slice(48, None),
            exclude_fixed=False,
            compute_uq=True,
        )
        assert recipe[0] == MACELLPRCalculator
        kw = recipe[2]
        assert Path(kw["model_paths"]).is_absolute()
        assert kw["model_paths"] == str(model_file.resolve())
        assert Path(kw["uq_file"]).is_absolute()
        assert kw["uq_file"] == str(uq_file.resolve())
        assert kw["dispersion"] is disp
        assert kw["compute_uq"] is True
        assert kw["atom_indices"] == slice(48, None)
        assert kw["exclude_fixed"] is False
        assert pickle.loads(pickle.dumps(recipe))[0] == MACELLPRCalculator

    # 2. Explicit compute_uq=False disables UQ even when .pt exists
    recipe_opt_disp = mace.get_calculator(
        job_name="mSG_", dispersion=True, compute_uq=False
    )
    assert recipe_opt_disp[0] == MACELLPRCalculator
    assert recipe_opt_disp[2]["compute_uq"] is False
    assert recipe_opt_disp[2]["uq_file"] is None

    recipe_opt_nodisp = mace.get_calculator(
        job_name="mSG_", dispersion=False, compute_uq=False
    )
    assert recipe_opt_nodisp[0] == MACECalculator


def test_mace_model_uq_file_default_and_override():
    """Test MACEModel uq_file initialization and explicit override."""
    mace_default = MACEModel(mlip_name="model_v0", run_mode="local")
    assert mace_default.mlip_name == "model_v0"
    assert mace_default.model_file == "model_v0.model"
    assert mace_default.uq_file == "model_v0_llpr.pt"

    mace_custom = MACEModel(
        mlip_name="model_v0", run_mode="local", uq_file="custom_llpr.pt"
    )
    assert mace_custom.uq_file == "custom_llpr.pt"


def test_mace_model_set_model_name_synchronizes_paths():
    """Test set_model_name keeps mlip_name, model_file, and uq_file synchronized."""
    mace = MACEModel(mlip_name="model", run_mode="local")

    for version in range(3):
        expected_name = f"model_v{version}"
        mace.set_model_name(expected_name)
        assert mace.mlip_name == expected_name
        assert mace.model_file == f"{expected_name}.model"
        assert mace.uq_file == f"{expected_name}_llpr.pt"

    mace.set_model_name("model_v3", uq_file="override_v3.pt")
    assert mace.mlip_name == "model_v3"
    assert mace.model_file == "model_v3.model"
    assert mace.uq_file == "override_v3.pt"

    # model_file must remain writable for historical evaluation workflows
    mace.model_file = "/tmp/model_v3.model"
    assert mace.model_file == "/tmp/model_v3.model"


def test_gap_model_initialization():
    """Verify GAPModel initializes cleanly via MLIPStrategy.__init__(mlip_name=...)."""
    gap = GAPModel(mlip_file="gap_pot.xml", run_mode="local")
    assert gap.mlip_name == "gap_pot.xml"
    assert gap.mlip_file == "gap_pot.xml"
    assert gap.run_mode == "local"
