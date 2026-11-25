from __future__ import annotations

import os

from ase.calculators.mixing import SumCalculator
from mace.calculators import MACECalculator

from mlipflow.strategies.mlip import MACEModel


def test_mace_model_initialization():
    """Test MACEModel initialization."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace_model = MACEModel(mlip_name=mlip_name, run_mode="local")
    assert mace_model.run_mode == "local"


def test_mace_model_get_calculator_local():
    """Test MACEModel's get_calculator method in local mode."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace_model = MACEModel(mlip_name=mlip_name, run_mode="local")
    calculator = mace_model.get_calculator(job_name="test_job")
    assert calculator is not None
    assert isinstance(calculator, tuple)
    # Default is dispersion=True, so it returns SumCalculator
    assert calculator[0] == SumCalculator
    assert mace_model.remote_info is None


def test_mace_model_get_calculator_remote():
    """Test MACEModel's get_calculator method in remote mode."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace_model = MACEModel(mlip_name=mlip_name, run_mode="remote")
    calculator = mace_model.get_calculator(job_name="test_job")
    assert calculator is not None
    assert isinstance(calculator, tuple)
    assert calculator[0] == SumCalculator
    assert mace_model.remote_info is not None
    assert mace_model.remote_info.job_name == "test_job"


def test_mace_model_get_calculator_no_dispersion():
    """Test MACEModel's get_calculator method without dispersion."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace_model = MACEModel(mlip_name=mlip_name, run_mode="local")
    calculator = mace_model.get_calculator(job_name="test_job", dispersion=False)
    assert calculator is not None
    assert isinstance(calculator, tuple)
    # dispersion=False -> MACECalculator directly
    assert calculator[0] == MACECalculator


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
