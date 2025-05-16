import pytest
from mlipflow.mlip_strategy import MACEModel

def test_mace_model_initialization():
    """Test MACEModel initialization."""
    mace_model = MACEModel(mlip_file="../data/mace_test.model", run_mode="local")
    assert mace_model.mlip_file == "../data/mace_test.model"
    assert mace_model.run_mode == "local"

def test_mace_model_get_calculator():
    """Test MACEModel's get_calculator method."""
    mace_model = MACEModel(mlip_file="../data/mace_test.model", run_mode="local")
    calculator = mace_model.get_calculator(job_name="test_job")
    assert calculator is not None
    assert isinstance(calculator, tuple)
