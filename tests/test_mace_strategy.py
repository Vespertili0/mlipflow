import os, pytest
from mlipflow.mlip_strategy import MACEModel

def test_mace_model_initialization():
    """Test MACEModel initialization."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_file = os.path.join(test_dir, 'data', 'mace_test.model')
    mace_model = MACEModel(
        mlip_file=mlip_file, 
        run_mode="local"
    )
    assert mace_model.run_mode == "local"

def test_mace_model_get_calculator():
    """Test MACEModel's get_calculator method."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_file = os.path.join(test_dir, 'data', 'mace_test.model')
    mace_model = MACEModel(
        mlip_file=mlip_file, 
        run_mode="local"
    )
    calculator = mace_model.get_calculator(job_name="test_job")
    assert calculator is not None
    assert isinstance(calculator, tuple)
