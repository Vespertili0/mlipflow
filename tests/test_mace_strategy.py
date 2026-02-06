import os, pytest
from ase.calculators.mixing import SumCalculator
from mace.calculators import MACECalculator
from mlipflow.strategies.mlip import MACEModel

def test_mace_model_initialization():
    """Test MACEModel initialization."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, 'data', 'mace_test')
    mace_model = MACEModel(
        mlip_name=mlip_name,
        run_mode="local"
    )
    assert mace_model.run_mode == "local"

def test_mace_model_get_calculator_local():
    """Test MACEModel's get_calculator method in local mode."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, 'data', 'mace_test')
    mace_model = MACEModel(
        mlip_name=mlip_name,
        run_mode="local"
    )
    calculator = mace_model.get_calculator(job_name="test_job")
    assert calculator is not None
    assert isinstance(calculator, tuple)
    # Default is dispersion=True, so it returns SumCalculator
    assert calculator[0] == SumCalculator
    assert mace_model.remote_info is None

def test_mace_model_get_calculator_remote():
    """Test MACEModel's get_calculator method in remote mode."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, 'data', 'mace_test')
    mace_model = MACEModel(
        mlip_name=mlip_name,
        run_mode="remote"
    )
    calculator = mace_model.get_calculator(job_name="test_job")
    assert calculator is not None
    assert isinstance(calculator, tuple)
    assert calculator[0] == SumCalculator
    assert mace_model.remote_info is not None
    assert mace_model.remote_info.job_name == "test_job"

def test_mace_model_get_calculator_no_dispersion():
    """Test MACEModel's get_calculator method without dispersion."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_name = os.path.join(test_dir, 'data', 'mace_test')
    mace_model = MACEModel(
        mlip_name=mlip_name,
        run_mode="local"
    )
    calculator = mace_model.get_calculator(job_name="test_job", dispersion=False)
    assert calculator is not None
    assert isinstance(calculator, tuple)
    # dispersion=False -> MACECalculator directly
    assert calculator[0] == MACECalculator
