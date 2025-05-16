import pytest
from mlipflow.mlip_strategy import MACEModel
from mlipflow.core.single_point import run_single_point


def test_mace_for_single_point_energy():
    """Test MACEModel's single_point_energy method."""
    in_file = 'data/test_data.xyz'
    mace_model = MACEModel(mlip_file="data/mace_test.model", run_mode="local")
    run_single_point(
        in_file=in_file,
        out_file='test_output.xyz',
        output_prefix=mace_model.mlip_prefix,
        calculator=mace_model.get_calculator(job_name="MP_"),
    )
    assert os.path.exists('test_output.xyz')

