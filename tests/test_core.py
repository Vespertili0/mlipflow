import os, pytest
from mlipflow.mlip_strategy import MACEModel
from mlipflow.core.single_point import run_single_point


def test_mace_for_single_point_energy():
    """Test MACEModel's single_point_energy method."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(test_dir, 'data', 'test_data.xyz')
    mlip_file = os.path.join(test_dir, 'data', 'mace_test.model')
    mace_model = MACEModel(
        mlip_file=mlip_file, 
        run_mode="local"
    )
    run_single_point(
        in_file=in_file,
        out_file='test_output.xyz',
        output_prefix=mace_model.mlip_prefix,
        calculator=mace_model.get_calculator(job_name="MP_"),
    )
    assert os.path.exists('test_output.xyz')

