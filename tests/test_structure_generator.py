import os, pytest
from mlipflow.strategies.structure_generators import OPTGen
from mlipflow.strategies.mlip import MACEModel

def test_optgen_run(tmp_path):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(test_dir, 'data', 'test_data.xyz')
    mlip_name = os.path.join(test_dir, 'data', 'mace_test')
    mace = MACEModel(
        mlip_name=mlip_name,
        run_mode="local"
    )
    out_file = tmp_path / 'opt_test.xyz'
    OPTGen(opt_params={'fmax': 5.0, 'steps': 2}).generate_new_structures(
        in_file=in_file,
        out_file=str(out_file),
        calculator=mace.get_calculator(
            job_name = 'mMP_'
        ),
        remote_info=mace.remote_info
    )
    assert out_file.exists()