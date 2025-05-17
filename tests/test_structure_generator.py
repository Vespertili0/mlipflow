import os, pytest
from mlipflow.structure_generator import OPTGen
from mlipflow.mlip_strategy import MACEModel

def test_optgen_run():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(test_dir, 'data', 'test_data.xyz')
    mlip_file = os.path.join(test_dir, 'data', 'mace_test.model')
    mace = MACEModel(
        mlip_file=mlip_file, 
        run_mode="local"
    )
    OPTGen(opt_params={'fmax': 0.75}).generate_new_structures(
        in_file=in_file,
        out_file='opt_test.xyz',
        calculator=mace.get_calculator(
            job_name = 'mMP_'
        ),
        remote_info=mace.remote_info
    )
    assert os.path.exists('opt_test.xyz')