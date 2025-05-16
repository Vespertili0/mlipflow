import pytest
from mlipflow.structure_generator import OPTGen
from mlipflow.mlip_strategy import MACEModel

def test_optgen_run():
    mace = MACEModel(
        mlip_file='data/mace_test.model',
        run_mode='local'
    )
    OPTGen(opt_params={'fmax': 0.5}).generate_new_structures(
        in_file='data/test_data.xyz',
        out_file='opt_test.xyz',
        calculator=mace.get_calculator(
            job_name = 'mMP_'
        ),
        remote_info=mace.remote_info
    )
    assert os.path.exists('opt_test.xyz')