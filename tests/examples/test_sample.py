import os
import pytest
from ase import Atoms
from mlipflow.data_manager import DataManager
from mlipflow.structure_generator import MDGen, OPTGen
from mlipflow.mlip_strategy import GAPModel

@pytest.fixture
def setup_data_manager(tmp_path):
    """Fixture to set up a DataManager instance."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    return DataManager(workdir=str(workdir))

def test_data_manager_setup_iteration(setup_data_manager):
    """Test DataManager's setup_iteration method."""
    dm = setup_data_manager
    dm.setup_iteration(fit_idx=0)
    assert os.path.exists(dm.iter_dir)
    assert os.path.exists(dm.MLIP_dir)
    assert os.path.exists(dm.SGen_dir)

def test_data_manager_update_training_data(setup_data_manager, tmp_path):
    """Test updating training data."""
    dm = setup_data_manager
    training_file = tmp_path / "training.xyz"
    add_file = tmp_path / "add.xyz"
    out_file = tmp_path / "out.xyz"

    # Create dummy Atoms objects
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]])
    atoms1.write(training_file)
    atoms2.write(add_file)

    dm.update_training_data(training_xyz=str(training_file), add_xyz=str(add_file), out_file=str(out_file))
    assert os.path.exists(out_file)

def test_mdgen_generate_new_structures(tmp_path):
    """Test MDGen's generate_new_structures method."""
    md_params = {'steps': 10, 'dt': 1, 'temperature': 300.0, 'traj_step_interval': 1}
    mdgen = MDGen(uncertainty_thrs=0.1, md_params=md_params)

    in_file = tmp_path / "input.xyz"
    out_file = tmp_path / "output.xyz"

    # Create dummy Atoms object
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.write(in_file)

    # Mock calculator
    class MockCalculator:
        def calculate(self, *args, **kwargs):
            pass

    mdgen.generate_new_structures(in_file=str(in_file), out_file=str(out_file), calculator=MockCalculator())
    assert os.path.exists(out_file)

def test_optgen_generate_new_structures(tmp_path):
    """Test OPTGen's generate_new_structures method."""
    opt_params = {'fmax': 0.1, 'steps': 50}
    optgen = OPTGen(opt_params=opt_params)

    in_file = tmp_path / "input.xyz"
    out_file = tmp_path / "output.xyz"

    # Create dummy Atoms object
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.write(in_file)

    # Mock calculator
    class MockCalculator:
        def calculate(self, *args, **kwargs):
            pass

    optgen.generate_new_structures(in_file=str(in_file), out_file=str(out_file), calculator=MockCalculator())
    assert os.path.exists(out_file)

def test_gapmodel_get_calculator():
    """Test GAPModel's get_calculator method."""
    gap_model = GAPModel(mlip_file="dummy.xml", run_mode="local")
    calculator = gap_model.get_calculator(job_name="test_job")
    assert calculator is not None
