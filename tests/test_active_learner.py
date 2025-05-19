import os, pytest
from unittest.mock import MagicMock
from mlipflow.active_learning import ActiveLearner
from mlipflow.data_manager import DataManager
from mlipflow.structure_generator import MDGen
from mlipflow.mlip_strategy import MACEModel

@pytest.fixture
def mock_active_learner():
    """Fixture to set up a mock ActiveLearner instance."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_file = os.path.join(test_dir, 'data', 'mace_test.model')
    data_manager = DataManager(workdir=test_dir)
    structure_gen = MDGen(uncertainty_thrs=0.1)
    mlip_strategy = MACEModel(mlip_file=mlip_file, run_mode="local")
    learner = ActiveLearner(
        data_manager=data_manager,
        structure_generation_strategy=structure_gen,
        mlip_strategy=mlip_strategy,
    )
    return learner

#def test_initialise_learning(mock_active_learner, tmp_path):
#    """Test the initialise_learning method."""
#    learner = mock_active_learner
#    learner.data_manager.initialise_ensembles = MagicMock()
#    learner.run_single_point = MagicMock()
#    learner.mlip_strategy.fit_new_model = MagicMock()
#
#    learner.initialise_learning(
#        ensemble_traj=str(tmp_path / "ensemble.traj"),
#        initial_xyz=str(tmp_path / "initial.xyz"),
#        qchem=False,
#    )
#
#    learner.data_manager.initialise_ensembles.assert_called_once()
#    learner.run_single_point.assert_not_called()  # No DFT when qchem=False
#    learner.mlip_strategy.fit_new_model.assert_called_once()

def test_run_iteration(mock_active_learner):
    """Test the run_iteration method."""
    learner = mock_active_learner
    learner.run_single_point = MagicMock()
    learner.structure_generation_strategy.generate_new_structures = MagicMock()
    learner.mlip_strategy.fit_new_model = MagicMock()

    learner.run_iteration(n_iter=1)

    learner.run_single_point.assert_called()
    learner.structure_generation_strategy.generate_new_structures.assert_called_once()
    learner.mlip_strategy.fit_new_model.assert_called_once()
