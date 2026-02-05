import os
import pytest
from unittest.mock import MagicMock, patch
from mlipflow.graphflow.nodes import run_dft_sp, EnsembleState
from mlipflow.strategies.dft import QChemStrategy
from mlipflow.graphflow.graphs import execute_dft_single_point_block

class MockQChemStrategy(QChemStrategy):
    def __init__(self):
        super().__init__()
        self.qe_prefix = 'DFT_'
        self.remote_info = None

    def get_calculator(self, job_name, **kwargs):
        return (MagicMock(), [], {})

@pytest.fixture
def mock_state(tmp_path):
    config_file = tmp_path / "test_config.xyz"
    with open(config_file, 'w') as f:
        f.write("dummy")

    qchem_strategy = MockQChemStrategy()
    mlip_strategy = MagicMock()
    mlip_strategy.mlip_prefix = 'MACE_'

    state = EnsembleState(
        configs=[str(config_file)],
        qchem_strategy=qchem_strategy,
        mlip_strategy=mlip_strategy
    )
    return state

def test_run_dft_sp(mock_state):
    with patch('mlipflow.graphflow.nodes.run_single_point') as mock_run:
        new_state = run_dft_sp(mock_state)

        assert len(new_state['configs']) == 1
        assert new_state['configs'][0].endswith('.dft.xyz')
        mock_run.assert_called_once()

        # Verify call arguments
        args, kwargs = mock_run.call_args
        assert kwargs['in_file'] == mock_state['configs']
        assert kwargs['output_prefix'] == 'DFT_'

def test_execute_dft_single_point_block(mock_state):
    # We need to mock nodes because we don't want to run actual DFT or selection
    with patch('mlipflow.graphflow.nodes.run_single_point') as mock_run_sp:
        # We also need to mock assess_n_select because it relies on files existing and content
        with patch('mlipflow.graphflow.nodes.split_configset_by_force_agreement') as mock_split:

            app = execute_dft_single_point_block()
            final_state = app.invoke(mock_state)

            # verify flow
            mock_run_sp.assert_called()
            mock_split.assert_called()

            # assess_n_select sets configs=['train_dft.xyz'], outfile=['test_dft.xyz']
            assert final_state['configs'] == ['train_dft.xyz']
            assert final_state['outfile'] == ['test_dft.xyz']
