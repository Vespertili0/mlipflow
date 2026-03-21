import os
import shutil
import pytest
from unittest.mock import MagicMock
from mlipflow.graphflow.nodes import run_dft_sp, assess_n_select, EnsembleState
from mlipflow.strategies.dft import EMTCalc
from mlipflow.graphflow.graphs import execute_dft_single_point_block
from ase.io import read, write

@pytest.fixture
def real_data_setup(tmp_path):
    """
    Setup fixture that copies real data to a temporary directory.
    Returns the path to the config file and the state.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    src_xyz = os.path.join(test_dir, 'data', 'test_data.xyz')

    # We need a model file for MACE strategy even if we mock the strategy itself
    # just to pass some checks if any. But here we primarily test DFT part.

    config_file = tmp_path / "test_data.xyz"
    shutil.copy2(src_xyz, config_file)

    # Setup strategies
    qchem_strategy = EMTCalc()

    # Mock MLIP strategy just to provide the prefix
    mlip_strategy = MagicMock()
    mlip_strategy.mlip_prefix = 'MACE_'

    state = EnsembleState(
        configs=[str(config_file)],
        qchem_strategy=qchem_strategy,
        mlip_strategy=mlip_strategy
    )
    return state, tmp_path

def test_run_dft_sp_with_emt(real_data_setup):
    """
    Test run_dft_sp using EMTCalc and real data.
    This verifies that run_single_point actually works with the strategy and data.
    """
    state, tmp_path = real_data_setup

    # Execute the node
    new_state = run_dft_sp(state)

    # Verify output file exists
    output_file = new_state['configs'][0]
    assert os.path.exists(output_file)
    assert output_file.endswith('.dft.xyz')

    # Verify content
    atoms = read(output_file, ':')
    assert len(atoms) > 0
    # Check if results are present. EMTCalc uses "DFT_" prefix in QChemStrategy?
    # No, EMTCalc inherits QChemStrategy which sets qe_prefix = 'DFT_'
    # generic_calc with output_prefix='DFT_' should store results in info/arrays with that prefix.
    # For 'energy', it should be 'DFT_energy'.

    # EMT calculator might write 'energy' directly to atoms.calc.results,
    # but generic_calc + output_prefix handles the storage.
    # Let's check info keys
    assert 'DFT_energy' in atoms[0].info
    assert 'DFT_forces' in atoms[0].arrays

def test_execute_dft_single_point_block_integration(real_data_setup):
    """
    Test the full DFT block execution with EMTCalc.
    Flow: dft_sp -> assess_n_select
    """
    state, tmp_path = real_data_setup

    # We need to ensure that the data has MACE_forces for assess_n_select to work,
    # as split_configset_by_force_agreement compares DFT_forces and MACE_forces.
    # Since we are not running MACE here, we can manually add dummy MACE forces to the input file
    # OR we can update the input file before running the graph.

    # Let's update the input file with dummy MACE forces.
    atoms = read(state['configs'][0], ':')
    import numpy as np
    for at in atoms:
        at.arrays['MACE_forces'] = np.random.random((len(at), 3))
    write(state['configs'][0], atoms)

    # Initialise graph
    app = execute_dft_single_point_block()

    # Check results
    # assess_n_select should produce 'train_dft.xyz' and 'test_dft.xyz' (prefixed/suffixed)
    # The node assess_n_select hardcodes output file name to 'dft.xyz'
    # and split_configset_by_force_agreement writes to {suffix}_{out_file}.
    # So we expect 'train_dft.xyz' and 'test_dft.xyz' in the CWD?
    # Wait, where does it write?

    # In assess_n_select:
    # out_file='dft.xyz'
    # split_configset_by_force_agreement(..., out_file='dft.xyz', ...)

    # If run in tmp_path context, it should be fine.
    # But graph execution might not respect tmp_path if not explicitly handled.
    # The state['configs'] are absolute paths from tmp_path.
    # But 'dft.xyz' is relative.

    # We should run this test changing cwd to tmp_path to capture outputs
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        final_state = app.invoke(state)

        assert 'configs' in final_state
        assert 'outfile' in final_state

        train_file = final_state['configs'][0] # Should be train_dft.xyz
        test_file = final_state['outfile'][0] # Should be test_dft.xyz

        assert os.path.exists(train_file)
        assert os.path.exists(test_file)

        # Verify filenames
        assert 'train_dft.xyz' in train_file
        assert 'test_dft.xyz' in test_file

        # Verify split happened
        train_atoms = read(train_file, ':')
        test_atoms = read(test_file, ':')

        assert len(train_atoms) > 0
        assert len(test_atoms) > 0
        assert len(train_atoms) + len(test_atoms) == len(atoms)

    finally:
        os.chdir(cwd)
