import os
import shutil
import pytest
import numpy as np
from unittest.mock import MagicMock
from ase.io import read, write
from mlipflow.graphflow.nodes import run_config_fps_selection, EnsembleState

def test_run_config_fps_selection(tmp_path):
    """
    Test run_config_fps_selection using real data and dummy info fields.
    """
    # Setup paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    src_xyz = os.path.join(test_dir, 'data', 'test_data.xyz')
    config_file = tmp_path / "test_data.xyz"
    shutil.copy2(src_xyz, config_file)

    # Add dummy energy info to the configurations
    atoms = read(config_file, ':')
    rng = np.random.default_rng(42)
    new_atoms_list = []
    for at in atoms:
        # Filter to keep only Cu atoms to avoid descriptor mismatch with simple Z=29 descriptor
        # The wfl/quippy wrapper expects descriptor arrays to match atom count if applied globally
        at = at[at.numbers == 29]
        at.info['dummy_energy'] = rng.random()
        new_atoms_list.append(at)
    write(config_file, new_atoms_list)

    # Change directory to tmp_path to capture outputs
    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Setup EnsembleState
        # Valid descriptors for Cu (Z=29)
        descriptor_string = "soap cutoff=4.0 l_max=4 n_max=4 atom_sigma=0.5 n_Z=1 Z={29} n_species=1 species_Z={29}"

        calculation_kwargs = {
            'selection': {
                'descriptor_string': descriptor_string,
                'descriptor_key': 'SOAP',
                'info_field': 'dummy_energy',
                'n_optimal': 5,
                'seed': 42
            }
        }

        # Mock strategies as they are required keys in EnsembleState TypedDict
        qchem_mock = MagicMock()
        mlip_mock = MagicMock()

        state = EnsembleState(
            configs=[str(config_file)],
            qchem_strategy=qchem_mock,
            mlip_strategy=mlip_mock,
            calculation_kwargs=calculation_kwargs
        )

        # Run selection
        result = run_config_fps_selection(state)

        # Verify output
        output_configs = result['configs']
        assert len(output_configs) == 1
        output_file = output_configs[0]

        assert os.path.exists(output_file)

        # Verify content
        selected_atoms = read(output_file, ':')
        assert len(selected_atoms) == 5

        # Verify descriptors are present (ConfigurationSelector adds them)
        assert 'SOAP' in selected_atoms[0].info

    finally:
        os.chdir(cwd)
