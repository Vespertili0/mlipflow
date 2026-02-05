import os
import pytest
import numpy as np
from ase import Atoms
from ase.io import write, read
from mlipflow.data.processing import check_maxforce_and_cleanarrays, split_success_failed_configs

def test_check_maxforce_and_cleanarrays(tmp_path):
    """Test check_maxforce_and_cleanarrays."""
    # Create dummy atoms
    atoms_low = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
    atoms_low.info['DFT_energy'] = -10.0
    atoms_low.arrays['DFT_forces'] = np.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]])
    # Add dummy keys that are expected to be present or filtered
    atoms_low.arrays['last_op__md_forces'] = np.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]])
    atoms_low.info['last_op__md_energy'] = -10.0
    atoms_low.set_tags([0, 0])

    atoms_high = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
    atoms_high.info['DFT_energy'] = -5.0
    atoms_high.set_tags([0, 0])
    atoms_high.arrays['DFT_forces'] = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    atoms_high.arrays['last_op__md_forces'] = np.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]])
    atoms_high.info['last_op__md_energy'] = -5.0

    in_file = tmp_path / "test_maxforce.xyz"
    out_file = tmp_path / "cleaned.xyz"

    write(in_file, [atoms_low, atoms_high])

    check_maxforce_and_cleanarrays(
        in_file=str(in_file),
        out_file=str(out_file),
        mlip_prefix='MACE',
        calc='md',
        max_force=10.0
    )

    cleaned_atoms = read(str(out_file), ':')
    assert len(cleaned_atoms) == 1
    assert cleaned_atoms[0].info['DFT_energy'] == -10.0
    # Check if keys are renamed
    assert 'MACE_forces' in cleaned_atoms[0].arrays
    assert 'MACE_energy' in cleaned_atoms[0].info


def test_split_success_failed_configs():
    """Test split_success_failed_configs."""
    atoms_success = Atoms('H')
    atoms_success.info['DFT_energy'] = -1.0

    atoms_failed = Atoms('H')
    # Missing DFT_energy

    configs = [atoms_success, atoms_failed, atoms_success]

    success, failed = split_success_failed_configs(configs, key='DFT_energy')

    assert len(success) == 2
    assert len(failed) == 1
    assert success[0] == atoms_success
    assert failed[0] == atoms_failed
