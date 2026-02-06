import pytest
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock
from mlipflow.adapters.wflio import ForceCheck, select_config

def test_force_check():
    abort = ForceCheck(threshold=1.0, n_failed_steps=2)

    # Mock atoms with calculator
    at = Atoms('H')
    calc = MagicMock()
    at.calc = calc

    # Case 1: Forces below threshold
    calc.get_forces.return_value = np.array([[0.5, 0.0, 0.0]])
    assert bool(abort.atoms_ok(at)) is True

    # Case 2: Forces above threshold
    calc.get_forces.return_value = np.array([[1.5, 0.0, 0.0]])
    assert bool(abort.atoms_ok(at)) is False

def test_select_config():
    # Helper to create list of dummy objects
    traj = [i for i in range(20)]

    # Case < 10
    short_traj = traj[:5]
    assert select_config(short_traj) == [4]

    # Case == 500
    long_traj = [i for i in range(500)]
    assert select_config(long_traj) == [499]

    # Case else (e.g. 20)
    assert select_config(traj) == [10]
