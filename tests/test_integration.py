from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase.io import read, write

from mlipflow.graphflow.graphs import execute_dft_single_point_block
from mlipflow.graphflow.nodes import EnsembleState, run_dft_sp
from mlipflow.strategies.dft import EMTCalc


@pytest.fixture
def real_data_setup(tmp_path):
    """
    Setup fixture that copies real data to a temporary directory.
    Returns the path to the config file and the state.
    """
    test_dir = Path(__file__).resolve().parent
    src_xyz = str(Path(test_dir) / "data" / "test_data.xyz")

    # We need a model file for MACE strategy even if we mock the strategy itself
    # just to pass some checks if any. But here we primarily test DFT part.

    config_file = tmp_path / "test_data.xyz"
    shutil.copy2(src_xyz, config_file)

    # Setup strategies
    qchem_strategy = EMTCalc()

    # Mock MLIP strategy just to provide the prefix
    mlip_strategy = MagicMock()
    mlip_strategy.mlip_prefix = "MACE_"

    state = EnsembleState(
        configs=[str(config_file)],
        qchem_strategy=qchem_strategy,
        mlip_strategy=mlip_strategy,
    )
    return state, tmp_path


def test_run_dft_sp_with_emt(real_data_setup):
    """
    Test run_dft_sp using EMTCalc and real data.
    This verifies that run_single_point actually works with the strategy and data.
    """
    state, _tmp_path = real_data_setup

    # Execute the node
    new_state = run_dft_sp(state)

    # Verify output file exists
    output_file = new_state["configs"][0]
    assert Path(output_file).exists()
    assert output_file.endswith(".dft.xyz")

    # Verify content
    atoms = read(output_file, ":")
    assert len(atoms) > 0
    # Check if results are present. EMTCalc uses "DFT_" prefix in QChemStrategy?
    # No, EMTCalc inherits QChemStrategy which sets qe_prefix = 'DFT_'
    # generic_calc with output_prefix='DFT_' should store results in info/arrays with that prefix.
    # For 'energy', it should be 'DFT_energy'.

    # EMT calculator might write 'energy' directly to atoms.calc.results,
    # but generic_calc + output_prefix handles the storage.
    # Let's check info keys
    assert "DFT_energy" in atoms[0].info
    assert "DFT_forces" in atoms[0].arrays


def test_execute_dft_single_point_block_integration(real_data_setup):
    """
    Test the full DFT block execution with EMTCalc.
    Flow: dft_sp -> assess_n_select
    """
    state, tmp_path = real_data_setup

    # Let's update the input file with dummy MACE forces.
    atoms = read(state["configs"][0], ":")

    rng = np.random.default_rng()
    for at in atoms:
        at.arrays["MACE_forces"] = rng.random((len(at), 3))
    write(state["configs"][0], atoms)

    # Initialise graph
    app = execute_dft_single_point_block()

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        final_state = app.invoke(state)

        assert "configs" in final_state
        assert "outfile" in final_state

        train_file = final_state["configs"][0]  # Should be train_dft.xyz
        test_file = final_state["outfile"][0]  # Should be test_dft.xyz

        assert Path(train_file).exists()
        assert Path(test_file).exists()

        # Verify filenames
        assert "train_dft.xyz" in train_file
        assert "test_dft.xyz" in test_file

        # Verify split happened
        train_atoms = read(train_file, ":")
        test_atoms = read(test_file, ":")

        assert len(train_atoms) > 0
        assert len(test_atoms) > 0
        assert len(train_atoms) + len(test_atoms) == len(atoms)

    finally:
        os.chdir(cwd)
