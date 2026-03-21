from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from mlipflow.data.selection import (
    select_by_uncertainty,
    split_configset_by_force_agreement,
)
from mlipflow.strategies.mlip import MACEModel


def test_split_configset_by_force_agreement(tmp_path):
    """Test split_configset_by_force_agreement."""
    # Create 100 dummy atoms
    atoms_list = []
    n_total = 100
    for i in range(n_total):
        at = Atoms("H", positions=[[0, 0, 0]])
        # DFT forces: all zeros
        at.arrays["DFT_forces"] = np.array([[0.0, 0.0, 0.0]])
        # MACE forces: varied error.
        # For i < 80, error is small (0.1)
        # For i >= 80, error is large (10.0)
        # So top 20% should be the ones with index >= 80.
        error = 10.0 if i >= 80 else 0.1
        at.arrays["MACE_forces"] = np.array([[error, 0.0, 0.0]])
        at.info["index"] = i
        atoms_list.append(at)

    # The function writes to f'{suffix}_{out_file}'
    # If out_file is absolute path /tmp/.../split.xyz
    # It writes to train_/tmp/.../split.xyz which is invalid path.
    # So out_file should be a filename, and we should control CWD or pass just filename and expect it in CWD.

    # Let's inspect the code again.
    # write(f'{suffix}_{out_file}', atoms)
    # If out_file is 'split.xyz', it writes 'train_split.xyz'.

    # I should change directory to tmp_path to make it safe.
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        write("test_selection.xyz", atoms_list)

        split_configset_by_force_agreement(
            in_file="test_selection.xyz",
            out_file="split.xyz",
            pair_tuple=("DFT_", "MACE_"),
            main_suffix="train",
            side_suffix="test",
        )

        train_file = "train_split.xyz"
        test_file = "test_split.xyz"

        assert Path(train_file).exists()
        assert Path(test_file).exists()

        train_atoms = read(train_file, ":")
        test_atoms = read(test_file, ":")

        # Logic:
        # Top 20% (highest MAE) -> 20 atoms (indices 80-99). These MUST be in train.
        # From bottom 80% (indices 0-79), select 60% of TOTAL (60 atoms).
        # Total train size = 20 + 60 = 80 atoms.
        # Total test size = 20 atoms.

        assert len(train_atoms) == 80
        assert len(test_atoms) == 20

        # Verify that all top 20% are in train
        train_indices = [at.info["index"] for at in train_atoms]
        for i in range(80, 100):
            assert i in train_indices

    finally:
        os.chdir(cwd)


def test_select_by_uncertainty(tmp_path):
    """Test GMM-driven selection of uncertain structures."""
    test_dir = Path(__file__).resolve().parent
    mlip_name = str(Path(test_dir) / "data" / "mace_test")
    test_data = str(Path(test_dir) / "data" / "test_data.xyz")

    # Initialise MACEModel strategy
    mace_model = MACEModel(mlip_name=mlip_name, run_mode="local")

    out_file = tmp_path / "selected.xyz"

    # Run the uncertainty selection using test data as both train and pool
    # We use very few GMM components and initialisations to keep the test fast
    select_by_uncertainty(
        train_file=test_data,
        pool_file=test_data,
        out_file=str(out_file),
        mlip_strategy=mace_model,
        certainty_threshold=0.8,  # The top 20% most uncertain will be selected
        pca_variance_threshold=0.95,
        max_gmm_components=2,
        gmm_n_init=1,
        device="cpu",
    )

    assert Path(out_file).exists()
    selected = read(out_file, ":")

    # The output should contain some selected structures
    assert len(selected) > 0
