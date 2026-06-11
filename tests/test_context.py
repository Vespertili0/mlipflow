from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read, write

from mlipflow.data.context import DataManager


def test_datamanager_deprecation_warnings(tmp_path):
    """Test that DataManager and its methods raise deprecation warnings.

    Verifies that DeprecationWarning is correctly raised to alert users
    to migrate away from DataManager.
    """
    # Instantiation warning
    with pytest.warns(DeprecationWarning, match="DataManager is deprecated"):
        dm = DataManager(workdir=str(tmp_path))

    # setup_iteration warning
    with pytest.warns(DeprecationWarning, match="setup_iteration is deprecated"):
        dm.setup_iteration(0)

    # _create_folder_structure warning
    with pytest.warns(
        DeprecationWarning, match="_create_folder_structure is deprecated"
    ):
        dm._create_folder_structure(1)

    # update_training_data warning
    file1 = tmp_path / "f1.xyz"
    file2 = tmp_path / "f2.xyz"
    out_file = tmp_path / "out.xyz"
    write(file1, Atoms("H"))
    write(file2, Atoms("He"))
    with pytest.warns(
        DeprecationWarning, match="DataManager.update_training_data is deprecated"
    ):
        dm.update_training_data(str(file1), str(file2), str(out_file))

    # check_maxforce_and_cleanarrays warning
    in_file = tmp_path / "in.xyz"
    out_clean = tmp_path / "clean.xyz"
    at = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    at.info["DFT_energy"] = -1.0
    import numpy as np

    at.arrays["DFT_forces"] = np.array([[0.0, 0.0, 0.0]])
    at.info["last_op__md_energy"] = -1.0
    at.arrays["last_op__md_forces"] = np.array([[0.0, 0.0, 0.0]])
    at.arrays["numbers"] = np.array([1])
    at.arrays["positions"] = np.array([[0.0, 0.0, 0.0]])
    at.arrays["tags"] = np.array([0])
    write(in_file, at)
    with pytest.warns(
        DeprecationWarning,
        match="DataManager.check_maxforce_and_cleanarrays is deprecated",
    ):
        dm.check_maxforce_and_cleanarrays(
            str(in_file), str(out_clean), "MACE", "md", max_force=5.0
        )

    # merge_clean_chunks warning
    with pytest.warns(
        DeprecationWarning, match="DataManager.merge_clean_chunks is deprecated"
    ):
        from unittest.mock import patch

        with patch("mlipflow.data.processing.OutputSpec"):
            dm.merge_clean_chunks([str(file1), str(file2)], str(out_file))


def test_datamanager_setup_iteration_and_properties(tmp_path):
    """Test setup_iteration directories creation and property pathways.

    Checks that setup_iteration correctly populates properties and folders
    by delegating to path_factory.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dm = DataManager(workdir=str(tmp_path))
        dm.setup_iteration(2)

    assert Path(dm.iter_dir).exists()
    assert Path(dm.MLIP_dir).exists()
    assert Path(dm.SGen_dir).exists()
    assert Path(dm._ensemble_dir).exists()

    assert dm.ensemble_dir == str(tmp_path / "ENSEMBLE")
    assert dm.mlip_dir == str(tmp_path / "iter_2" / "MLIP")

    # Check files mapping values are set correctly
    assert "training" in dm.files
    assert dm.files["training"] == str(tmp_path / "iter_2" / "MLIP" / "training_2.xyz")


def test_datamanager_get_model_name(tmp_path):
    """Test get_model_name configuration matching prefix formats.

    Verifies format matching for model file stems for both MACE and GAP.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dm = DataManager(workdir=str(tmp_path))
        dm.setup_iteration(1)

        # Test MACE model type
        dm.get_model_name(1, "MACE")
        assert dm.files["mlip_model"] == str(
            tmp_path / "iter_1" / "MLIP" / "MACE_1.model"
        )

        # Test GAP model type
        dm.get_model_name(1, "GAP")
        assert dm.files["mlip_model"] == str(tmp_path / "iter_1" / "MLIP" / "GAP_1.xml")


def test_datamanager_initialise_ensembles(tmp_path):
    """Test initialisation of ensembles copy trajectory file.

    Verifies copying of the ensemble traj file and exceptions when missing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dm = DataManager(workdir=str(tmp_path))
        dm.setup_iteration(0)

    # FileNotFoundError test
    non_existent = tmp_path / "non_existent.traj"
    with pytest.raises(FileNotFoundError):
        dm.initialise_ensembles(str(non_existent))

    # Correct copy test
    src_traj = tmp_path / "source.traj"
    write(src_traj, [Atoms("H"), Atoms("He")])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dm.initialise_ensembles(str(src_traj))

    assert Path(dm.files["ensemble_traj"]).exists()
    assert Path(dm.files["ensemble_xyz"]).exists()

    # Read back and verify
    atoms = read(dm.files["ensemble_xyz"], ":")
    assert len(atoms) == 2


def test_datamanager_move_mace_model_file(tmp_path):
    """Test copying of compiled model to the final destination.

    Verifies that the compiled model is successfully copied from its build location.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dm = DataManager(workdir=str(tmp_path))
        dm.setup_iteration(0)

    # Create dummy model file inside a nested MACE_model folder
    model_dir = Path(dm.MLIP_dir) / "MACE_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    compiled_model = model_dir / "my_pot_compiled.model"
    compiled_model.write_text("dummy model content")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dm.move_mace_model_file("my_pot")

    dest_model = Path(dm.MLIP_dir) / "my_pot.model"
    assert dest_model.exists()
    assert dest_model.read_text() == "dummy model content"


def test_datamanager_update_configset_tag(tmp_path):
    """Test updating wfl configset tag fields.

    Verifies updating tags for an active OutputSpec.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dm = DataManager(workdir=str(tmp_path))

    in_file = tmp_path / "tag_in.xyz"
    out_file = tmp_path / "tag_out.xyz"
    write(in_file, [Atoms("H"), Atoms("He")])

    tag_dict = {"my_custom_tag": "test_val"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dm.update_configset_tag(str(in_file), str(out_file), tag_dict)

    atoms = read(out_file, ":")
    assert len(atoms) == 2
    assert atoms[0].info["my_custom_tag"] == "test_val"
    assert atoms[1].info["my_custom_tag"] == "test_val"


def test_datamanager_filter_info_dict(tmp_path):
    """Test filtering dictionary entries."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dm = DataManager(workdir=str(tmp_path))

    info = {"keep_this": 1, "drop_this": 2, "another_keep": 3}
    filtered = dm.filter_info_dict(info, ["keep_this", "another_keep"])
    assert filtered == {"keep_this": 1, "another_keep": 3}


def test_datamanager_split_success_failed_configs(tmp_path):
    """Test success and failure config splits."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dm = DataManager(workdir=str(tmp_path))

    a1 = Atoms("H")
    a1.info["energy"] = -1.0
    a2 = Atoms("He")
    # Missing energy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        success, failed = dm.split_success_failed_configs([a1, a2], key="energy")

    assert len(success) == 1
    assert len(failed) == 1
    assert success[0] == a1
    assert failed[0] == a2


def test_datamanager_clean_up(tmp_path):
    """Test clean_up removes chunk files and directories.

    Verifies directory scanning and clean deletions of temporary workspace chunks.
    """
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dm = DataManager(workdir=str(tmp_path))

        # Create dummy chunk folder and file
        chunk_dir = tmp_path / "run_chunk_123"
        chunk_dir.mkdir()
        chunk_file = tmp_path / "test_chunk_abc"
        chunk_file.touch()

        # Create a non-chunk file that should NOT be deleted
        safe_file = tmp_path / "safe_file"
        safe_file.touch()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dm.clean_up(key="_chunk_")

        assert not chunk_dir.exists()
        assert not chunk_file.exists()
        assert safe_file.exists()
    finally:
        os.chdir(cwd)
