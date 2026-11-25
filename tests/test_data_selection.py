from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from mlipflow.data.selection import select_by_uncertainty, split_configset
from mlipflow.strategies.mlip import MACEModel


def test_split_configset_stratified_real_data(tmp_path):
    """Test stratified_random splitting on real multi-species MD dataset."""
    test_dir = Path(__file__).resolve().parent
    src_xyz = test_dir / "data" / "test_data_II.xyz"
    atoms = read(src_xyz, ":")
    for idx, at in enumerate(atoms):
        at.info["frame_id"] = idx
    in_xyz = tmp_path / "md_pool.xyz"
    write(in_xyz, atoms)

    out_1 = {
        "train": str(tmp_path / "train_1.xyz"),
        "test": str(tmp_path / "test_1.xyz"),
    }
    out_2 = {
        "train": str(tmp_path / "train_2.xyz"),
        "test": str(tmp_path / "test_2.xyz"),
    }
    out_diff_seed = {
        "train": str(tmp_path / "train_3.xyz"),
        "test": str(tmp_path / "test_3.xyz"),
    }

    split_configset(
        in_file=str(in_xyz),
        out_file=out_1,
        split_mode="stratified_random",
        test_fraction=0.2,
        seed=42,
        clean_data=False,
    )
    split_configset(
        in_file=str(in_xyz),
        out_file=out_2,
        split_mode="stratified_random",
        test_fraction=0.2,
        seed=42,
        clean_data=False,
    )
    split_configset(
        in_file=str(in_xyz),
        out_file=out_diff_seed,
        split_mode="stratified_random",
        test_fraction=0.2,
        seed=99,
        clean_data=False,
    )

    train_1 = read(out_1["train"], ":")
    test_1 = read(out_1["test"], ":")
    test_2 = read(out_2["test"], ":")
    test_diff = read(out_diff_seed["test"], ":")

    assert len(train_1) == 272
    assert len(test_1) == 68

    train_ids = {at.info["frame_id"] for at in train_1}
    test_ids = {at.info["frame_id"] for at in test_1}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids | test_ids) == len(atoms)

    # Reproducible for fixed seed, distinct for a different seed
    assert [at.info["frame_id"] for at in test_1] == [
        at.info["frame_id"] for at in test_2
    ]
    assert [at.info["frame_id"] for at in test_1] != [
        at.info["frame_id"] for at in test_diff
    ]

    # Species proportions preserved within ±1 of exact 20% per stratum
    total_counts = Counter(at.info["species"] for at in atoms)
    test_counts = Counter(at.info["species"] for at in test_1)
    for species, total_n in total_counts.items():
        expected = total_n * 0.2
        assert abs(test_counts[species] - expected) <= 1.0


def test_split_configset_singleton_species_kept_in_train(tmp_path):
    """Test that singleton species in real test_data.xyz remain in training set."""
    test_dir = Path(__file__).resolve().parent
    src_xyz = test_dir / "data" / "test_data.xyz"
    atoms = read(src_xyz, ":")

    out_files = {
        "train": str(tmp_path / "train.xyz"),
        "test": str(tmp_path / "test.xyz"),
    }
    split_configset(
        in_file=str(src_xyz),
        out_file=out_files,
        split_mode="stratified_random",
        test_fraction=0.2,
        seed=42,
        clean_data=False,
    )

    train_atoms = read(out_files["train"], ":")
    test_atoms = read(out_files["test"], ":")

    assert len(train_atoms) == 14
    assert len(test_atoms) == 3
    assert len(train_atoms) + len(test_atoms) == len(atoms)

    # FA has only 1 frame in test_data.xyz and must stay in train
    assert any(at.info.get("species") == "FA" for at in train_atoms)
    assert all(at.info.get("species") != "FA" for at in test_atoms)


@pytest.mark.parametrize("split_mode", ["random", "stratified_random"])
def test_split_configset_preserves_force_error_tail(tmp_path, split_mode):
    """Verify unbiased splitting preserves the upper force-error tail in dft_test."""
    atoms_list = []
    n_total = 100
    for i in range(n_total):
        at = Atoms("H", positions=[[0, 0, 0]])
        at.arrays["DFT_forces"] = np.array([[0.0, 0.0, 0.0]])
        error = 10.0 if i >= 80 else 0.1
        at.arrays["MACE_forces"] = np.array([[error, 0.0, 0.0]])
        at.info["DFT_energy"] = 0.0
        at.info["species"] = "sp_A" if i % 2 == 0 else "sp_B"
        at.info["index"] = i
        atoms_list.append(at)

    in_file = tmp_path / "tail_check.xyz"
    write(in_file, atoms_list)

    out_a = {
        "train": str(tmp_path / f"train_a_{split_mode}.xyz"),
        "test": str(tmp_path / f"test_a_{split_mode}.xyz"),
    }
    out_b = {
        "train": str(tmp_path / f"train_b_{split_mode}.xyz"),
        "test": str(tmp_path / f"test_b_{split_mode}.xyz"),
    }

    split_configset(
        in_file=str(in_file),
        out_file=out_a,
        split_mode=split_mode,
        test_fraction=0.2,
        seed=123,
    )
    split_configset(
        in_file=str(in_file),
        out_file=out_b,
        split_mode=split_mode,
        test_fraction=0.2,
        seed=123,
    )

    train_atoms = read(out_a["train"], ":")
    test_atoms = read(out_a["test"], ":")
    test_atoms_b = read(out_b["test"], ":")

    assert len(train_atoms) == 80
    assert len(test_atoms) == 20

    train_indices = {at.info["index"] for at in train_atoms}
    test_indices = {at.info["index"] for at in test_atoms}
    assert train_indices.isdisjoint(test_indices)
    assert len(train_indices | test_indices) == n_total

    # Upper force-error tail (indices 80..99) must NOT be truncated out of test
    tail_in_test = [idx for idx in test_indices if idx >= 80]
    assert len(tail_in_test) > 0

    # Fixed-seed reproducibility
    assert [at.info["index"] for at in test_atoms] == [
        at.info["index"] for at in test_atoms_b
    ]


def test_split_configset_missing_species_and_validation(tmp_path):
    """Test fallback when species key is missing and validation of invalid parameters."""
    atoms_list = []
    for i in range(20):
        at = Atoms("H", positions=[[0, 0, 0]])
        at.info["DFT_energy"] = 0.0
        at.info["index"] = i
        atoms_list.append(at)

    in_file = tmp_path / "no_species.xyz"
    write(in_file, atoms_list)

    # Legacy string out_file format still works in tmp_path
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        split_configset(
            in_file=str(in_file),
            out_file="split.xyz",
            split_mode="stratified_random",
            test_fraction=0.25,
            seed=7,
        )
        assert Path("train_split.xyz").exists()
        assert Path("test_split.xyz").exists()
        assert len(read("train_split.xyz", ":")) == 15
        assert len(read("test_split.xyz", ":")) == 5
    finally:
        os.chdir(cwd)

    with pytest.raises(ValueError, match="split_mode"):
        split_configset(
            in_file=str(in_file),
            out_file={
                "train": str(tmp_path / "t.xyz"),
                "test": str(tmp_path / "v.xyz"),
            },
            split_mode="force_agreement",
        )

    with pytest.raises(ValueError, match="test_fraction"):
        split_configset(
            in_file=str(in_file),
            out_file={
                "train": str(tmp_path / "t.xyz"),
                "test": str(tmp_path / "v.xyz"),
            },
            test_fraction=1.0,
        )


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
