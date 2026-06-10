from __future__ import annotations

from pathlib import Path

from mlipflow.utils.path_factory import create_iteration_directory, resolve_step_path


def test_resolve_step_path_basic(tmp_path):
    """Test resolve_step_path with basic input files and directories.

    This ensures that paths are resolved deterministically and parent
    directories are created.
    """
    input_file = "test.xyz"
    workdir = tmp_path / "run_dir"

    # Check simple path resolution
    resolved = resolve_step_path(
        input_file=input_file, step_suffix="dft_sp", iteration=2, workdir=workdir
    )

    expected_path = workdir / "iter_2" / "test_dft_sp.xyz"
    assert resolved == str(expected_path)
    assert (workdir / "iter_2").exists()


def test_resolve_step_path_tag_stripping(tmp_path):
    """Test resolve_step_path strips loop designations from stems correctly.

    This verifies that internal workflow loop labels are removed to prevent
    recursive path nesting.
    """
    workdir = tmp_path / "run_dir"

    tags = [".cleaned", ".dft", ".mace", "_dft_sp", "_mace_sp", "_md", "_opt", "_neb"]

    for tag in tags:
        input_file = f"molecule{tag}.xyz"
        resolved = resolve_step_path(
            input_file=input_file, step_suffix="opt", iteration=1, workdir=workdir
        )
        expected_path = workdir / "iter_1" / "molecule_opt.xyz"
        assert resolved == str(expected_path)


def test_resolve_step_path_multiple_tags(tmp_path):
    """Test resolve_step_path strips multiple tags or first occurrence.

    Verifies that multiple tags in a single filename are correctly cleaned
    back to the original base stem.
    """
    workdir = tmp_path / "run_dir"
    input_file = "molecule_md_dft_sp.xyz"
    resolved = resolve_step_path(
        input_file=input_file, step_suffix="opt", iteration=0, workdir=workdir
    )
    expected_path = workdir / "iter_0" / "molecule_opt.xyz"
    assert resolved == str(expected_path)


def test_create_iteration_directory(tmp_path):
    """Test create_iteration_directory creates correct folder structures.

    Verifies that all required folders under the iteration are created and
    the returned dictionary maps them correctly.
    """
    workdir = tmp_path / "run_dir"

    dirs = create_iteration_directory(iteration=3, workdir=workdir)

    expected_keys = {"iter_dir", "mlip_dir", "sgen_dir", "ensemble_dir"}
    assert set(dirs.keys()) == expected_keys

    assert dirs["iter_dir"] == str(workdir / "iter_3")
    assert dirs["mlip_dir"] == str(workdir / "iter_3" / "MLIP")
    assert dirs["sgen_dir"] == str(workdir / "iter_3" / "SGEN")
    assert dirs["ensemble_dir"] == str(workdir / "ENSEMBLE")

    for path_str in dirs.values():
        assert Path(path_str).exists()
