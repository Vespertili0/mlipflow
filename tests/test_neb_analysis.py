from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from wfl.configset import ConfigSet

from mlipflow.core import NEBAnalysis


def _make_image(
    energy: float, forces_scale: float = 1.0, fix_first: bool = False
) -> Atoms:
    """Return a 3-atom H2O-like Atoms with prefixed energy/forces populated."""
    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]])
    atoms = Atoms("H2O", positions=positions)
    atoms.info["species"] = "H2O"
    atoms.info["MACE_energy"] = energy
    atoms.info["DFT_energy"] = energy * 1.05
    rng = np.random.default_rng(seed=42)
    f = rng.standard_normal((3, 3)) * forces_scale
    atoms.arrays["MACE_forces"] = f
    atoms.arrays["DFT_forces"] = f * 1.05
    if fix_first:
        atoms.set_constraint(FixAtoms(indices=[0]))
    return atoms


@pytest.fixture
def single_path() -> list[Atoms]:
    """A minimal 5-image NEB path with a simple energy barrier."""
    energies = [0.0, 0.5, 1.0, 0.5, 0.1]
    return [_make_image(e, forces_scale=0.3) for e in energies]


@pytest.fixture
def two_path_images() -> list[Atoms]:
    """Ten images forming two 5-image paths for the same reaction."""
    energies = [0.0, 0.5, 1.0, 0.5, 0.1, 0.0, 0.4, 0.9, 0.3, 0.05]
    return [_make_image(e, forces_scale=0.3) for e in energies]


@pytest.fixture
def neb(single_path) -> NEBAnalysis:
    return NEBAnalysis(images=single_path)


@pytest.fixture
def neb_two_paths(two_path_images) -> NEBAnalysis:
    return NEBAnalysis(images=two_path_images, n_frames=5)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.mark.unit
def test_init_single_path(neb):
    assert len(neb.paths) == 1
    assert len(neb.reaction_groups) == 1


@pytest.mark.unit
def test_init_multi_path(neb_two_paths):
    assert len(neb_two_paths.paths) == 2


@pytest.mark.unit
def test_init_invalid_n_frames(single_path):
    with pytest.raises(ValueError, match="is not a multiple"):
        NEBAnalysis(images=single_path, n_frames=3)


@pytest.mark.unit
def test_from_file_configset(single_path):
    cs = ConfigSet(single_path)
    neb_from_cs = NEBAnalysis.from_file(cs)
    assert len(neb_from_cs.paths) == 1


@pytest.mark.unit
def test_from_file_xyz(single_path, tmp_path):
    from ase.io import write

    filepath = tmp_path / "test.xyz"
    write(filepath, single_path, format="extxyz")
    neb_from_xyz = NEBAnalysis.from_file(filepath)
    assert len(neb_from_xyz.paths) == 1


@pytest.mark.unit
def test_from_file_missing():
    with pytest.raises(FileNotFoundError):
        NEBAnalysis.from_file("/nonexistent/file.xyz")


@pytest.mark.unit
def test_from_file_type_error():
    with pytest.raises(TypeError):
        NEBAnalysis.from_file(12345)


@pytest.mark.unit
def test_get_force_components_shape(neb, single_path):
    fmax_list, f_para, f_perp = neb.get_force_components(single_path, prefix="MACE")
    n = len(single_path)
    assert len(fmax_list) == n
    assert len(f_para) == n
    assert len(f_perp) == n


@pytest.mark.unit
def test_get_force_components_too_short(single_path):
    with pytest.raises(ValueError, match="at least 3 images"):
        NEBAnalysis(single_path).get_force_components(single_path[:2], prefix="MACE")


@pytest.mark.unit
def test_get_barrier_returns_floats(neb):
    bf, br, de = neb.get_barrier(path_index=0, prefix="MACE")
    # get_barrier returns numpy scalars from fit_raw; accept both float and np.floating
    assert isinstance(bf, (float, np.floating))
    assert isinstance(br, (float, np.floating))
    assert isinstance(de, (float, np.floating))


@pytest.mark.unit
def test_get_reaction_string(neb):
    rs = neb.get_reaction_string(path_index=0)
    assert "->" in rs or " -> " in rs


@pytest.mark.unit
def test_plot_band_no_show(neb):
    with patch("matplotlib.pyplot.show") as mock_show:
        ax = neb.plot_band()
        assert ax is not None
        mock_show.assert_not_called()


@pytest.mark.unit
def test_plot_comparison_no_show(neb):
    with patch("matplotlib.pyplot.show") as mock_show:
        res = neb.plot_comparison()
        assert res is not None
        mock_show.assert_not_called()


@pytest.mark.unit
def test_plot_multiple_pathways(neb_two_paths):
    res = neb_two_paths.plot_multiple_pathways()
    assert len(res) == 1
    key = next(iter(res.keys()))
    assert isinstance(res[key], tuple)
    assert len(res[key]) == 2
    assert hasattr(neb_two_paths, "mep_indices")
    assert key in neb_two_paths.mep_indices


@pytest.mark.unit
def test_plot_mep_pathways_requires_prior_call(neb_two_paths):
    with pytest.raises(ValueError, match="MEP indices not found"):
        neb_two_paths.plot_mep_pathways()


@pytest.mark.unit
def test_plot_fmax_distribution(neb):
    with patch("matplotlib.pyplot.show") as mock_show:
        fig, axes = neb.plot_fmax_distribution()
        assert fig is not None
        assert len(axes) == 2
        mock_show.assert_not_called()


@pytest.mark.unit
def test_save_plots(neb, tmp_path):
    saved = neb.save_plots(tmp_path)
    assert "multiple_pathways" in saved
    assert "mep_pathways" in saved
    assert "fmax_distribution" in saved
    for paths in saved.values():
        assert len(paths) > 0
        for p in paths:
            assert Path(p).exists()


@pytest.mark.unit
def test_save_plots_creates_dir(neb, tmp_path):
    new_dir = tmp_path / "new_dir"
    neb.save_plots(new_dir)
    assert new_dir.exists()


@pytest.mark.unit
def test_constraint_aware_fmax():
    img1 = _make_image(0.0, fix_first=True)
    img2 = _make_image(0.5, fix_first=True)
    img3 = _make_image(1.0, fix_first=True)
    images = [img1, img2, img3]
    neb_c = NEBAnalysis(images)
    fmax_list, _, _ = neb_c.get_force_components(images, prefix="MACE")
    assert len(fmax_list) == 3


@pytest.mark.unit
def test_no_print_statements(neb, single_path):
    with patch.object(builtins, "print") as mock_print:
        # Pass ax to analyse_neb_force_decomposition to ensure it generates plot
        neb.analyse_neb_force_decomposition(single_path, ax=plt.subplots()[1])
        neb.plot_fmax_distribution()
        mock_print.assert_not_called()
