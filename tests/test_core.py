from __future__ import annotations

import os

from mlipflow.core.md_nvt import _get_temperature
from mlipflow.core.single_point import run_single_point
from mlipflow.strategies.mlip import MACEModel


def test_get_temperature_profiles():
    """Verify _get_temperature handles scalars, list/tuple ramps, and stage dicts."""
    # Scalar constant temperature
    res_scalar = _get_temperature(300.0, temperature_tau=100.0, steps=100)
    assert len(res_scalar) == 1
    assert res_scalar[0]["T_i"] == 300.0
    assert res_scalar[0]["T_f"] == 300.0

    # List / tuple ramp must not be double-wrapped
    res_ramp = _get_temperature([300.0, 600.0], temperature_tau=100.0, steps=100)
    assert len(res_ramp) == 1
    assert res_ramp[0]["T_i"] == 300.0
    assert res_ramp[0]["T_f"] == 600.0
    assert res_ramp[0]["n_stages"] == 10

    res_tuple = _get_temperature((300.0, 600.0, 5), temperature_tau=100.0, steps=100)
    assert res_tuple[0]["T_i"] == 300.0
    assert res_tuple[0]["T_f"] == 600.0
    assert res_tuple[0]["n_stages"] == 5

    # Single dict stage should be wrapped into a 1-element list
    single_dict = {"T_i": 300.0, "T_f": 500.0, "traj_frac": 1.0}
    res_dict = _get_temperature(single_dict, temperature_tau=100.0, steps=100)
    assert isinstance(res_dict, list)
    assert len(res_dict) == 1
    assert res_dict[0]["T_i"] == 300.0
    assert res_dict[0]["n_stages"] == 10


def test_mace_for_single_point_energy(tmp_path):
    """Test MACEModel's single_point_energy method."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(test_dir, "data", "test_data.xyz")
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace_model = MACEModel(mlip_name=mlip_name, run_mode="local")
    out_file = tmp_path / "test_output.xyz"
    run_single_point(
        in_file=in_file,
        out_file=str(out_file),
        output_prefix=mace_model.mlip_prefix,
        calculator=mace_model.get_calculator(job_name="MP_"),
    )
    assert out_file.exists()
