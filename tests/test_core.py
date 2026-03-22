from __future__ import annotations

import os

from mlipflow.core.single_point import run_single_point
from mlipflow.strategies.mlip import MACEModel


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
