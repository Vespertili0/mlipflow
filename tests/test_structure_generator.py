from __future__ import annotations

import os
from unittest.mock import MagicMock

from mlipflow.strategies.dft import EMTCalc
from mlipflow.strategies.mlip import MACEModel
from mlipflow.strategies.structure_generators import OPTGen


def test_optgen_run(tmp_path):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(test_dir, "data", "test_data.xyz")
    mlip_name = os.path.join(test_dir, "data", "mace_test")
    mace = MACEModel(mlip_name=mlip_name, run_mode="local")

    # Mock get_calculator to return EMT calculator to avoid MACE execution errors
    mace.get_calculator = MagicMock(side_effect=EMTCalc().get_calculator)
    mace.remote_info = None

    out_file = tmp_path / "opt_test.xyz"
    # Use traj_subselect=None to ensure output is written even if not converged
    OPTGen(
        params={"fmax": 5.0, "steps": 2}, traj_subselect=None
    ).generate_new_structures(
        in_file=in_file,
        out_file=str(out_file),
        calculator=mace.get_calculator(job_name="mMP_"),
        remote_info=mace.remote_info,
    )
    assert out_file.exists()
