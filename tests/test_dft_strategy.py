from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ase.calculators.emt import EMT
from wfl.calculators.espresso import Espresso

from mlipflow.strategies.dft import EMTCalc, QECalculator


def test_emt_calc_get_calculator():
    strategy = EMTCalc()
    calc_class, args, kwargs = strategy.get_calculator(job_name="test")
    assert calc_class == EMT
    assert args is None
    assert kwargs == {"fixed_cutoff": True}
    assert strategy.remote_info is None

def test_qe_calculator_init(tmp_path):
    # Create dummy basic params file
    params_file = tmp_path / "basic_params.json"
    params_content = {
        "control": {
            "nstep": 100,
            "etot_conv_thr": 1e-4,
            "forc_conv_thr": 1e-3,
            "dipfield": True,
            "tefield": True
        },
        "system": {
            "eamp": 0.0,
            "edir": 1,
            "emaxpos": 0.5,
            "eopreg": 0.1,
            "dftd3_version": 3,
            "vdw_corr": "dft-d3"
        },
        "electrons": {}
    }

    params_file.write_text(json.dumps(params_content))

    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()

    strategy = QECalculator(basic_params=str(params_file), pseudopots={}, pseudo_dir=str(pseudo_dir))
    assert strategy.basic_params == str(params_file)
    assert strategy.pseudo_dir == str(pseudo_dir)
    assert strategy.qe_prefix == "DFT_"

@patch("mlipflow.strategies.dft.prepare_remote")
def test_qe_calculator_get_calculator(mock_prepare_remote, tmp_path):
    # Setup
    params_file = tmp_path / "basic_params.json"
    params_content = {
        "control": {
            "nstep": 100,
            "etot_conv_thr": 1e-4,
            "forc_conv_thr": 1e-3,
            "dipfield": True,
            "tefield": True
        },
        "system": {
            "eamp": 0.0,
            "edir": 1,
            "emaxpos": 0.5,
            "eopreg": 0.1,
            "dftd3_version": 3,
            "vdw_corr": "dft-d3"
        },
        "electrons": {}
    }

    params_file.write_text(json.dumps(params_content))
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()

    strategy = QECalculator(basic_params=str(params_file), pseudopots={}, pseudo_dir=str(pseudo_dir))
    mock_prepare_remote.return_value = MagicMock()

    # Test scf
    calc_class, args, kwargs = strategy.get_calculator(job_name="test_job", calc_type="scf")

    assert calc_class == Espresso
    assert strategy.remote_info is not None
    assert kwargs["rundir_prefix"] == "QE_"
    assert kwargs["input_data"]["control"]["calculation"] == "scf"
    assert kwargs["input_data"]["system"]["ecutwfc"] == 64.97 # Based on default logic

    # Check remote settings
    mock_prepare_remote.assert_called()
    call_kwargs = mock_prepare_remote.call_args[1]
    assert call_kwargs["max_time"] == "01:25:00"
    assert call_kwargs["job_name"] == "test_job"

@patch("mlipflow.strategies.dft.prepare_remote")
def test_qe_calculator_get_calculator_relax(mock_prepare_remote, tmp_path):
    # Setup
    params_file = tmp_path / "basic_params.json"
    params_content = {
        "control": {
            "nstep": 100,
            "etot_conv_thr": 1e-4,
            "forc_conv_thr": 1e-3,
            "dipfield": True,
            "tefield": True
        },
        "system": {
            "eamp": 0.0,
            "edir": 1,
            "emaxpos": 0.5,
            "eopreg": 0.1,
            "dftd3_version": 3,
            "vdw_corr": "dft-d3"
        },
        "electrons": {}
    }

    params_file.write_text(json.dumps(params_content))
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()

    strategy = QECalculator(basic_params=str(params_file), pseudopots={}, pseudo_dir=str(pseudo_dir))

    # Test relax
    strategy.get_calculator(job_name="test_job_relax", calc_type="relax")

    # Check remote settings for relax
    call_kwargs = mock_prepare_remote.call_args[1]
    assert call_kwargs["max_time"] == "06:25:00"
    assert call_kwargs["n_cores"] == 32
