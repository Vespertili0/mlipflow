from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from ase.constraints import FixAtoms, FixBondLength
from ase.io import read

from mlipflow.core.neb_pairing import create_neb_pairs
from mlipflow.strategies.dft import EMTCalc
from mlipflow.strategies.structure_generators import OPTGen


@pytest.fixture
def test_data_setup(tmp_path):
    current_dir = Path(__file__).resolve().parent
    src_xyz = str(Path(current_dir) / "data" / "test_data.xyz")
    config_file = tmp_path / "test_data.xyz"
    shutil.copy2(src_xyz, config_file)
    return config_file, tmp_path


def test_neb_pairing_and_opt(test_data_setup):
    xyz_file, tmp_path = test_data_setup

    # Define parameters
    descriptor_string = "soap n_species=4 species_Z={1 6 8 29} l_max=6 n_max=8 cutoff=3.5 atom_sigma=0.5 zeta=6"
    rxn_constraints_dict = {
        "tFUR+2H -> tFURha+H": [FixAtoms(list(range(32))), FixBondLength(70, 76)],
        "tFUR+2H -> tFURao+H": [FixAtoms(list(range(32))), FixBondLength(69, 75)],
        "tFURha+H -> FA": [FixAtoms(list(range(32))), FixBondLength(69, 75)],
        "tFURao+H -> FA": [FixAtoms(list(range(32))), FixBondLength(70, 76)],
    }
    n_pathways = 3
    opt_params = {"fmax": 0.05, "steps": 5}

    # Initialise strategy and get calculator tuple
    calculator_strategy = EMTCalc()
    calculator_tuple = calculator_strategy.get_calculator(job_name="test")

    # Methods to test
    methods = ["similarity", "random"]

    for method in methods:
        # Run create_neb_pairs
        results = create_neb_pairs(
            xyz_file=str(xyz_file),
            rxn_constraints_dict=rxn_constraints_dict,
            method=method,
            n_pathways=n_pathways,
            descriptor_string=descriptor_string if method == "similarity" else None,
        )

        assert len(results) == len(rxn_constraints_dict)

        # Run OPTGen on resulting structures
        # Use traj_subselect=None to keep full trajectory even if unconverged
        opt_gen = OPTGen(params=opt_params, traj_subselect=None)

        # Iterate over results (ConfigSets)
        for i, config_set in enumerate(results):
            # Extract all atoms from all bands and flatten
            atoms_list = []
            for band in config_set:
                if isinstance(band, list):
                    atoms_list.extend(band)
                else:
                    atoms_list.append(band)

            # Create a new ConfigSet for OPTGen from flattened list
            # OPTGen expects inputs to be a list or ConfigSet.

            out_file = tmp_path / f"opt_results_{method}_{i}.xyz"

            opt_gen.generate_new_structures(
                in_file=atoms_list,
                out_file=str(out_file),
                calculator=calculator_tuple,  # passing the tuple as calculator
            )

            # Check if output file exists and has content
            assert out_file.exists()
            optimized_atoms = read(str(out_file), ":")
            assert len(optimized_atoms) > 0

            # Check if optimization happened
            # Use generic check, or specific key if known
            # Since we use EMTCalc (generic), it might add DFT_energy or similar
            # OPTGen puts 'optimize_config_type' in info
            assert "optimize_config_type" in optimized_atoms[0].info
