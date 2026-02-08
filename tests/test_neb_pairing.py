import pytest
import os
import shutil
import numpy as np
from ase.constraints import FixAtoms, FixBondLength
from ase.io import read
from mlipflow.core.neb_pairing import create_neb_pairs
from mlipflow.strategies.structure_generators import NEBGen
from mlipflow.strategies.dft import EMTCalc
from wfl.configset import ConfigSet

@pytest.fixture
def test_data_setup(tmp_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_xyz = os.path.join(current_dir, 'data', 'test_data.xyz')
    config_file = tmp_path / "test_data.xyz"
    shutil.copy2(src_xyz, config_file)
    return config_file, tmp_path

def test_neb_pairing_and_opt(test_data_setup):
    xyz_file, tmp_path = test_data_setup
    
    # Define parameters
    descriptor_string = 'soap n_species=4 species_Z={1 6 8 29} l_max=6 n_max=8 cutoff=3.5 atom_sigma=0.5 zeta=6'
    rxn_constraints_dict = {
        'tFUR+2H -> tFURha+H': [FixAtoms(list(range(32))), FixBondLength(70, 76)], 
        'tFUR+2H -> tFURao+H': [FixAtoms(list(range(32))), FixBondLength(69, 75)], 
        'tFURha+H -> FA': [FixAtoms(list(range(32))), FixBondLength(69, 75)], 
        'tFURao+H -> FA': [FixAtoms(list(range(32))), FixBondLength(70, 76)]
    }
    n_pathways = 3
    # NEBGen accepts neb_params. Using low steps for testing.
    neb_params = {'fmax': 0.05, 'steps': 5}
    
    # Initialize strategy and get calculator tuple
    calculator_strategy = EMTCalc()
    calculator_tuple = calculator_strategy.get_calculator(job_name="test")

    # Methods to test
    methods = ['similarity', 'random']
    
    for method in methods:
        print(f"Testing method: {method}")
        # Run create_neb_pairs
        results = create_neb_pairs(
            xyz_file=str(xyz_file),
            rxn_constraints_dict=rxn_constraints_dict,
            method=method,
            n_pathways=n_pathways,
            descriptor_string=descriptor_string if method == 'similarity' else None
        )
        
        assert len(results) == len(rxn_constraints_dict)
        
        # Run NEBGen on resulting structures
        # Use traj_subselect=None to keep full trajectory even if unconverged
        neb_gen = NEBGen(neb_params=neb_params, traj_subselect=None)
        
        # Iterate over results (list of lists of bands)
        for i, bands_list in enumerate(results):
             # results is now a list of lists of bands.
             # bands_list is a list of bands.

             # Check first item
             if len(bands_list) > 0:
                 assert isinstance(bands_list[0], list), "Expected list of Atoms (band)"
                 assert len(bands_list[0]) > 0, "Band should not be empty"
                 # Ensure contents are Atoms
                 assert hasattr(bands_list[0][0], 'get_positions'), "Band should contain Atoms objects"

             out_file = tmp_path / f"neb_results_{method}_{i}.xyz"

             neb_gen.generate_new_structures(
                 in_file=bands_list,
                 out_file=str(out_file),
                 calculator=calculator_tuple # passing the tuple as calculator
             )

             # Check if output file exists and has content
             assert out_file.exists()
             # Result of NEB is usually the relaxed band or trajectory.
             # If traj_subselect=None, it might be the full trajectory of bands?
             # Or just the relaxed bands?
             # wfl.generate.neb usually outputs the relaxed images.

             optimized_atoms = read(str(out_file), ':')
             assert len(optimized_atoms) > 0

             # Check for NEB specific info tags if any, or just generic optimization tags
             # wfl NEB stores 'neb_config_type'
             assert 'neb_config_type' in optimized_atoms[0].info
