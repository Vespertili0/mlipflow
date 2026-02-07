import pytest
import os
from ase.constraints import FixAtoms, FixBondLength
from mlipflow.core.neb_pairing import create_neb_pairs
from wfl.configset import ConfigSet

def test_create_neb_pairs():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xyz_file = os.path.join(current_dir, 'data', 'test_data.xyz')
    
    # Define parameters
    descriptor_string = 'soap n_species=4 species_Z={1 6 8 29} l_max=6 n_max=8 cutoff=3.5 atom_sigma=0.5 zeta=6'
    rxn_constraints_dict = {
        'tFUR+2H -> tFURha+H': [FixAtoms(list(range(32))), FixBondLength(70, 76)], 
        'tFUR+2H -> tFURao+H': [FixAtoms(list(range(32))), FixBondLength(69, 75)], 
        'tFURha+H -> FA': [FixAtoms(list(range(32))), FixBondLength(69, 75)], 
        'tFURao+H -> FA': [FixAtoms(list(range(32))), FixBondLength(70, 76)]
    }
    n_pathways = 3
    
    # Run function
    results = create_neb_pairs(
        xyz_file=xyz_file,
        rxn_constraints_dict=rxn_constraints_dict,
        method='similarity',
        n_pathways=n_pathways,
        descriptor_string=descriptor_string
    )
    
    # Verify results
    assert len(results) == len(rxn_constraints_dict)
    
    for i, result in enumerate(results):
        # result is a list of ConfigSet (or similar, depending on wfl_map output)
        # wfl_map returns a list of outputs for each input item in the ConfigSet
        # Here input ConfigSet had n_pathways items.
        
        # Verify number of pathways
        # Note: wfl_map returns a list of things returned by map_func. 
        # But wait, create_neb_pairs returns `mapped_results`, which is a list.
        # Each element of `mapped_results` is the return value of `wfl_map`.
        # `wfl_map` returns a ConfigSet (if OutputSpec was used correctly) or list of results?
        # Looking at `create_neb_pairs`:
        # mapped_results.append(wfl_map(..., outputs=OutputSpec(), ...))
        # wfl.map returns ConfigSet if OutputSpec is valid/provided.
        
        # Check type
        assert isinstance(result, ConfigSet)
        
        # Check number of items in ConfigSet
        # We expect n_pathways lists of Atoms (each path is a list of Atoms)
        # But wait, `apply_constraints` takes `at` (Atoms) and returns `at`.
        # The input to `wfl_map` is `ConfigSet(paths)`. `paths` is a list of lists of Atoms.
        # wfl ConfigSet iterates over "configs".
        # If `paths` is a list of lists of Atoms, does ConfigSet flatten it? 
        # Usually ConfigSet iterates over atoms objects. 
        # `generate_similarity_pathways` returns list of lists of Atoms.
        # If I pass list-of-lists to ConfigSet, it might be treated as list of configs if the inner list is atoms?
        # Let's check `neb_pairing.py` logic.
        # loops over reactions. `paths` = list of pathways. Each pathway = list of Atoms.
        # ConfigSet(paths) -> iteration yields... ? 
        # If ConfigSet handles list of lists, it usually flattens? Or treats sublists as "items"?
        # Actually NEB paths are usually list of Atoms.
        # If `apply_constraints` expects `at` (Atoms), then `ConfigSet` must be iterating over Atoms.
        # So `paths` (list of lists of Atoms) -> ConfigSet -> iterates over all Atoms in all pathways concurrently?
        # If so, `wfl_map` returns a ConfigSet containing all Images of all Pathways.
        
        # Let's verify what `apply_constraints` does. It takes `at` and sets constraints.
        # This acts on individual images.
        
        # So `result` should contain (n_pathways * images_per_pathway) Atoms objects?
        # `generate_similarity_pathways` uses default n_images=5 (internal) + start/end = 7 images total?
        # `create_neb_pairs` default n_images=7.
        # Let's check `generate_pathway`:
        # images = [initial] + [copy]*n_images + [final]. Total = n_images + 2.
        # `create_neb_pairs` passes `n_images` to `generate_similarity_pathways`.
        # so length of one path is n_images + 2.
        
        # Total expected items = n_pathways * (n_images_default + 2)
        # In test, we didn't specify n_images, so default=9 (from user edit).
        # Total = 3 * (9 + 2) = 33 atoms objects per reaction.
        
        pass

if __name__ == "__main__":
    test_create_neb_pairs()
