import random
import numpy as np
from ase.io import read
from ase.mep import NEB
from ase import Atoms
from quippy.descriptors import Descriptor


class NEBPairFinder:
    """Generates NEB pathways from a pool of structures using random or similarity-based selection.
    
    Attributes:
        structures (list): List of ASE Atoms objects loaded from the input file.
    """
    def __init__(self, structures_file):
        """Initializes with path to structure file and prints statistics.
        
        Args:
            structures_file (str): Path to the XYZ file containing structures.
        """
        self.structures = read(structures_file, index=':')
        
        # Count and print structure statistics
        counts = {}
        for atoms in self.structures:
            species = atoms.info.get('species')
            counts[species] = counts.get(species, 0) + 1
        print(f"Found structures: {counts}")

    def generate_pathway(self, initial_image: Atoms, final_image: Atoms, n_images=5, method='idpp'):
        """Generates a NEB pathway between two images.
        
        Args:
            initial_image (Atoms): Reactant structure.
            final_image (Atoms): Product structure.
            n_images (int): Number of intermediate images.
            method (str): Interpolation method (e.g., 'idpp').
            
        Returns:
            list: List of Atoms objects representing the band.
        """
        # Create a list of images for the band
        images = [initial_image]
        for _ in range(n_images):
            images.append(initial_image.copy())
        images.append(final_image)

        # Setup NEB
        neb = NEB(images)
        neb.interpolate(method=method)
        
        return images
    
    def generate_random(self, transition_string, n_pairings, n_images=5, method='idpp'):
        """Generates random NEB pathways for a specified transition.
        
        Args:
            transition_string (str): Transition label in format 'reactant -> product'.
            n_pairings (int): Number of random pathways to generate.
            n_images (int): Number of intermediate images per pathway.
            method (str): Interpolation method.
            
        Returns:
            list: List of generated NEB pathways (lists of Atoms objects).
        """
        start_pool, end_pool = self._get_pools_from_transition(transition_string)
        generated_pathways = []
        seen_pairs = set()
        for _ in range(n_pairings):
            # Try to find a unique pair
            start_image = None
            end_image = None
            
            # Simple retry mechanism to avoid duplicates
            for attempt in range(100):
                start_image = random.choice(start_pool)
                end_image = random.choice(end_pool)
                
                # Check composition compatibility
                if start_image.get_chemical_formula() != end_image.get_chemical_formula():
                   continue
                
                pair_id = (id(start_image), id(end_image))
                if pair_id not in seen_pairs:
                    seen_pairs.add(pair_id)
                    break 
            else:
                print("Warning: Could not find unique, compatible pair after 100 attempts.")
                break
            pathway = self.generate_pathway(start_image, end_image, n_images, method)
            generated_pathways.append(pathway)
        
        return generated_pathways
    
    def generate_similarity_pathways(self, transition_string, n_pathways, descriptor_string, atom_slice=slice(None), n_images=5, method='idpp'):
        """Generates pathways based on highest structural similarity between pools.
        
        Args:
            transition_string (str): Transition label in format 'reactant -> product'.
            n_pathways (int): Number of top similarity pathways to generate.
            descriptor_string (str): SOAP descriptor string for similarity calculation.
            atom_slice (slice, optional): Slice to select atoms for descriptor calculation.
            n_images (int): Number of intermediate images.
            method (str): Interpolation method.
            
        Returns:
            list: List of generated NEB pathways.
        """
        start_pool, end_pool = self._get_pools_from_transition(transition_string)
        # Calculate similarity matrix between start and end pools
        sim_matrix = create_similarity_matrix(start_pool, descriptor_string, atom_slice, mols2=end_pool)
        
        # Identify top n_pathways indices (i, j) with largest values
        # Flatten matrix and sort indices
        flat_indices = np.argsort(sim_matrix.flatten())[::-1] # Descending order
        
        generated_pathways = []
        seen_pairs = set()
        
        count = 0
        for idx in flat_indices:
            if count >= n_pathways:
                break
            
            i, j = np.unravel_index(idx, sim_matrix.shape)
            start_image = start_pool[i]
            end_image = end_pool[j]
            
            # Check composition compatibility
            if start_image.get_chemical_formula() != end_image.get_chemical_formula():
                continue
            
            # Check if pair has been processed
            pair_id = (id(start_image), id(end_image))
            if pair_id in seen_pairs:
                continue
            
            seen_pairs.add(pair_id)
            
            # Generate pathway
            pathway = self.generate_pathway(start_image, end_image, n_images, method)
            generated_pathways.append(pathway)
            count += 1
            
        if count < n_pathways:
            print(f"Warning: Only generated {count} pathways out of requested {n_pathways}.")
            
    def _get_pools_from_transition(self, transition_string):
        """Extracts and validates reactant and product pools from a transition string.
        
        Args:
            transition_string (str): Label in format 'reactant -> product'.
            
        Returns:
            tuple: (start_pool, end_pool) lists of Atoms objects.
        """
        try:
            start_label, end_label = map(str.strip, transition_string.split('->'))
        except ValueError:
            raise ValueError("Transition string must be in format 'label1 -> label2'")
        
        start_pool = [atoms for atoms in self.structures if atoms.info.get('species') == start_label]
        end_pool = [atoms for atoms in self.structures if atoms.info.get('species') == end_label]
        
        if not start_pool:
            raise ValueError(f"No structures found with label '{start_label}'")
        if not end_pool:
            raise ValueError(f"No structures found with label '{end_label}'")
        
        return start_pool, end_pool


def create_similarity_matrix(mols1, descriptor_string, atom_slice=slice(None), mols2=None):
    """Creates a similarity matrix between mols1 and mols2 (or mols1 itself) using given descriptor.
    
    Args:
        mols1 (list): First list of Atoms objects.
        descriptor_string (str): SOAP descriptor string.
        atom_slice (slice, optional): Slice to select atoms for descriptor calculation.
        mols2 (list, optional): Second list of Atoms objects. If None, computes self-similarity of mols1.
        
    Returns:
        np.ndarray: Calculated similarity matrix (normalized dot product).
    """
    desc = Descriptor(descriptor_string)
    
    soaps1 = [desc.calc_descriptor(m[atom_slice]).flatten() for m in mols1]
    X1 = np.array(soaps1)
    norms1 = np.linalg.norm(X1, axis=1, keepdims=True)
    X1_norm = X1 / norms1
    
    if mols2 is None:
        return np.dot(X1_norm, X1_norm.T)
    
    soaps2 = [desc.calc_descriptor(m[atom_slice]).flatten() for m in mols2]
    X2 = np.array(soaps2)
    norms2 = np.linalg.norm(X2, axis=1, keepdims=True)
    X2_norm = X2 / norms2
    
    return np.dot(X1_norm, X2_norm.T)