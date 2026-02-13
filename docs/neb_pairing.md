# NEB Pairing

The `mlipflow.core.neb_pairing` module provides functionality to automatically generate Nudged Elastic Band (NEB) pathways. This is essential for exploring transition states and reaction barriers in complex systems.

## Key Concepts

The core function is `create_neb_pairs`, which takes a pool of reactant and product structures (usually from an XYZ file) and pairs them to create initial NEB bands.

### Pairing Methods

Two methods are available for selecting pairs:

1.  **Similarity (`method='similarity'`)**: Pairs reactants and products based on their structural similarity. This uses the **SOAP** (Smooth Overlap of Atomic Positions) descriptor to calculate a similarity matrix and selects the pairs with the highest similarity. This is generally preferred as it leads to more physically reasonable pathways.
2.  **Random (`method='random'`)**: Randomly pairs compatible reactants and products. This is useful for unbiased exploration or when similarity metrics are not applicable.

### Constraints

You can apply constraints to specific atoms during the NEB calculation (e.g., fixing the bottom layers of a slab). These are passed via a dictionary mapping the reaction string to a list of ASE constraints.

## Usage

### `create_neb_pairs`

This is the main entry point.

```python
from mlipflow.core.neb_pairing import create_neb_pairs
from ase.constraints import FixAtoms

# 1. Define Constraints
# Map 'reactant_label -> product_label' to a list of constraints
constraints = {
    'slab+CO -> slab_CO': [FixAtoms(indices=[0, 1, 2, 3])]
}

# 2. Generate NEB Pairs
# This returns a list of ConfigSets, where each ConfigSet contains the bands for a reaction.
neb_pathways = create_neb_pairs(
    xyz_file='data.xyz',                # Path to your structure file
    rxn_constraints_dict=constraints,   # Your constraints dictionary
    method='similarity',                # Use similarity-based pairing
    n_pathways=5,                       # Number of pathways to generate per reaction
    n_images=7,                         # Number of intermediate images (excluding start/end)
    descriptor_string='soap cutoff=3.0 l_max=6 n_max=9 atom_sigma=0.5' # SOAP descriptor string
)
```

### `NEBPairFinder`

For more granular control, you can use the `NEBPairFinder` class directly.

```python
from mlipflow.core.neb_pairing import NEBPairFinder

# Initialize with your structure file
finder = NEBPairFinder('data.xyz')

# Generate random pathways for a specific transition
random_paths = finder.generate_random(
    transition_string='slab+CO -> slab_CO',
    n_pairings=10,
    n_images=5
)

# Generate similarity-based pathways
similarity_paths = finder.generate_similarity_pathways(
    transition_string='slab+CO -> slab_CO',
    n_pathways=5,
    descriptor_string='soap ...',
    n_images=5
)
```

## Input File Requirements

The `xyz_file` must contain ASE `Atoms` objects with an `info['species']` tag matching the labels used in your `rxn_constraints_dict` (e.g., `'slab+CO'` and `'slab_CO'`).

## Output

The `create_neb_pairs` function returns a list of `wfl.configset.ConfigSet` objects (or similar iterable structure depending on WFL version). Each item in the sequence is a **list of Atoms objects** representing a single NEB band (initial + intermediates + final).
