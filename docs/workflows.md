# Workflows

`mlipflow` provides several pre-defined workflows to orchestrate common tasks. These are implemented as `LangGraph` graphs.

## Initial Basin & Path Sampling

The `execute_initial_basin_pathsampling_md_block` workflow is designed for the initial exploration of potential energy surfaces. It performs the following steps:

1.  **Apply Basin Constraints**: Applies geometric constraints to the input structures (e.g., fixing atoms).
2.  **Basin MD**: Runs Molecular Dynamics (MD) or Geometry Optimisation within the defined basins to generate diverse local structures.
3.  **Generate NEB Pairs**: Automatically pairs structures to form Nudged Elastic Band (NEB) pathways.
4.  **Path Sampling MD**: Runs MD/Optimisation on the generated NEB images to sample the reaction pathways.
5.  **Single Point Calculations**: Performs single-point calculations (MLIP or DFT) on the generated structures to evaluate their energies and forces.

### Usage

```python
from mlipflow.graphflow.graphs import execute_initial_basin_pathsampling_md_block

# Compile the graph
workflow = execute_initial_basin_pathsampling_md_block()

# Run the workflow with an initial state
# result = workflow.compile().invoke(state)
```

## EnsembleState

The `EnsembleState` is the central data structure passed between nodes in the workflow. It contains all necessary information, including configuration files, strategies, and calculation parameters.

### Structure

```python
class EnsembleState(TypedDict):
    configs: list[str]                  # List of configuration file paths
    outfile: NotRequired[list[str]]     # Optional output file paths
    qchem_strategy: QChemStrategy       # Quantum chemistry calculation strategy
    mlip_strategy: MLIPStrategy         # MLIP fitting strategy
    structure_gen_strategy: NotRequired[StructureGenStrategy] # Strategy for structure generation (MD/OPT)
    calculation_kwargs: NotRequired[dict[str, Any]] # Dictionary containing parameters for various calculation steps
```

### Complete Example with Default Settings

Here is a complete example of how to construct an `EnsembleState` for the `execute_initial_basin_pathsampling_md_block` workflow, including default settings for various parameters.

```python
from mlipflow.graphflow.nodes import EnsembleState
from mlipflow.strategies.dft import QChemStrategy, QECalculator
from mlipflow.strategies.mlip import MLIPStrategy, MACEModel
from mlipflow.strategies.structure_generators import MDGen
from ase.constraints import FixAtoms

# 1. Define Strategies
qchem_strategy = QChemStrategy(
    calculator=QECalculator(
        command="pw.x",
        pseudopotentials="./pseudos"
    )
)

mlip_strategy = MLIPStrategy(
    model=MACEModel(model_path="mace_model.model")
)

structure_gen_strategy = MDGen(
    steps=4000,
    timestep=1.0,
    temperature=600
)

# 2. Define Calculation Arguments
calculation_kwargs = {
    # MLIP Single Point (evaluating generated structures)
    'mlip_sp': {
        'dispersion': False,
        'num_inputs_per_queued_job': 300,
        'max_time': '01:30:00'
    },
    
    # MLIP Structure Generation (MD/Opt parameters are primarily in structure_gen_strategy, 
    # but specific cleaner/job params can be here)
    'mlip_gen': {
        'dispersion': True,
        'num_inputs_per_queued_job': 15,
        'max_time': '01:30:00'
    },

    # Initial Sampling Specifics
    'initial_sampling': {
        # Constraints to apply during Basin MD
        'basin_constraints': [
            FixAtoms(indices=[0, 1, 2, 3]) # any ase.constraint works here
        ],
        
        # Configuration for NEB Pair Generation
        'neb_config': {
            'rxn_constraints_dict': {
                'chem_A+chem_B -> product_C': [FixAtoms(indices=[0, 1])]
            },
            'method': 'similarity',
            'n_pathways': 5,
            'n_images': 7,
            'descriptor_string': 'soap cutoff=3.0 l_max=6 n_max=9 atom_sigma=0.5'
        }
    },
     
    # Configuration Selection (if needed for later steps)
    'selection': {
        'descriptor_key': 'SOAP',
        'descriptor_string': 'soap cutoff=3.0 l_max=6 n_max=9 atom_sigma=0.5',
        'info_field': 'MACE_energy', # field to use for selection (e.g. histogramming)
        '**kwargs': {'kT': 0.05}          
    }
}

# 3. Construct the State
state: EnsembleState = {
    'configs': ['initial_structures.xyz'],
    'qchem_strategy': qchem_strategy,
    'mlip_strategy': mlip_strategy,
    'structure_gen_strategy': structure_gen_strategy,
    'calculation_kwargs': calculation_kwargs
}
```
