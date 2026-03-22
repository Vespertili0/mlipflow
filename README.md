# mlipflow

<p align="center">
  [![Tests](https://github.com/Vespertili0/mlipflow/actions/workflows/tests.yaml/badge.svg)](https://github.com/Vespertili0/mlipflow/actions/workflows/tests.yaml)
  [![Documentation](https://github.com/Vespertili0/mlipflow/actions/workflows/docs.yaml/badge.svg)](https://github.com/Vespertili0/mlipflow)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
  [![codecov](https://img.shields.io/codecov/c/github/Vespertili0/mlipflow)](https://codecov.io/gh/Vespertili0/mlipflow)
</p>

**Automating Reaction Pathway Exploration with Machine-Learned Interatomic Potentials (MLIPs)**

`mlipflow` automates the development of system-specific MLIPs for modeling complex reaction pathways in catalysis and surface chemistry. It provides a modular, scalable framework that accelerates the exploration of reaction mechanisms with DFT-level accuracy at a fraction of the computational cost. By providing easily accessible workflow blocks that form an active learning loop with uncertainty-driven configuration selection, `mlipflow` iteratively refines the potential energy surface (PES) to ensure robust and reliable path discovery.

---

## 🚩 The Challenge: The "Catalysis Bottleneck"
Traditional heterogeneous catalysis research faces significant hurdles that `mlipflow` is built to solve:

*   **The Cost-Accuracy Trade-off:** Density Functional Theory (DFT) is the gold standard for accuracy but is computationally prohibitive for exhaustive pathway exploration.
*   **Human Selection Bias:** Researchers often manually "guess" reaction pathways, potentially missing rate-determining steps or low-energy configurations.
*   **Data Scarcity:** High-quality quantum chemical data is expensive to generate; `mlipflow` optimises data acquisition through active learning.

## ✨ Key Features
*   **Modular Architecture:** Built on [ASE](https://wiki.fysik.dtu.dk/ase/) and [LangGraph](https://github.com/langchain-ai/langgraph), allowing for seamless extension of calculators and workflow nodes.
*   **HPC Scalability:** Integrated with [WFL](https://github.com/libAtoms/workflow) for parallel execution on high-performance computing platforms.
*   **Multi-Model Support:** Native support for state-of-the-art MLIPs including **MACE** and **GAP**.
*   **Advanced Sampling:** Strategies for Molecular Dynamics (MD), Geometry Optimisation (OPT), and Nudged Elastic Band (NEB) structure generation.
*   **Smart Refinement:** GMM-based configuration selection to target high-uncertainty regions of the chemical space.

## 🔄 The Active Learning Workflow
`mlipflow` orchestrates a continuous loop to iteratively improve model accuracy:

1.  **Generate:** Create new configurations (via MD, OPT, or NEB) to explore unknown regions.
2.  **Compute:** Automatically trigger high-level DFT (Quantum ESPRESSO) for "ground truth" data on selected configurations.
3.  **Train:** Update the MLIP model, refining the PES for the next iteration.

## 🚀 Installation

Install `mlipflow` directly from GitHub:

```bash
pip install git+https://github.com/Vespertili0/mlipflow.git
```

## 🛠️ Quick Start

`mlipflow` provides a modular API that can be used at various levels of abstraction depending on your needs.

### 1. Feature Level: Simple Structure Generation
Use individual strategies for specific tasks like NEB-based structure generation with custom constraints.

```python
from ase.io import read
from ase.constraints import FixAtoms
from wfl.configset import ConfigSet, OutputSpec
from wfl.map import map as wfl_map
from mlipflow.strategies.structure_generators import NEBGen
from mlipflow.strategies.mlip import MACEModel

# 1. Initialise MACE and NEB generator
mlip = MACEModel(mlip_name="my_model", run_mode="remote")
neb_gen = NEBGen(params={"fmax": 0.2, "steps": 250, "climb": True})

# 2. Prepare inputs of pre-build NEB-frames
neb_configs = read("interpolated_path.xyz", ":")

# 3. Run CI-NEB computations
neb_gen.generate_new_structures(
    in_file=neb_configs,
    out_file="neb_trajectory.xyz",
    calculator=mlip.get_calculator(n_cores=1, max_time="00:30:00"),
    remote_info=mlip.remote_info,
)
```

### 2. Node Level: Modular Workflow Steps
Leverage pre-defined nodes utilising `langgraph` as a state-driven task. Here, we define a state with generalised neb_config for multiple reaction pathways. Reaction pairs are identified based on SOAP-similarity of provided initial and final configurations. The node will automatically interpolate between the initial and final configurations to generate the required number of images and pathways and run CI-NEB computations.

```python
from mlipflow.graphflow.nodes import run_generate_neb_pairs, EnsembleState
from mlipflow.strategies.structure_generators import MDGen
from ase.constraints import FixAtoms

# Define state with generalised neb_config for multiple reaction pathways
state: EnsembleState = {
    "configs": ["input.xyz"],  # collection of initial and product configurations
    "structure_gen_strategy": NEBGen(params={"fmax": 0.05}),
    "mlip_strategy": MACEModel(mlip_name="my_model"),
    "calculation_kwargs": {
        "neb_config": {
            "rxn_constraints_dict": {"initial -> final": [FixAtoms(...)]},
            "method": "similarity",  # uses SOAP to automatically identify pairs
            "n_pathways": 5,
            "n_images": 7,
            "descriptor_string": "soap n_species=4 species_Z={...} l_max=6 n_max=8 cutoff=3.5 atom_sigma=0.5 zeta=6",
        }
    },
}

# Execute node: creates NEB pairs and runs structure generation
new_state = run_generate_neb_pairs(state)
```

### 3. Graph Level: Complete Workflow Blocks
Run predefined, high-level workflow blocks for full automation. These blocks are designed to be used as building blocks for more complex workflows, like active learning loops.
The shown `execute_opt_neb_combination_block` wraps MLIP relaxation of multiple initial and final configurations, CI-NEB structure generation, MLIP single-point without dispersion correction and uncertainty-based selection of configurations for the next iteration.

```python
from mlipflow.graphflow.graphs import execute_opt_neb_combination_block
from mlipflow.strategies.mlip import MACEModel
from mlipflow.strategies.structure_generators import MDGen

# Compile the modular graph
graph = execute_opt_neb_combination_block()

# Run the workflow
mlip = MACEModel(mlip_name="my_model")
initial_state = {
    "configs": ["input_ensemble.xyz"],
    "mlip_strategy": mlip,
    "structure_gen_strategy": MDGen(uncertainty_thrs=0.1),
    "calculation_kwargs": {
        "mlip_sp": {"dispersion": False},
        "mlip_gen": {"dispersion": True},
        "neb_config": {
            "rxn_constraints_dict": {"initial -> final": [FixAtoms(...)]},
            "method": "similarity",
            "n_pathways": 5,
            "n_images": 7,
            "descriptor_string": "soap n_species=4 species_Z={...} l_max=6 n_max=8 cutoff=3.5 atom_sigma=0.5 zeta=6",
        },
        "neb__structure_gen_params": {"steps": 100},
        "neb__mlip_gen": {
            "dispersion": True,
            "num_inputs_per_queued_job": 20,
            "max_time": "06:30:00",
        },
    },
}
final_state = graph.invoke(initial_state)
```
This makes blocks ideal for building more complex workflows, like active learning loops.
