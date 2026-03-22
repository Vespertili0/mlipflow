---
title: Usage
layout: default
---

# Usage Guide

This guide demonstrates common tasks and workflows in `mlipflow`.

## 📍 Basic Active Learning Loop

The fundamental workflow in `mlipflow` is the active learning loop, which iteratively refines an MLIP model by selecting and labeling informative configurations.

```python
from mlipflow.strategies.mlip import MACEModel
from mlipflow.strategies.dft import QECalculator
from mlipflow.strategies.structure_generators import MDGen
from mlipflow.graphflow.activelearner import run_active_learning_loop

# 1. Define Strategies
mlip = MACEModel(model_path="initial_model.model", mlip_prefix="mace_")

dft = QECalculator(pw_path="/path/to/pw.x", qe_prefix="dft_")

md_strategy = MDGen(params={"temperature": 500, "n_steps": 1000, "time_step": 0.5})

# 2. Set Up Initial State
initial_state = {
    "configs": ["input_structures.xyz"],
    "mlip_strategy": mlip,
    "qchem_strategy": dft,
    "structure_gen_strategy": md_strategy,
    "iteration": 1,
    "calculation_kwargs": {"dft_scf": {"ecut_eV": 500, "kpts": (3, 3, 1)}},
}

# 3. Run the Loop
final_state = run_active_learning_loop(initial_state)

print(f"Completed iteration: {final_state['iteration']}")
```

## 🏗️ Generating New Structures

You can use the structure generation strategies independently to explore configurations.

### Molecular Dynamics (MD)

```python
from mlipflow.strategies.structure_generators import MDGen
from mlipflow.strategies.mlip import MACEModel

calc = MACEModel(model_path="model.model").get_calculator()
md = MDGen(params={"temperature": 600, "n_steps": 2000})

md.generate_new_structures(
    in_file="reactant.xyz", out_file="trajectory.xyz", calculator=calc
)
```

### Nudged Elastic Band (NEB)

```python
from mlipflow.strategies.structure_generators import NEBGen

neb = NEBGen(params={"n_images": 8, "fmax": 0.05})
# Note: NEB implementation details depend on specific strategy logic
```

## 📊 Error Analysis

Assess the accuracy of your MLIP compared to DFT data.

```python
from mlipflow.core.calculate_error import calculate_mlip_error

calculate_mlip_error(
    in_configs="labeled_data.xyz",
    out_file="error_summary.xyz",
    calc_property_prefix="mace_",
    ref_property_prefix="dft_",
)
```

## 🔍 Configuration Selection

Select the most informative configurations from a pool for labeling.

```python
from mlipflow.data.selector import ConfigurationSelector

selector = ConfigurationSelector(inputs=["pool.xyz"], output_prefix="best_configs")

# Calculate global descriptors (e.g., SOAP)
selector.calculate_global_descriptors(
    descs=["soap cutoff=5.0 n_max=8 l_max=6"], key="SOAP"
)

# Run Farthest Point Sampling selection
selected = selector.run_two_stage_selection(n_optimal=50, descriptor_key="SOAP")
```
