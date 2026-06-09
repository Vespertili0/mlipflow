---
title: Examples
layout: default
---

# Advanced Examples

This page explores more complex scenarios and advanced usage of `mlipflow`.

## 🔄 Custom Active Learning Workflow

You can customise the active learning cycle by defining a new LangGraph state and nodes.

```python
from langgraph.graph import StateGraph, START, END
from mlipflow.graphflow.nodes import run_mlip_sp, run_dft_sp, run_mace_fit


# 1. Define custom state
class CustomState(dict):
    configs: list[str]
    # ... other state keys


# 2. Build graph
workflow = StateGraph(CustomState)
workflow.add_node("compute_mlip", run_mlip_sp)
workflow.add_node("compute_dft", run_dft_sp)
workflow.add_node("relabel", run_mace_fit)

workflow.add_edge(START, "compute_mlip")
workflow.add_edge("compute_mlip", "compute_dft")
workflow.add_edge("compute_dft", "relabel")
workflow.add_edge("relabel", END)

# 3. Compile and execute
app = workflow.compile()
results = app.invoke({"configs": ["initial.xyz"]})
```

## 🏞️ Basin-Constrained Exploration

Use fixed constraints during structural exploration to focus on specific regions of the PES.

```python
from mlipflow.graphflow.nodes import run_apply_basin_constraints
from ase.constraints import FixAtoms

# Define constraints (e.g., fix all catalyst slab atoms)
constraints = [FixAtoms(indices=[0, 1, 2, 3])]

state = {
    "configs": ["adsorbate_on_slab.xyz"],
    "calculation_kwargs": {
        "initial_sampling": {
            "basin_constraints": constraints,
            "basin__mlip_gen": {"dispersion": True},
        }
    },
}

new_state = run_apply_basin_constraints(state)
```

## 🧪 NEB Path Discovery

Generate transition state candidates by linking local minima via NEB.

```python
from mlipflow.graphflow.nodes import run_generate_neb_pairs

state = {
    "configs": ["reactant.xyz", "product.xyz"],
    "calculation_kwargs": {
        "neb_config": {
            "rxn_constraints_dict": {"type": "bond", "indices": [0, 5], "target": 1.5},
            "n_images": 12,
        }
    },
}

# Generates intermediate images and runs MLIP refinement
neb_state = run_generate_neb_pairs(state)
```
