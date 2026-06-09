---
title: API Reference
layout: default
---

# API Reference

This page provides detailed documentation for the core modules, classes, and strategies of `mlipflow`.

## 📂 `mlipflow.core`

Low-level functions for calculations and data manipulation.

### `run_single_point`

Runs single-point energy and force calculations.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `in_file` | `list[str] \| ConfigSet` | Input configuration files. |
| `out_file` | `list[str] \| OutputSpec` | Output configuration files. |
| `calculator` | `ASE Calculator` | ASE calculator instance to use. |
| `remote_info` | `dict \| None` | Optional HPC execution parameters. |

**Example:**
```python
from mlipflow.core.single_point import run_single_point
from mlipflow.strategies.mlip import MACEModel

calc = MACEModel(model_path="model.model").get_calculator()
run_single_point(in_file=["atoms.xyz"], out_file=["results.xyz"], calculator=calc)
```

---

## 📂 `mlipflow.strategies`

Abstract base classes and concrete implementations for calculation backends.

### `MLIPStrategy` (Abstract)

Base class for Machine-Learned Interatomic Potentials.

| Method | Returns | Description |
| :--- | :--- | :--- |
| `get_calculator()` | `ASE Calculator` | Returns an ASE-compatible calculator. |
| `fit_new_model()` | `None` | Trains or refines the MLIP model. |

### `MACEModel` (Implements `MLIPStrategy`)

Support for the [MACE](https://github.com/ACEsuit/mace) architecture.

**Example:**
```python
from mlipflow.strategies.mlip import MACEModel

mace = MACEModel(model_path="my_model.model")
calc = mace.get_calculator(job_name="MACE_Calc")
```

---

## 📂 `mlipflow.data`

Tools for data management and selection.

### `ConfigurationSelector`

Class for selecting a subset of configurations from a pool.

**Methods:**

| Method | Parameters | Description |
| :--- | :--- | :--- |
| `calculate_global_descriptors` | `descs (list), key (str)` | Calculates global descriptors (e.g., SOAP). |
| `run_two_stage_selection` | `n_optimal (int), ...` | Runs FPS selection based on descriptors. |

**Example:**
```python
from mlipflow.data.selector import ConfigurationSelector

selector = ConfigurationSelector(inputs=["pool.xyz"])
selector.calculate_global_descriptors(descs=["soap cutoff=5.0"], key="SOAP")
selected = selector.run_two_stage_selection(n_optimal=10, descriptor_key="SOAP")
```

---

## 📂 `mlipflow.graphflow`

LangGraph-based workflow nodes and state management.

### `ActiveLearningFlow` (State Definition)

Typed dictionary representing the global flow state.

- `configs`: List of current configuration file paths.
- `mlip_strategy`: Current MLIP strategy instance.
- `qchem_strategy`: Current DFT strategy instance.
- `iteration`: Current iteration count.

### `run_active_learning_loop`

Starts the full active learning cycle.

**Signature:** `run_active_learning_loop(initial_state: dict)`

---

## 📂 `mlipflow.strategies.structure_generators`

Strategies for exploring new atomic configurations.

### `MDGen`

Molecular Dynamics-based sampling.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `temperature` | `float` | - | Simulation temperature. |
| `n_steps` | `int` | - | Number of MD steps. |
| `time_step` | `float` | 0.5 | Integration time step (fs). |
| `dr_threshold`| `float` | 0.5 | Displacement threshold for sampling images. |

---

## 🚨 Error Conditions

- **`RuntimeError`**: Raised during calculator failure or file access issues.
- **`ValueError`**: Raised when input parameters are invalid or missing required keys.
- **`WorkflowError`**: Specific to LangGraph node failures.
