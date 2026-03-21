---
title: Home
layout: default
---

# mlipflow

**A Python Package for Reaction Pathway Exploration using Machine-Learned Interatomic Potentials (MLIPs)**

mlipflow bridges the critical gap between high-level machine learning development and practical catalysis research. It provides a modular, scalable framework to automate the exploration of complex reaction pathways with near-quantum accuracy at a fraction of the computational cost.

## 🚀 Quick Start

Get started with `mlipflow` in minutes.

### Installation

```bash
pip install git+https://github.com/Vespertili0/mlipflow.git
```

### Basic Usage

```python
from mlipflow.strategies.mlip import MACEModel
from mlipflow.strategies.dft import QECalculator
from mlipflow.graphflow.activelearner import run_active_learning_loop

# Initialise strategies
mlip = MACEModel(model_path="path/to/model.model")
dft = QECalculator(pw_path="pw.x")

# Define initial state
initial_state = {
    "configs": ["initial_structure.xyz"],
    "mlip_strategy": mlip,
    "qchem_strategy": dft,
    "iteration": 0
}

# Run the active learning loop
run_active_learning_loop(initial_state)
```

## 🗺️ Navigation

- [Overview](docs/overview.md): Learn about the purpose and architecture.
- [Installation](docs/installation.md): Detailed setup instructions.
- [Usage](docs/usage.md): Practical code examples and guides.
- [API Reference](docs/api-reference.md): Detailed documentation of all modules.
- [Examples](docs/examples.md): Advanced use cases.
- [Best Practices](docs/best-practices.md): Tips for optimal performance.
