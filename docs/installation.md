---
title: Installation
layout: default
---

# Installation

`mlipflow` is a Python shell for machine learning interatomic potential workflows. It relies on several key scientific libraries for atomic simulations and machine learning.

## 📋 Environment Requirements

- **Python**: 3.9 or higher (3.10+ recommended)
- **Operating System**: Linux (required for `wfl` and many DFT calculators)
- **GPU Acceleration**: NVIDIA GPU with CUDA support is highly recommended for MLIP (MACE) training and inference.

## 🛠️ Setup Steps

### 1. Create a Conda Environment (Recommended)

```bash
conda create -n mlipflow python=3.10
conda activate mlipflow
```

### 2. Install PyTorch

Refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command for your CUDA version.

```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install mlipflow

You can install `mlipflow` directly from the Git repository:

```bash
pip install git+https://github.com/Vespertili0/mlipflow.git
```

This will automatically install most dependencies, including:
- `mace-torch`
- `quippy-ase`
- `langgraph`
- `wfl` (from its git repository)

## ⚙️ Configuration Notes

### HPC Integration

`mlipflow` uses `wfl` for workflow execution, which can interface with various HPC schedulers (Slurm, PBS). To configure `wfl` for your cluster, you may need to set specific environment variables or provide a `remote_info` dictionary to your strategies.

### DFT Calculators

Ensure your DFT codes (e.g., Quantum Espresso) are installed and accessible in your shell path. You will need to provide the path to the executable (e.g., `pw.x`) to the `QECalculator` strategy.

### MACE Models

If using pre-trained MACE models, ensure they are compatible with the version of `mace-torch` installed.
