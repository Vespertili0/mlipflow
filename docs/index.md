# Welcome to mlipflow

**mlipflow** is a Python package designed to automate the exploration of complex reaction pathways using Machine-Learned Interatomic Potentials (MLIPs). It bridges the gap between high-level machine learning development and practical catalysis research by providing a modular, scalable framework.

## 🚀 Key Features

*   **Modularity**: Built on [ASE](https://wiki.fysik.dtu.dk/ase/) and [LangGraph](https://langchain-ai.github.io/langgraph/), allowing for easy extension of functions and integration of custom calculators.
*   **Scalability**: Integrated with [WFL](https://github.com/libAtoms/workflow) to support seamless execution on HPC platforms, scaling from small tests to massive datasets.
*   **Extensibility**: Supports multiple strategies for structure generation (e.g., Molecular Dynamics, Geometry Optimization, Nudged Elastic Band) and MLIP training (e.g., MACE).

## 🎯 Purpose

Traditional heterogeneous catalysis research often faces a trade-off between the accuracy of Density Functional Theory (DFT) and the computational cost required for exhaustive pathway exploration. `mlipflow` addresses this by:

1.  **Automating Exploration**: Using MLIPs to rapidly explore Potential Energy Surfaces (PES).
2.  **Active Learning**: Employing an iterative loop to refine MLIP models with targeted DFT calculations.
3.  **Graph-Based Workflow**: Utilizing a flexible graph architecture to manage complex dependencies and workflows.

## 📚 Documentation Structure

*   **[Installation](install.md)**: Get started with installing `mlipflow` and its dependencies.
*   **[Overview](overview.md)**: Understand the core concepts, including the graph-based workflow and available strategies.
*   **[Workflows](workflows.md)**: Detailed documentation on pre-defined workflows and state configuration.
*   **[NEB Pairing](neb_pairing.md)**: A detailed guide on generating NEB pathways using the `neb_pairing` module.
