# Overview

`mlipflow` provides a structured approach to exploring reaction pathways using Machine-Learned Interatomic Potentials (MLIPs). The core of the package is an active learning workflow orchestrated by a graph-based engine.

## Graph-Based Workflow

The central component of `mlipflow` is the **Active Learning Flow**, implemented using `LangGraph`. This workflow manages the cyclic process of generating structures, validating them with DFT, and retraining the MLIP model.

The `ActiveLearningFlow` consists of the following key nodes:

1.  **`generate_new_structures`**: Uses the current MLIP model to generate new candidate structures (e.g., via Molecular Dynamics or NEB).
2.  **`calculate_dft_level`**: Selects a subset of the generated structures and performs high-accuracy DFT calculations to obtain ground-truth energies and forces.
3.  **`train_new_mlip_model`**: Retrains the MLIP model using the newly acquired DFT data, improving its accuracy for the next iteration.

This loop continues until a stopping criterion is met (e.g., model convergence or maximum iterations).

## Structure Generation Strategies

`mlipflow` offers several strategies for generating new structures, all inheriting from `StructureGenStrategy`. These are located in `mlipflow.strategies.structure_generators`.

*   **`MDGen` (Molecular Dynamics)**: Runs MD simulations using the MLIP to explore the phase space. Useful for sampling diverse configurations.
*   **`OPTGen` (Geometry Optimization)**: Relaxes structures to their local minima. Useful for finding stable intermediates.
*   **`NEBGen` (Nudged Elastic Band)**: Performs NEB calculations to find transition states between reactant and product pairs. This is critical for pathway exploration.

## MLIP Strategies

The package supports different Machine Learning Potential architectures via `MLIPStrategy`, located in `mlipflow.strategies.mlip`.

*   **`MACEModel`**: Interface for the **MACE** (Multi-ACE) architecture. It supports running MACE models locally or remotely and includes optional DFT-D3 dispersion corrections.
*   **`GAPModel`**: Interface for the **GAP** (Gaussian Approximation Potential) framework. It supports multistage fitting (e.g., 2-body + SOAP).

## DFT Strategies

For the "ground truth" calculations, `mlipflow` uses `QChemStrategy`, located in `mlipflow.strategies.dft`.

*   **`QECalculator`**: Interface for **Quantum Espresso**. It handles input file generation, pseudopotentials, and remote job submission for SCF and relaxation calculations.
*   **`EMTCalc`**: A fast, approximate calculator (Effective Medium Theory) used primarily for testing and debugging workflows without the cost of full DFT.
