# Installation

This guide will help you install `mlipflow` and its dependencies.

## Prerequisites

*   **Python 3.9** or higher is required.
*   **Git** for cloning the repository.
*   **pip** for installing Python packages.

## Installation Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Vespertili0/mlipflow.git
    cd mlipflow
    ```

2.  **Create a Virtual Environment (Recommended)**

    It is best practice to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies and Package**

    Install the package in editable mode along with development dependencies:

    ```bash
    pip install -e .[dev]
    ```

    This command will automatically install the required dependencies listed in `pyproject.toml`, including:
    *   `wfl`: Workflow framework for atomistic simulations.
    *   `mace-torch`: Machine Learning Interatomic Potentials (MACE).
    *   `quippy-ase`: Interface for QUIP potentials and descriptors.
    *   `torch-dftd`: DFT-D3/D4 dispersion corrections.
    *   `langgraph`: Graph-based workflow orchestration.

## Verifying Installation

To verify that `mlipflow` is installed correctly, you can run the tests using `pytest`:

```bash
pytest tests/
```

If all tests pass, you are ready to use `mlipflow`!
