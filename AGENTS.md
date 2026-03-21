# AGENTS.md - mlipflow Context & Reminders

## Project Goal
mlipflow bridges the critical gap between high-level machine learning development and practical catalysis research. It provides a modular, scalable framework to automate the exploration of complex reaction pathways with near-quantum accuracy at a fraction of the computational cost, employing an active learning loop to refine the Potential Energy Surface (PES).

## Key Constraints & Environment
- **OS**: Linux
- **Core Libraries**: 
  - `wfl`: Workflow execution and parallelisation.
  - `mace-torch`: State-of-the-art MLIP model support.
  - `quippy-ase`: GAP model support and ASE integration.
  - `langgraph`: Modular active learning loop and workflow orchestration.
  - `torch`: GPU-accelerated operations, e.g., for MACE and GMM refinement.
- **Primary Mechanism**: Active learning loop (Generate -> Compute -> Train) using `langgraph` and `wfl` to orchestrate structure generation (MD, OPT, NEB), quantum chemistry calculations (QE), and MLIP training.

## Architecture
- **`mlipflow/`**: Main package directory.
  - **`adapters/`**: Interfaces and adapters for various calculators and tools.
  - **`core/`**: Core logic including `DataManager` and `ActiveLearner` classes.
  - **`data/`**: Data processing, context management, and GMM-based configuration selection.
  - **`graphflow/`**: LangGraph nodes and state definitions for the active learning workflow.
  - **`strategies/`**: Implementations for `StructureGenStrategy`, `MLIPStrategy`, and `QChemStrategy`.
- **`tests/`**: Pytest-based testing framework.

### CI/CD
- GitHub Actions workflows (`.github/workflows/`) for testing (`tests.yaml`), documentation building (`docs.yaml`), and releases (`release.yaml`).

## Current Status
- Implemented modular active learning loop with ASE and LangGraph.
- Developed and integrated Semi-Supervised GMM Refinement for chemical configurations.
- Optimised dictionary lookups and general data processing performance.

## Reminders
- [ ] Ensure all code passes `ruff` linting with `numpy` docstring conventions as specified in `pyproject.toml`.
- [ ] Run tests via `python -m pytest tests/` before committing.

## Next Steps
1. **Integrate and Test Semi-Supervised GMM in Active Learner**: Ensure the standalone GMM refinement script is fully embedded into the `ActiveLearner` loop, complete with relevant Pytest coverage to validate its robustness.
2. **Expand Structure Generation Strategies**: Flesh out the NEB and OPT generation strategies to fully leverage the newly optimized MACE/GMM uncertainty predictions for targeting specific exploratory edge cases.
3. **Enhance Documentation and Workflows**: Populate the `README.md` with practical code examples and refine the `docs/` folder to effectively guide users on composing custom LangGraph workflows.
