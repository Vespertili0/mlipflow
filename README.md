# mlipflow
**A Python Package for Reaction Pathway Exploration using Machine-Learned Interatomic Potentials (MLIPs)**

mlipflow bridges the critical gap between high-level machine learning development and practical catalysis research. It provides a modular, scalable framework to automate the exploration of complex reaction pathways with near-quantum accuracy at a fraction of the traditional computational cost.

>### 🚩 The Challenge: The "Catalysis Bottleneck"
>
>Traditional heterogeneous catalysis research faces three major hurdles that mlipflow is built to solve:
>
>The Cost-Accuracy Trade-off: Density Functional Theory (DFT) is the gold standard for accuracy but is too slow for exhaustive pathway exploration.
>
>Human Selection Bias: Researchers often manually "guess" reaction pathways, potentially missing the actual rate-determining steps or low-energy configurations.
>
>Architecture Lock-in: Most existing MLIP packages are tied to specific, often outdated, ML architectures. Transitioning to state-of-the-art models like MACE typically requires a complete rewrite of the simulation workflow.

## ✨ Key Features

Modularity: Built on ASE and LangGraph, allowing for modular extension of functions and custom calculators.

Scalability: Integrated with WFL to support seamless execution on HPC platforms, scaling from small tests to massive datasets.

Extensibility: Supports multiple strategies for structure generation (e.g., MD, optimization, NEB) and MLIP training (e.g., GAP, MACE)..

## 🔄 The Iterative Workflow

mlipflow employs an active learning loop to refine the Potential Energy Surface (PES):

Generate: Create new configurations (e.g., via MLIP-NEB) to explore unknown regions.

Compute: Automatically trigger high-level DFT (Quantum Espresso) for "ground truth" data.

Train: Update the MLIP model, refining its accuracy for the next iteration of exploration.

## Installation
```bash
$ git install https://github.com/Vespertili0/mlipflow.git
```

## Usage
Code examples demonstrating primary functions and expected outcomes, including graphical outputs if applicable.

## Features
- Easy to use and extend
- Optimized for performance
- Comprehensive error handling

## Code Structure & Roadmap
### Code Structure
This package follows an object-oriented approach, with core classes interacting as follows:

```mermaid
classDiagram
    class DataManager {
        +base_dir: str
        +setup_iteration(iteration: int): str
        +get_file_path(iteration: int, filename: str): str
    }

    class ActiveLearner {
        -structure_generation_strategy: StructureGenStrategy
        -mlip_strategy: MLIPStrategy
        -qchem_strategy: QChemStrategy
        -iteration: int
        +initialise_learning(ensemble_traj, initial_xyz, qchem: bool)
        +run_iteration(n_iter: int)
        +run_single_point(in_file, out_file, output_prefix, calculator, remote_info)
        +calculate_mlip_error(in_configs, out_file, fit_idx, iter_dir)
        +clean_up()
    }

    class StructureGenStrategy {
        +generate_new_structures()
    }

    class MDGen {
        +generate_new_structures(in_file, out_file, calculator, remote_info)
    }

    class OPTGen {
        +generate_new_structures(in_file, out_file, calculator, remote_info)
    }

    class NEBGen {
        +generate_new_structures()
    }

    class MLIPStrategy {
        +get_calculator(job_name)
        +fit_new_model(in_file, model_name, run_dir)
    }

    class GAPModel {
        +get_calculator(job_name)
        +fit_new_model(in_file, model_name, run_dir)
    }

    class MACEModel {
        +get_calculator(job_name)
        +fit_new_model(in_file, model_name, run_dir, config_file)
    }

    class QChemStrategy {
        +get_calculator(job_name)
    }

    class EMTCalc {
        +get_calculator(job_name)
    }

    class QECalculator {
        +get_calculator(job_name, ecut_eV, kpts, dipole, dftd3)
    }

    ActiveLearner --> StructureGenStrategy : uses
    ActiveLearner --> MLIPStrategy : uses
    ActiveLearner --> QChemStrategy : uses
    StructureGenStrategy <|-- MDGen : extends
    StructureGenStrategy <|-- OPTGen : extends
    StructureGenStrategy <|-- NEBGen : extends
    MLIPStrategy <|-- GAPModel : extends
    MLIPStrategy <|-- MACEModel : extends
    QChemStrategy <|-- EMTCalc : extends
    QChemStrategy <|-- QECalculator : extends
```

### Roadmap
Planned improvements:
- ...


## Contributing
Guidelines for developers interested in contributing.

## Testing
Instructions for running tests and CI integration info.

## License
MIT License.

## Contact
Your email or professional social media links.
