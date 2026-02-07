# mlipflow
> A succinct tagline explaining what it does.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Code Structure & Roadmap](#roadmap)
- [Contributing](#contributing)
- [Testing](#testing)
- [License](#license)
- [Contact](#contact)

## Overview
What and Why: Immediately explain the motivation behind the package. What problem does it solve? Why did you build it?
Audience: Briefly mention who the package is for—whether it’s meant for data scientists, web developers, or another niche.
Value Proposition: Emphasize what makes your implementation unique. This not only highlights your technical skill but also your design choices.

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
