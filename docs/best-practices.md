---
title: Best Practices
layout: default
---

# Best Practices

Ensure your active learning workflows are efficient, robust, and accurate by following these recommendations.

## 🚀 Performance Optimisation

-   **HPC Chunking**: When running large sets of configurations (e.g., thousands of MD images), use the `run_chunked_qe_sp` or similar batching functions within nodes to avoid hitting scheduler limits.
-   **GPU Parallelism**: Ensure `mace-torch` is configured to use multiple GPUs if available, especially during the training phases of the active learning loop.
-   **Data Management**: Periodically clean up temporary calculation files using `mlipflow.data.clean_up()` to save disk space on shared systems.

## 🎯 Model Accuracy

-   **Diverse Seeding**: Start your active learning loop with a diverse pool of configurations (e.g., varying coverage, different surface sites).
-   **Uncertainty Thresholds**: Carefully tune your GMM uncertainty thresholds. Too restrictive, and you skip important data; too loose, and you waste DFT budget on redundant configurations.
-   **Validation Set**: Always maintain a separate, "unseen" test set to monitor the generalisation of your MLIP model across iterations.

## 🛡️ Avoiding Pitfalls

-   **Configuration Flooding**: Avoid generating single trajectories that are too long. Instead, use multiple short "burst" trajectories from different starting points to explore the PES more effectively.
-   **Force Agreement**: When merging new DFT data, filter for configurations where the previous MLIP model showed high force errors. These are the most valuable points for the next training cycle.
-   **Atomic Displacement**: In structure generation, use a reasonable `dr_threshold` (e.g., 0.3 - 0.5 Å) to avoid saving nearly identical atomic images, which leads to slow training and poor convergence.

## 🛠️ Debugging

-   **Log Files**: `mlipflow` uses standard Python logging. Set the log level to `DEBUG` in your run script to see detailed wfl and ASE output.
-   **State Inspection**: Since LangGraph is state-persistent, you can inspect the `EnsembleState` dictionary at any node to verify that file paths and calculation parameters are propagating correctly.
