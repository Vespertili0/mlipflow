from __future__ import annotations

import os

from mlipflow.data.semi_supervised_gmm import GMMLabelChecker
from mlipflow.strategies.mlip import MACEModel


def test_semi_supervised_gmm_pipeline():
    """Test the 5-step semi-supervised GMM refinement pipeline using GMMRefiner."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_model = os.path.join(test_dir, "data", "mace_test")
    train_file = os.path.join(test_dir, "data", "test_data.xyz")
    pool_file = os.path.join(test_dir, "data", "test_data_II.xyz")

    # Initialise the MLIP Strategy using the test model
    strategy = MACEModel(mlip_name=mlip_model)

    # Run the pipeline using a subset for train and another for pool.
    # We use a low high_certainty threshold to ensure some configs pass to step 4 (refit).
    try:
        pipeline = GMMLabelChecker(
            train_file=train_file,
            pool_file=pool_file,
            mlip_strategy=strategy,
            device="cpu",
            high_certainty=0.1,  # Very low to guarantee retention for the refit step
            final_certainty=0.0,  # 0.0 to retain everything at the end so we can count
            pca_threshold=0.95,
            gmm_iters=2,  # Fast fit for testing
        )

        certain_configs, uncertain_configs, certainty_scores = pipeline.run()

        # Check that it returns the expected types
        assert isinstance(certain_configs, list)
        assert isinstance(uncertain_configs, list)
        assert (len(certain_configs) + len(uncertain_configs)) > 0
        assert (len(certain_configs) + len(uncertain_configs)) == len(certainty_scores)

        # Check that the retained configs have the new annotation and correct labels
        for config in certain_configs:
            assert "gmm_certainty" in config.info
            assert config.info["species"] != "unknown"

        for config in uncertain_configs:
            assert "gmm_certainty" in config.info
            assert config.info["species"] == "unknown"

    finally:
        pass
