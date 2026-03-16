import os
import pytest
from ase.io import read
from mace.calculators import MACECalculator

from mlipflow.data.semi_supervised_gmm import GMMRefiner


def test_semi_supervised_gmm_pipeline():
    """Test the 5-step semi-supervised GMM refinement pipeline using GMMRefiner."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlip_model = os.path.join(test_dir, 'data', 'mace_test.model')
    test_data = os.path.join(test_dir, 'data', 'test_data.xyz')
    
    # Read the data
    configs = read(test_data, ':')
    
    # The pipeline requires 'species' in atoms.info to determine K (number of clusters)
    # The test data might not have it, so we'll assign a dummy species label.
    # We assign two species to test multi-component GMM fitting.
    for i, at in enumerate(configs):
        at.info['species'] = 'A' if i % 2 == 0 else 'B'
        
    # Initialise the calc using the test model
    calc = MACECalculator(model_paths=mlip_model, device='cpu', default_dtype='float32')
    
    # Run the pipeline using a subset for train and another for pool.
    # We use a low high_certainty threshold to ensure some configs pass to step 4 (refit).
    pipeline = GMMRefiner(
        train_configs=configs[:5],
        pool_configs=configs[5:15],
        calc=calc,
        device='cpu',
        high_certainty=0.1,  # Very low to guarantee retention for the refit step
        final_certainty=0.0, # 0.0 to retain everything at the end so we can count
        pca_threshold=0.95,
        gmm_iters=2          # Fast fit for testing
    )
    
    final_configs, certainty_scores = pipeline.run()
    
    # Check that it returns the expected types
    assert isinstance(final_configs, list)
    assert len(final_configs) > 0
    assert len(final_configs) == len(certainty_scores)
    
    # Check that the retained configs have the new annotation
    for config in final_configs:
        assert 'gmm_certainty' in config.info
