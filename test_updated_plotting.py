"""
Quick test for the updated plotting functionality with KL analysis for zero-one loss
"""

import numpy as np
import matplotlib.pyplot as plt

def test_updated_plotting():
    """Test the updated plotting with KL analysis but no generalization bounds."""
    
    # Create mock results data
    beta_values = [1, 10, 50]
    results = {}
    
    for beta in [0] + beta_values:  # Include beta=0 for bounds computation
        results[beta] = {
            'train_bce_mean': 0.4 - 0.05 * beta,
            'test_bce_mean': 0.5 - 0.03 * beta,
            'train_bce_std': 0.02,
            'test_bce_std': 0.03,
            'train_01_mean': 0.45 - 0.04 * beta,
            'test_01_mean': 0.55 - 0.02 * beta,
            'train_01_std': 0.025,
            'test_01_std': 0.035,
            'raw_train_bce': [0.4 - 0.05 * beta + np.random.normal(0, 0.02) for _ in range(5)],
            'raw_test_bce': [0.5 - 0.03 * beta + np.random.normal(0, 0.03) for _ in range(5)],
            'raw_train_01': [0.45 - 0.04 * beta + np.random.normal(0, 0.025) for _ in range(5)],
            'raw_test_01': [0.55 - 0.02 * beta + np.random.normal(0, 0.035) for _ in range(5)]
        }
    
    try:
        from plot_utils import plot_beta_results
        
        print("Testing updated plotting functionality...")
        print("This should create plots with:")
        print("- Train/Test losses only (no generalization bounds/errors)")
        print("- KL analysis for both BCE and Zero-One losses")
        print()
        
        # Test the plotting
        plot_beta_results(
            results=results,
            beta_values=beta_values,
            num_repetitions=5,
            num_epochs=1000,
            a0=0.1,
            sigma_gauss_prior=10,
            dataset_type='synth'
        )
        
        print("✅ Plotting test completed successfully!")
        print("Check the generated plot - it should show:")
        print("  Left plot: BCE Train/Test + KL analysis")
        print("  Right plot: Zero-One Train/Test + KL analysis")
        print("  No generalization bounds or errors")
        
    except Exception as e:
        print(f"❌ Error in plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_updated_plotting()
