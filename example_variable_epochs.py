"""
Example: Variable Epochs per Beta Value

This script demonstrates how to train with fewer epochs for small beta values
and more epochs for larger beta values, exactly as requested.
"""

from training import run_beta_experiments
from bounds import save_results_to_file
from plot_utils import plot_beta_results

def main():
    """Run experiments with variable epochs per beta."""
    
    # Define beta values to test
    beta_values = [0, 1, 10, 50, 200]
    
    # Define epochs per beta - exactly what you requested!
    epochs_dict = {
        0: 100,    # Beta=0: only 100 epochs
        1: 100,    # Beta=1: only 100 epochs  
        10: 1000,  # Beta=10: 1000 epochs
        50: 1000,  # Beta=50: 1000 epochs
        200: 1000  # Beta=200: 1000 epochs
    }
    
    print("SGLD Experiments with Variable Epochs per Beta")
    print("=" * 50)
    print("Configuration:")
    for beta, epochs in epochs_dict.items():
        print(f"  Beta {beta:>3}: {epochs:>4} epochs")
    print()
    
    # Run experiments
    results = run_beta_experiments(
        beta_values=beta_values,
        num_repetitions=10,  # You can increase this for more robust results
        num_epochs=epochs_dict,  # Pass the dictionary here!
        a0=0.1,
        b=0.5,
        sigma_gauss_prior=1000000,
        dataset_type='synth',
        use_random_labels=False
    )
    
    # Save results
    save_results_to_file(
        results, 
        beta_values=beta_values,
        num_repetitions=10,
        num_epochs=epochs_dict,  # This handles variable epochs in filename
        a0=0.1,
        sigma_gauss_prior=1000000,
        dataset_type='synth'
    )
    
    # Create plots
    plot_beta_results(
        results,
        beta_values=beta_values,
        num_repetitions=10,
        num_epochs=epochs_dict,  # This handles variable epochs in filename
        a0=0.1,
        sigma_gauss_prior=1000000,
        dataset_type='synth'
    )
    
    print("\nExperiment completed!")
    print("Notice the training output showed different epochs for each beta value.")


if __name__ == "__main__":
    main()
