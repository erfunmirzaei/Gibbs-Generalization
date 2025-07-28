"""
Quick test for variable a0 functionality
"""

from training import run_beta_experiments

def quick_test():
    """Quick test with variable a0 and epochs per beta."""
    
    beta_values = [1, 10]
    
    # Different epochs per beta
    epochs_dict = {
        1: 100,   # Beta=1: 100 epochs
        10: 1000  # Beta=10: 1000 epochs
    }
    
    # Different learning rates per beta
    a0_dict = {
        1: 0.01,  # Beta=1: low learning rate
        10: 0.1   # Beta=10: higher learning rate
    }
    
    print("Quick test: Variable a0 and epochs per beta")
    print("Configuration:")
    for beta in beta_values:
        print(f"  Beta {beta}: {epochs_dict[beta]} epochs, a0 = {a0_dict[beta]}")
    print()
    
    # Run with just 2 repetitions for quick demo
    results = run_beta_experiments(
        beta_values=beta_values,
        num_repetitions=2,
        num_epochs=epochs_dict,  # Variable epochs
        a0=a0_dict,             # Variable learning rates
        sigma_gauss_prior=1000000
    )
    
    print("Test completed! Check the output above to see:")
    print("- Different epochs per beta in the training output")
    print("- Different learning rates (a0) per beta in the configuration")
    return results

if __name__ == "__main__":
    quick_test()
