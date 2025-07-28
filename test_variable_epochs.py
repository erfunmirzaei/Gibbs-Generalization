"""
Quick test to demonstrate variable epochs functionality
"""

from training import run_beta_experiments

def quick_test():
    """Quick test with very few repetitions to show the functionality."""
    
    beta_values = [1, 10]
    
    # 100 epochs for beta=1, 1000 epochs for beta=10
    epochs_dict = {
        1: 100,
        10: 1000
    }
    
    print("Quick test: Variable epochs per beta")
    print(f"Beta 1: {epochs_dict[1]} epochs")
    print(f"Beta 10: {epochs_dict[10]} epochs")
    print()
    
    # Run with just 2 repetitions for quick demo
    results = run_beta_experiments(
        beta_values=beta_values,
        num_repetitions=2,
        num_epochs=epochs_dict,
        a0=0.1,
        sigma_gauss_prior=1000000
    )
    
    print("Test completed! You should see different epoch counts in the output above.")
    return results

if __name__ == "__main__":
    quick_test()
