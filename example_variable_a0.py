"""
Example: Variable Learning Rates (a0) per Beta Value

This script demonstrates how to use different learning rates for different beta values
in the SGLD experiments, exactly as requested.
"""

from training import (run_beta_experiments, create_beta_a0_mapping, 
                     create_adaptive_a0_function)

def main():
    """Demonstrate variable a0 per beta."""
    
    # Define beta values to test
    beta_values = [0, 1, 10, 50, 200]
    
    # Option 1: Dictionary mapping (recommended for precise control)
    a0_dict = {
        0: 0.01,   # Beta=0: low learning rate
        1: 0.01,   # Beta=1: low learning rate  
        10: 0.1,   # Beta=10: higher learning rate
        50: 0.1,   # Beta=50: higher learning rate
        200: 0.1   # Beta=200: higher learning rate
    }
    
    # Also define variable epochs per beta (from previous example)
    epochs_dict = {
        0: 100,    # Beta=0: 100 epochs
        1: 100,    # Beta=1: 100 epochs
        10: 1000,  # Beta=10: 1000 epochs
        50: 1000,  # Beta=50: 1000 epochs
        200: 1000  # Beta=200: 1000 epochs
    }
    
    print("SGLD Experiments with Variable Learning Rates per Beta")
    print("=" * 60)
    print("Configuration:")
    print("Learning rates (a0):")
    for beta, a0_val in a0_dict.items():
        print(f"  Beta {beta:>3}: a0 = {a0_val}")
    print("Epochs:")
    for beta, epochs in epochs_dict.items():
        print(f"  Beta {beta:>3}: {epochs} epochs")
    print()
    
    # Run experiments with both variable a0 and variable epochs
    results = run_beta_experiments(
        beta_values=beta_values,
        num_repetitions=10,  # Small number for quick demo
        num_epochs=epochs_dict,  # Variable epochs
        a0=a0_dict,             # Variable learning rates - NEW!
        b=0.5,
        sigma_gauss_prior=1000000,
        dataset_type='synth',
        use_random_labels=False
    )
    
    print("\nExperiment completed!")
    print("Notice how different beta values used different learning rates and epochs.")
    return results


def demonstrate_different_a0_options():
    """Show different ways to specify variable a0."""
    
    beta_values = [0, 1, 10, 50, 200]
    
    print("Different ways to specify variable a0:")
    print("=" * 50)
    
    # Option 1: Dictionary mapping
    a0_dict = {0: 0.01, 1: 0.01, 10: 0.1, 50: 0.1, 200: 0.1}
    print("1. Dictionary mapping:")
    print(f"   a0_dict = {a0_dict}")
    
    # Option 2: Using convenience function
    a0_mapping = create_beta_a0_mapping([
        (0, 0.01), (1, 0.01), (10, 0.1), (50, 0.1), (200, 0.1)
    ])
    print("\\n2. Convenience function mapping:")
    print(f"   a0_mapping = {a0_mapping}")
    
    # Option 3: Adaptive function
    a0_func = create_adaptive_a0_function(
        low_beta_a0=0.01,   # For beta <= 5
        high_beta_a0=0.1,   # For beta > 5
        threshold=5.0
    )
    print("\\n3. Adaptive function:")
    print("   a0_func = create_adaptive_a0_function(0.01, 0.1, 5.0)")
    for beta in beta_values:
        print(f"   a0_func({beta}) = {a0_func(beta)}")
    
    # Option 4: Lambda function
    a0_lambda = lambda beta: 0.01 if beta <= 5 else 0.1
    print("\\n4. Lambda function:")
    print("   a0_lambda = lambda beta: 0.01 if beta <= 5 else 0.1")
    for beta in beta_values:
        print(f"   a0_lambda({beta}) = {a0_lambda(beta)}")
    
    # Option 5: Complex custom function
    def complex_a0_function(beta):
        """Custom function for learning rate based on beta value."""
        if beta == 0:
            return 0.005  # Very low for pure noise
        elif beta <= 1:
            return 0.01   # Low for low beta
        elif beta <= 10:
            return 0.05   # Medium for moderate beta
        else:
            return 0.1    # High for high beta
    
    print("\\n5. Complex custom function:")
    for beta in beta_values:
        print(f"   complex_a0_function({beta}) = {complex_a0_function(beta)}")


if __name__ == "__main__":
    # Show different options
    demonstrate_different_a0_options()
    
    print("\\n" + "="*60)
    print("Running example experiment...")
    print("="*60)
    
    # Run the main example (uncomment to actually run)
    # results = main()
    print("Example ready! Uncomment the main() call to run the experiment.")
