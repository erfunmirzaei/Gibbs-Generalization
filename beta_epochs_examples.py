"""
Beta-Specific Epochs Configuration Examples

This file demonstrates the different ways to configure variable epochs
per beta value in the SGLD experiments.
"""

from training import (run_beta_experiments, create_beta_epochs_mapping, 
                     create_adaptive_epochs_function)

# Example 1: Dictionary mapping (explicit epochs per beta)
epochs_dict = {
    0: 100,    # Beta=0: only 100 epochs (fast noise convergence)
    1: 100,    # Beta=1: 100 epochs
    10: 1000,  # Beta=10: 1000 epochs
    50: 1000,  # Beta=50: 1000 epochs
    200: 1000  # Beta=200: 1000 epochs
}

# Example 2: Using convenience function to create mapping
epochs_mapping = create_beta_epochs_mapping([
    (0, 100), (1, 100), (10, 1000), (50, 1000), (200, 1000)
])

# Example 3: Adaptive function (automatic threshold-based)
epochs_func = create_adaptive_epochs_function(
    low_beta_epochs=100,   # For beta <= 5
    high_beta_epochs=1000, # For beta > 5
    threshold=5.0
)

# Example 4: Custom lambda function
epochs_lambda = lambda beta: 100 if beta <= 5 else 1000

# Example 5: More complex custom function
def complex_epochs_function(beta):
    """Custom function for epochs based on beta value."""
    if beta == 0:
        return 50   # Very fast for pure noise
    elif beta <= 1:
        return 100  # Fast for low beta
    elif beta <= 10:
        return 500  # Medium for moderate beta
    else:
        return 1000 # Full training for high beta


def run_example_experiment():
    """Run an example experiment with beta-specific epochs."""
    
    beta_values = [1, 10, 50, 200]
    
    print("Running SGLD experiment with variable epochs per beta...")
    
    # Use the dictionary approach
    results = run_beta_experiments(
        beta_values=beta_values,
        num_repetitions=3,
        num_epochs=epochs_dict,  # <-- This is the key change!
        a0=1e-1,
        b=0.5,
        sigma_gauss_prior=10,
        dataset_type='synth'
    )
    
    print("\\nExperiment completed!")
    print("Notice how beta=0 and beta=1 used only 100 epochs,")
    print("while beta=10, 50, 200 used 1000 epochs each.")
    
    return results


if __name__ == "__main__":
    # Uncomment to run the example
    # results = run_example_experiment()
    
    print("Beta-specific epochs configuration examples:")
    print("\\n1. Dictionary mapping:")
    print(f"   epochs_dict = {epochs_dict}")
    
    print("\\n2. Convenience function mapping:")
    print(f"   epochs_mapping = {epochs_mapping}")
    
    print("\\n3. Adaptive function:")
    print("   epochs_func = create_adaptive_epochs_function(100, 1000, 5.0)")
    for beta in [0, 1, 5, 10, 50]:
        print(f"   epochs_func({beta}) = {epochs_func(beta)}")
    
    print("\\n4. Lambda function:")
    print("   epochs_lambda = lambda beta: 100 if beta <= 5 else 1000")
    for beta in [0, 1, 5, 10, 50]:
        print(f"   epochs_lambda({beta}) = {epochs_lambda(beta)}")
    
    print("\\n5. Complex custom function:")
    for beta in [0, 1, 5, 10, 50, 100]:
        print(f"   complex_epochs_function({beta}) = {complex_epochs_function(beta)}")
