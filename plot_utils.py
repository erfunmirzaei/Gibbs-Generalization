"""
Plotting utilities for the Gibbs generalization bound experiments.

This modu    # Create comprehensive plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define consistent colors
    train_color = 'blue'
    test_color = 'orange'
    gen_error_color = 'green'
    bound_color = 'red'
    individual_bound_color = 'purple'
    kl_color = 'brown's functions for creating visualizations of experimental results,
including generalization error plots, bound comparisons, and training curves.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_beta_results(results, n, filename=None, beta_values=None, num_repetitions=None, num_epochs=None, 
                     a0=None, sigma_gauss_prior=None, dataset_type='synth', use_random_labels=False, hyperparams=None):
    """
    Plot the generalization errors with confidence intervals, generalization bounds, 
    individual bounds, KL divergence analysis, and test error bounds across different beta values.
    
    Args:
        results: Dictionary containing experimental results for each beta value
        n: Training set size (required for bounds computation)
        filename: Optional custom filename. If None, generates descriptive filename
        beta_values: List of beta values (for filename generation). If None, uses results.keys()
        num_repetitions: Number of repetitions per beta (for filename generation)
        num_epochs: Number of training epochs (for filename generation)
        a0: Learning rate (for filename generation)
        sigma_gauss_prior: Prior parameter (for filename generation)
        dataset_type: Dataset type (for filename generation)
        use_random_labels: Whether random labels were used (for filename generation)
        hyperparams: Full hyperparameter dictionary for hash generation
    """
    from bounds import (compute_generalization_bound, compute_generalization_errors, 
                       compute_individual_generalization_bounds, compute_kl_divergence_analysis, 
                       generate_filename)
    
    # Use beta_values from results if not provided
    if beta_values is None:
        beta_values = sorted(results.keys())
    else:
        beta_values = sorted(beta_values)
    
    # For bounds computation, we need all beta values in results (including beta=0 if present)
    all_beta_values = sorted(results.keys())
    
    # Compute all analyses
    gen_errors = compute_generalization_errors(all_beta_values, results)
    bounds = compute_generalization_bound(beta_values, results, n, loss_type='bce')
    zero_one_bounds = compute_generalization_bound(beta_values, results, n, loss_type='zero_one')
    individual_bounds = compute_individual_generalization_bounds(beta_values, results, n, loss_type='bce')
    individual_zero_one_bounds = compute_individual_generalization_bounds(beta_values, results, n, loss_type='zero_one')
    kl_analysis = compute_kl_divergence_analysis(beta_values, results, n, loss_type='bce')
    kl_analysis_zo = compute_kl_divergence_analysis(beta_values, results, n, loss_type='zero_one')
    
    # Extract data for plotting
    bce_gen_errors = [gen_errors[beta]['bce_gen_error'] for beta in beta_values]
    bce_gen_error_stds = [gen_errors[beta]['bce_gen_error_std'] for beta in beta_values]
    zero_one_gen_errors = [gen_errors[beta]['zero_one_gen_error'] for beta in beta_values]
    zero_one_gen_error_stds = [gen_errors[beta]['zero_one_gen_error_std'] for beta in beta_values]
    
    # Train/test errors
    train_bce_means = [results[beta]['train_bce_mean'] for beta in beta_values]
    test_bce_means = [results[beta]['test_bce_mean'] for beta in beta_values]
    train_bce_stds = [results[beta]['train_bce_std'] for beta in beta_values]
    test_bce_stds = [results[beta]['test_bce_std'] for beta in beta_values]
    
    train_01_means = [results[beta]['train_01_mean'] for beta in beta_values]
    test_01_means = [results[beta]['test_01_mean'] for beta in beta_values]
    train_01_stds = [results[beta]['train_01_std'] for beta in beta_values]
    test_01_stds = [results[beta]['test_01_std'] for beta in beta_values]
    
    # Bounds
    theoretical_bounds = [bounds[beta]['generalization_bound'] for beta in beta_values]
    zero_one_theoretical_bounds = [zero_one_bounds[beta]['generalization_bound'] for beta in beta_values]
    individual_bound_means = [individual_bounds[beta]['bound_mean'] for beta in beta_values]
    individual_bound_stds = [individual_bounds[beta]['bound_std'] for beta in beta_values]
    individual_zo_bound_means = [individual_zero_one_bounds[beta]['bound_mean'] for beta in beta_values]
    individual_zo_bound_stds = [individual_zero_one_bounds[beta]['bound_std'] for beta in beta_values]
    
    # KL analysis data
    kl_means = [kl_analysis[beta]['kl_mean'] for beta in beta_values]
    kl_stds = [kl_analysis[beta]['kl_std'] for beta in beta_values]
    kl_bound_means = [kl_analysis[beta]['kl_bound_mean'] for beta in beta_values]
    kl_bound_stds = [kl_analysis[beta]['kl_bound_std'] for beta in beta_values]
    test_bound_means = [kl_analysis[beta]['test_bound_mean'] for beta in beta_values]
    test_bound_stds = [kl_analysis[beta]['test_bound_std'] for beta in beta_values]
    
    # Zero-one KL analysis data
    kl_zo_means = [kl_analysis_zo[beta]['kl_mean'] for beta in beta_values]
    kl_zo_stds = [kl_analysis_zo[beta]['kl_std'] for beta in beta_values]
    kl_zo_bound_means = [kl_analysis_zo[beta]['kl_bound_mean'] for beta in beta_values]
    kl_zo_bound_stds = [kl_analysis_zo[beta]['kl_bound_std'] for beta in beta_values]
    test_bound_zo_means = [kl_analysis_zo[beta]['test_bound_mean'] for beta in beta_values]
    test_bound_zo_stds = [kl_analysis_zo[beta]['test_bound_std'] for beta in beta_values]
    
    # Create comprehensive plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define consistent colors
    train_color = 'blue'
    test_color = 'orange'
    gen_error_color = 'green'
    bound_color = 'red'
    individual_bound_color = 'purple'
    kl_color = 'brown'
    
    # Plot 1: BCE Train/Test + KL Analysis (no generalization bounds/errors)
    ax1.errorbar(beta_values, train_bce_means, yerr=train_bce_stds, 
                 fmt='o-', label='Train BCE', linewidth=2, markersize=5, capsize=3, color=train_color)
    ax1.errorbar(beta_values, test_bce_means, yerr=test_bce_stds, 
                 fmt='s-', label='Test BCE', linewidth=2, markersize=5, capsize=3, color=test_color)
    ax1.errorbar(beta_values, test_bound_means, yerr=test_bound_stds, 
                 fmt='p-', label='Test Bound (via KL)', linewidth=2, markersize=5, capsize=3, color=kl_color)
    ax1.errorbar(beta_values, kl_means, yerr=kl_stds, 
                 fmt='h-', label='KL(train||test)', linewidth=2, markersize=4, capsize=3, color='darkblue')
    ax1.errorbar(beta_values, kl_bound_means, yerr=kl_bound_stds, 
                 fmt='*-', label='KL Bound', linewidth=2, markersize=4, capsize=3, color='gray')
    
    ax1.set_xlabel('Beta (Inverse Temperature)')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('BCE: Train/Test Losses & KL Analysis')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    # Use linear scale for x-axis
    ax1.set_xscale('linear')
    
    # Plot 2: Zero-One Train/Test + KL Analysis (no generalization bounds/errors)
    ax2.errorbar(beta_values, train_01_means, yerr=train_01_stds, 
                 fmt='o-', label='Train 0-1', linewidth=2, markersize=5, capsize=3, color=train_color)
    ax2.errorbar(beta_values, test_01_means, yerr=test_01_stds, 
                 fmt='s-', label='Test 0-1', linewidth=2, markersize=5, capsize=3, color=test_color)
    ax2.errorbar(beta_values, test_bound_zo_means, yerr=test_bound_zo_stds, 
                 fmt='p-', label='Test Bound (via KL)', linewidth=2, markersize=5, capsize=3, color=kl_color)
    ax2.errorbar(beta_values, kl_zo_means, yerr=kl_zo_stds, 
                 fmt='h-', label='KL(train||test)', linewidth=2, markersize=4, capsize=3, color='darkblue')
    ax2.errorbar(beta_values, kl_zo_bound_means, yerr=kl_zo_bound_stds, 
                 fmt='*-', label='KL Bound', linewidth=2, markersize=4, capsize=3, color='gray')
    
    ax2.set_xlabel('Beta (Inverse Temperature)')
    ax2.set_ylabel('Loss Value')
    ax2.set_title('Zero-One: Train/Test Losses & KL Analysis')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    # Use linear scale for x-axis
    ax2.set_xscale('linear')
    
    plt.tight_layout()
    
    # Generate descriptive filename if not provided
    if filename is None and all(param is not None for param in [num_repetitions, num_epochs, a0, sigma_gauss_prior]):
        filename = generate_filename(
            beta_values=beta_values,
            num_repetitions=num_repetitions,
            num_epochs=num_epochs,
            a0=a0,
            sigma_gauss_prior=sigma_gauss_prior,
            dataset_type=dataset_type,
            file_type='plot',
            extension='png',
            use_random_labels=use_random_labels,
            hyperparams=hyperparams
        )
        filename = f"results/{filename}"
    elif filename is None:
        # Fallback to default naming with random labels info
        labels_str = "randlabels" if use_random_labels else "reallabels"
        filename = f'results/sgld_plot_{dataset_type}_{labels_str}_beta_experiments.png'
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{filename}'")
    
    plt.show()


def plot_training_curves(train_losses, test_losses, train_zero_one_losses, test_zero_one_losses, 
                        learning_rates=None, beta=None, save_path=None):
    """
    Plot training curves for a single experiment.
    
    Args:
        train_losses: List of training losses over epochs
        test_losses: List of test losses over epochs
        train_zero_one_losses: List of training zero-one losses over epochs
        test_zero_one_losses: List of test zero-one losses over epochs
        learning_rates: Optional list of learning rates over epochs
        beta: Beta value for the experiment (for title)
        save_path: Optional path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    if learning_rates is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot BCE losses
    ax1.plot(epochs, train_losses, 'b-', label='Train BCE', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test BCE', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Bounded Cross-Entropy Loss')
    ax1.set_title(f'BCE Loss vs Epoch' + (f' (β={beta})' if beta is not None else ''))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot Zero-One losses
    ax2.plot(epochs, train_zero_one_losses, 'b-', label='Train 0-1', linewidth=2)
    ax2.plot(epochs, test_zero_one_losses, 'r-', label='Test 0-1', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Zero-One Loss (Error Rate)')
    ax2.set_title(f'Zero-One Loss vs Epoch' + (f' (β={beta})' if beta is not None else ''))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot learning rates if provided
    if learning_rates is not None:
        ax3.plot(epochs, learning_rates, 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title(f'Learning Rate vs Epoch' + (f' (β={beta})' if beta is not None else ''))
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved as '{save_path}'")
    
    plt.show()


def plot_bound_comparison(beta_values, results, loss_type='bce', save_path=None):
    """
    Plot a comparison between theoretical bounds and actual generalization errors.
    
    Args:
        beta_values: List of beta values
        results: Dictionary containing experimental results
        loss_type: 'bce' for bounded cross-entropy or 'zero_one' for zero-one loss
        save_path: Optional path to save the plot
    """
    from bounds import compute_generalization_bound, compute_generalization_errors
    
    # Compute bounds and generalization errors
    bounds = compute_generalization_bound(beta_values, results, loss_type=loss_type)
    gen_errors = compute_generalization_errors(beta_values, results)
    
    # Extract data
    if loss_type == 'bce':
        gen_error_key = 'bce_gen_error'
        gen_error_std_key = 'bce_gen_error_std'
        title_suffix = 'BCE'
    else:
        gen_error_key = 'zero_one_gen_error'
        gen_error_std_key = 'zero_one_gen_error_std'
        title_suffix = 'Zero-One'
    
    actual_gen_errors = [gen_errors[beta][gen_error_key] for beta in beta_values]
    gen_error_stds = [gen_errors[beta][gen_error_std_key] for beta in beta_values]
    theoretical_bounds = [bounds[beta]['generalization_bound'] for beta in beta_values]
    
    plt.figure(figsize=(10, 6))
    
    # Plot actual generalization errors with error bars
    plt.errorbar(beta_values, actual_gen_errors, yerr=gen_error_stds, 
                 fmt='o-', label=f'Actual Gen. Error (mean ± std)', 
                 linewidth=2, markersize=8, capsize=5, color='blue')
    
    # Plot theoretical bounds
    plt.plot(beta_values, theoretical_bounds, 's-', 
             label='Theoretical Upper Bound', 
             linewidth=2, markersize=8, color='red')
    
    plt.xlabel('Beta (Inverse Temperature)')
    plt.ylabel('Generalization Error / Bound')
    plt.title(f'{title_suffix} Loss: Theoretical Bound vs Actual Generalization Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use linear scale for x-axis
    plt.xscale('linear')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bound comparison plot saved as '{save_path}'")
    
    plt.show()


def plot_bound_tightness(beta_values, results, save_path=None):
    """
    Plot the bound tightness (difference between theoretical bound and actual error) vs beta.
    
    Args:
        beta_values: List of beta values
        results: Dictionary containing experimental results
        save_path: Optional path to save the plot
    """
    from bounds import compute_generalization_bound, compute_generalization_errors
    
    # Compute bounds and generalization errors for both loss types
    bce_bounds = compute_generalization_bound(beta_values, results, loss_type='bce')
    zero_one_bounds = compute_generalization_bound(beta_values, results, loss_type='zero_one')
    gen_errors = compute_generalization_errors(beta_values, results)
    
    # Compute bound gaps (positive means bound is valid)
    bce_gaps = [bce_bounds[beta]['generalization_bound'] - gen_errors[beta]['bce_gen_error'] 
                for beta in beta_values]
    zero_one_gaps = [zero_one_bounds[beta]['generalization_bound'] - gen_errors[beta]['zero_one_gen_error'] 
                     for beta in beta_values]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(beta_values, bce_gaps, 'o-', label='BCE Bound Gap', 
             linewidth=2, markersize=6, color='blue')
    plt.plot(beta_values, zero_one_gaps, 's-', label='Zero-One Bound Gap', 
             linewidth=2, markersize=6, color='red')
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Tight Bound (Gap=0)')
    
    plt.xlabel('Beta (Inverse Temperature)')
    plt.ylabel('Bound Gap (Theoretical Bound - Actual Error)')
    plt.title('Bound Tightness vs Beta')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use linear scale for x-axis
    plt.xscale('linear')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bound tightness plot saved as '{save_path}'")
    
    plt.show()
