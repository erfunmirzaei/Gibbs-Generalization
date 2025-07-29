"""
Generalization bound computation and analysis for the Gibbs generalization bound experiments.

This module contains functions for computing PAC-Bayesian generalization bounds,
actual generalization errors, and saving results from the SGLD experiments.
"""

import numpy as np
import torch
import math


def ln(x):
    """Natural logarithm function."""
    return math.log(x)


def kl(p, q, eps=1e-10):
    """
    Compute KL divergence between two Bernoulli distributions.
    
    Args:
        p: First probability
        q: Second probability
        eps: Small epsilon to prevent numerical issues with zero values
    
    Returns:
        KL divergence KL(p||q)
    """
    # Add epsilon to prevent log(0) and division by 0
    p = max(min(p, 1.0 - eps), eps)  # Clamp p to [eps, 1-eps]
    q = max(min(q, 1.0 - eps), eps)  # Clamp q to [eps, 1-eps]
    
    y = p * ln(p / q) + (1 - p) * ln((1 - p) / (1 - q))
    return y


def invert_kl(p, kl_val, eps=1e-10):
    """
    Invert KL divergence to find q such that KL(p||q) = kl_val.
    
    Args:
        p: First probability (fixed)
        kl_val: Target KL divergence value
        eps: Small epsilon to prevent numerical issues with zero values
    
    Returns:
        q such that KL(p||q) = kl_val
    """
    # Clamp p to prevent numerical issues
    p = max(min(p, 1.0 - eps), eps)
    
    l, u, r = p, 1 - eps, 0.5
    while ((u - l) > 1 / 100000):
        if kl(p, r, eps) < kl_val:
            l = r
            r = (r + u) / 2
        else:
            u = r
            r = (r + l) / 2
    return r

def _compute_single_bound(emp_loss, current_beta, results, integration_betas, train_key, n, delta=0.05, delta_prime=0.05, M=1):
    """
    Compute a single generalization bound for a given empirical loss.
    
    Args:
        emp_loss: Empirical loss value
        current_beta: Current beta value
        results: Dictionary containing experimental results
        integration_betas: List of beta values for integration
        train_key: Key for training loss in results
        n: Training set size
        delta: Confidence parameter for main bound
        delta_prime: Confidence parameter for integral bound
        M: Number of repetitions (for confidence term)
        
    Returns:
        Dictionary containing bound components
    """
    # Compute integral bound using all available betas from 0 to current_beta
    integral_bound = 0.0
    variance_term = 0.0
    
    # Filter integration betas to only include those up to current_beta
    relevant_betas = [b for b in integration_betas if b <= current_beta]
    
    # Always start from 0 and integrate up to current_beta
    prev_beta = 0.0
    
    for beta_k in relevant_betas:
        # Calculate beta difference from previous point
        beta_diff = beta_k - prev_beta
        
        # Average loss for the previous beta (starting point of interval)
        # Always use the mean value for integration, regardless of train_key
        if train_key.startswith('raw_'):
            # If train_key is raw_*, get the corresponding mean key
            mean_key = train_key.replace('raw_', '').replace('bce', 'bce_mean').replace('01', '01_mean')
            if 'bce' in train_key:
                mean_key = 'train_bce_mean'
            else:  # '01' in train_key
                mean_key = 'train_01_mean'
        else:
            mean_key = train_key
            
        avg_loss_k = results[prev_beta][mean_key]
        
        # Add to integral approximation
        integral_bound += beta_diff * avg_loss_k
        variance_term += beta_diff ** 2
        
        prev_beta = beta_k
    
    # Add the variance/confidence term for the integral
    integral_confidence = np.sqrt((variance_term + np.log(1 / delta_prime)) / M)
    integral_upper_bound = integral_bound #TODO: Change to use: + integral_confidence

    # Compute the main generalization bound
    # Inner term: integral - β * L̂(h,x) + ln(2√n/δ)
    inner_term = integral_upper_bound - current_beta * emp_loss + np.log(2 * np.sqrt(n) / delta)
    
    # Linear term
    linear_term = (2 * inner_term) / n

    # Ensure inner_term is positive for square root
    inner_term = max(inner_term, 1e-10)
    
    # Square root term
    sqrt_term = np.sqrt((2 * emp_loss * inner_term) / n)
    
    # Total bound: L(h) - L̂(h,x) ≤ sqrt_term + linear_term
    generalization_bound = linear_term + sqrt_term 
    
    return {
        'integral_bound': integral_bound,
        'integral_upper_bound': integral_upper_bound,
        'generalization_bound': generalization_bound,
        'sqrt_term': sqrt_term,
        'linear_term': linear_term,
        'inner_term': inner_term
    }


def compute_generalization_bound(beta_values, results, n, loss_type='bce', delta=0.05, delta_prime=0.05):
    """
    Compute the generalization bound for each beta value cumulatively.
    
    This function automatically handles the requirement for beta=0 in bounds computation.
    If beta=0 is not present in the results, it will raise an error since it's mathematically
    required for proper integral computation.
    
    Args:
        beta_values: List of beta values (sorted) - these are the values to compute bounds for
        results: Dictionary containing experimental results (must include beta=0)
        n: Training set size (required)
        loss_type: 'bce' for bounded cross-entropy or 'zero_one' for zero-one loss
        delta: Confidence parameter for main bound (default: 0.05)
        delta_prime: Confidence parameter for integral bound (default: 0.05)
        
    Returns:
        Dictionary containing bounds for each beta value
    """
    bounds = {}
    
    # Validate that n is provided
    if n is None:
        raise ValueError("Training set size 'n' must be provided as an explicit argument")
    
    # Check that beta=0 is available in results for proper integral computation
    if 0.0 not in results and 0 not in results:
        raise ValueError("beta=0 is required in results for proper generalization bound computation. "
                        "This should be handled automatically by run_beta_experiments().")
    
    # Create extended beta list that includes all values from 0 to max(beta_values)
    # for proper integral computation
    all_available_betas = sorted([float(b) for b in results.keys()])
    max_requested_beta = max(beta_values)
    
    # Filter to only include betas from 0 up to the maximum requested beta
    integration_betas = [b for b in all_available_betas if 0.0 <= b <= max_requested_beta]
    
    # Choose the appropriate loss type
    if loss_type == 'bce':
        train_key = 'train_bce_mean'
        raw_key = 'raw_train_bce'
    else:  # zero_one
        train_key = 'train_01_mean'
        raw_key = 'raw_train_01'
    
    for current_beta in beta_values:
        # Get current beta results
        average_emp_loss = results[current_beta][train_key]  # L̂(h, x)
        M = len(results[current_beta][raw_key])  # Number of repetitions
        
        # Compute bound using the unified function
        bound_components = _compute_single_bound(
            emp_loss=average_emp_loss,
            current_beta=current_beta,
            results=results,
            integration_betas=integration_betas,
            train_key=train_key,
            n=n,
            delta=delta,
            delta_prime=delta_prime,
            M=M
        )
        
        # Check for NaN in final bound
        if np.isnan(bound_components['generalization_bound']):
            print(f"WARNING: NaN generalization_bound for beta={current_beta}, loss_type={loss_type}")
            bound_components['generalization_bound'] = 1.0  # Use fallback value
        
        bounds[current_beta] = {
            'average_emp_loss': average_emp_loss,
            'predicted_test_loss': average_emp_loss + bound_components['generalization_bound'],  # L̂(h,x) + bound
            **bound_components  # Include all bound components
        }
    
    return bounds


def compute_generalization_errors(beta_values, results):
    """
    Compute the actual generalization errors for both BCE and zero-one loss.
    
    This function computes errors only for the specified beta_values, not requiring
    beta=0 to be included in the beta_values list.
    
    Args:
        beta_values: List of beta values (sorted) - these are the values to compute errors for
        results: Dictionary containing experimental results
        
    Returns:
        Dictionary containing generalization errors for each beta value
    """
    generalization_errors = {}
    
    for beta in beta_values:
        # BCE generalization error: test_loss - train_loss
        bce_gen_error = results[beta]['test_bce_mean'] - results[beta]['train_bce_mean']
        bce_gen_error_std = np.sqrt(results[beta]['test_bce_std']**2 + results[beta]['train_bce_std']**2)
        
        # Zero-one generalization error: test_loss - train_loss
        zero_one_gen_error = results[beta]['test_01_mean'] - results[beta]['train_01_mean']
        zero_one_gen_error_std = np.sqrt(results[beta]['test_01_std']**2 + results[beta]['train_01_std']**2)
        
        generalization_errors[beta] = {
            'bce_gen_error': bce_gen_error,
            'bce_gen_error_std': bce_gen_error_std,
            'zero_one_gen_error': zero_one_gen_error,
            'zero_one_gen_error_std': zero_one_gen_error_std
        }
    
    return generalization_errors


def save_results_to_file(results, n, filename=None, beta_values=None, num_repetitions=None, 
                        num_epochs=None, a0=None, sigma_gauss_prior=None, dataset_type='synth'):
    """
    Save the experimental results to a text file with descriptive filename.
    
    Args:
        results: Dictionary containing experimental results
        n: Training set size (required)
        filename: Optional custom filename. If None, generates descriptive filename
        beta_values: List of beta values tested (for filename generation)
        num_repetitions: Number of repetitions per beta (for filename generation)
        num_epochs: Number of training epochs (for filename generation)
        a0: Learning rate (for filename generation)
        sigma_gauss_prior: Prior parameter (for filename generation)
        dataset_type: Dataset type (for filename generation)
    """
    # Validate that n is provided
    if n is None:
        raise ValueError("Training set size 'n' must be provided as an explicit argument")
    
    # Generate descriptive filename if not provided
    if filename is None and all(param is not None for param in [beta_values, num_repetitions, num_epochs, a0, sigma_gauss_prior]):
        filename = generate_filename(
            beta_values=beta_values,
            num_repetitions=num_repetitions,
            num_epochs=num_epochs,
            a0=a0,
            sigma_gauss_prior=sigma_gauss_prior,
            dataset_type=dataset_type,
            file_type='results',
            extension='txt'
        )
        filename = f"results/{filename}"
    elif filename is None:
        # Fallback to default naming
        filename = 'results/sgld_beta_experiments.txt'
    
    print(f"Using training set size n = {n} for bound computations")
    
    # Compute generalization bounds and errors
    # Use only the specified beta_values for bounds computation - the bounds function
    # will handle the need for beta=0 internally for proper integral computation
    bounds = compute_generalization_bound(beta_values or sorted(results.keys()), results, n, loss_type='bce')
    zero_one_bounds = compute_generalization_bound(beta_values or sorted(results.keys()), results, n, loss_type='zero_one')
    individual_bounds = compute_individual_generalization_bounds(beta_values or sorted(results.keys()), results, n, loss_type='bce')
    individual_zero_one_bounds = compute_individual_generalization_bounds(beta_values or sorted(results.keys()), results, n, loss_type='zero_one')
    kl_analysis = compute_kl_divergence_analysis(beta_values or sorted(results.keys()), results, n)
    gen_errors = compute_generalization_errors(beta_values or sorted(results.keys()), results)
    
    # Use specified beta_values for file content (excluding beta=0 if not originally requested)
    if beta_values is None:
        display_beta_values = sorted([b for b in results.keys() if b != 0.0 and b != 0])
    else:
        display_beta_values = sorted(beta_values)
    
    with open(filename, 'w') as f:
        f.write("SGLD Beta Experiments Results\n")
        f.write("="*80 + "\n\n")
        
        # Write experimental parameters
        f.write("Experimental Parameters:\n")
        f.write("-"*40 + "\n")
        if beta_values is not None:
            f.write(f"Requested beta values: {beta_values}\n")
        
        # Get all beta values for informational purposes
        all_beta_values = sorted(results.keys())
        f.write(f"All beta values in results: {all_beta_values}\n")
        if beta_values is not None and 0.0 in all_beta_values and 0.0 not in beta_values:
            f.write("Note: Beta=0 was automatically added for proper generalization bound computation\n")
        if num_repetitions is not None:
            f.write(f"Repetitions per beta: {num_repetitions}\n")
        if num_epochs is not None:
            f.write(f"Training epochs: {num_epochs}\n")
        if a0 is not None:
            f.write(f"Learning rate (a0): {a0}\n")
        if sigma_gauss_prior is not None:
            f.write(f"Gaussian prior sigma: {sigma_gauss_prior}\n")
        f.write(f"Dataset type: {dataset_type}\n")
        if 'random_labels' in dataset_type:
            f.write("Note: Using random labels dataset (no linear relationship between inputs and outputs)\n")
        f.write(f"Generated filename: {filename.split('/')[-1]}\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # Results summary table (same format as console output)
        f.write("RESULTS SUMMARY TABLE\n")
        f.write("="*80 + "\n")
        f.write(f"{'Beta':<8} {'Train Error':<12} {'Test Error':<12} {'Min Train Error':<15}\n")
        f.write(f"{'(β)':<8} {'(Mean)':<12} {'(Mean)':<12} {'(Min per β)':<15}\n")
        f.write(f"{'-'*8} {'-'*12} {'-'*12} {'-'*15}\n")
        
        # Print results for each beta, including beta=0 if present, then display_beta_values
        # Create a combined list that includes beta=0 if it exists, followed by display_beta_values
        summary_betas = []
        if 0.0 in results:
            summary_betas.append(0.0)
        elif 0 in results:
            summary_betas.append(0)
        
        # Add the originally requested beta values
        for beta in sorted(display_beta_values):
            if beta != 0.0 and beta != 0:  # Avoid duplicating beta=0
                summary_betas.append(beta)
        
        for beta in summary_betas:
            train_error = results[beta]['train_bce_mean']
            test_error = results[beta]['test_bce_mean']
            
            # For this beta, find the minimum train error among all repetitions
            min_train_error_for_beta = min(results[beta]['raw_train_bce'])
            
            f.write(f"{beta:<8.1f} {train_error:<12.4f} {test_error:<12.4f} {min_train_error_for_beta:<15.4f}\n")
        
        f.write(f"{'-'*8} {'-'*12} {'-'*12} {'-'*15}\n")
        f.write("Notes:\n")
        f.write("  - Train/Test Error: Bounded Cross-Entropy (BCE) Loss\n")
        f.write("  - Min Train Error: Lowest train error among all repetitions for each β\n")
        f.write("  - β=0: Pure Gradient Descent (no SGLD noise)\n")
        f.write("  - Higher β: More SGLD noise, potentially better generalization\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed summary table
        f.write("DETAILED SUMMARY TABLE:\n")
        f.write("Beta\tTrain_BCE\tTest_BCE\tBCE_GenErr\tBCE_Bound\tIndiv_BCE_Mean\tIndiv_BCE_Std\tKL_Mean\tKL_Std\tKL_Bound_Mean\tTest_Bound_Mean\tTrain_01\tTest_01\tZO_GenErr\tZO_Bound\tIndiv_ZO_Mean\tIndiv_ZO_Std\n")
        for beta in display_beta_values:
            f.write(f"{beta}\t{results[beta]['train_bce_mean']:.4f}\t\t{results[beta]['test_bce_mean']:.4f}\t\t")
            f.write(f"{gen_errors[beta]['bce_gen_error']:.4f}\t\t{bounds[beta]['generalization_bound']:.4f}\t\t")
            f.write(f"{individual_bounds[beta]['bound_mean']:.4f}\t\t{individual_bounds[beta]['bound_std']:.4f}\t\t")
            f.write(f"{kl_analysis[beta]['kl_mean']:.4f}\t\t{kl_analysis[beta]['kl_std']:.4f}\t\t")
            f.write(f"{kl_analysis[beta]['kl_bound_mean']:.4f}\t\t{kl_analysis[beta]['test_bound_mean']:.4f}\t\t")
            f.write(f"{results[beta]['train_01_mean']:.4f}\t\t{results[beta]['test_01_mean']:.4f}\t\t")
            f.write(f"{gen_errors[beta]['zero_one_gen_error']:.4f}\t\t{zero_one_bounds[beta]['generalization_bound']:.4f}\t\t")
            f.write(f"{individual_zero_one_bounds[beta]['bound_mean']:.4f}\t\t{individual_zero_one_bounds[beta]['bound_std']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed results
        for beta in display_beta_values:
            f.write(f"Beta = {beta}:\n")
            f.write(f"  Bounded Cross-Entropy Loss:\n")
            f.write(f"    Train: mean={results[beta]['train_bce_mean']:.4f}, var={results[beta]['train_bce_var']:.6f}, std={results[beta]['train_bce_std']:.4f}\n")
            f.write(f"    Test:  mean={results[beta]['test_bce_mean']:.4f}, var={results[beta]['test_bce_var']:.6f}, std={results[beta]['test_bce_std']:.4f}\n")
            f.write(f"    Generalization Error: {gen_errors[beta]['bce_gen_error']:.4f} ± {gen_errors[beta]['bce_gen_error_std']:.4f}\n")
            f.write(f"  Theoretical Generalization Bound:\n")
            f.write(f"    Upper Bound: {bounds[beta]['generalization_bound']:.4f}\n")
            f.write(f"    Bound Tightness: {bounds[beta]['generalization_bound'] - gen_errors[beta]['bce_gen_error']:.4f}\n")
            f.write(f"    Sqrt Term: {bounds[beta]['sqrt_term']:.4f}, Linear Term: {bounds[beta]['linear_term']:.4f}\n")
            f.write(f"  Individual Bounds (per repetition):\n")
            f.write(f"    Bound Mean: {individual_bounds[beta]['bound_mean']:.4f}\n")
            f.write(f"    Bound Std: {individual_bounds[beta]['bound_std']:.4f}\n")
            f.write(f"    Bound Range: [{individual_bounds[beta]['bound_min']:.4f}, {individual_bounds[beta]['bound_max']:.4f}]\n")
            f.write(f"    Individual Bound Tightness: {individual_bounds[beta]['bound_mean'] - gen_errors[beta]['bce_gen_error']:.4f} ± {individual_bounds[beta]['bound_std']:.4f}\n")
            f.write(f"  Zero-One Loss:\n")
            f.write(f"    Train: mean={results[beta]['train_01_mean']:.4f}, var={results[beta]['train_01_var']:.6f}, std={results[beta]['train_01_std']:.4f}\n")
            f.write(f"    Test:  mean={results[beta]['test_01_mean']:.4f}, var={results[beta]['test_01_var']:.6f}, std={results[beta]['test_01_std']:.4f}\n")
            f.write(f"    Generalization Error: {gen_errors[beta]['zero_one_gen_error']:.4f} ± {gen_errors[beta]['zero_one_gen_error_std']:.4f}\n")
            f.write(f"  Zero-One Theoretical Generalization Bound:\n")
            f.write(f"    Upper Bound: {zero_one_bounds[beta]['generalization_bound']:.4f}\n")
            f.write(f"    Bound Tightness: {zero_one_bounds[beta]['generalization_bound'] - gen_errors[beta]['zero_one_gen_error']:.4f}\n")
            f.write(f"    Sqrt Term: {zero_one_bounds[beta]['sqrt_term']:.4f}, Linear Term: {zero_one_bounds[beta]['linear_term']:.4f}\n")
            f.write(f"  Individual Zero-One Bounds (per repetition):\n")
            f.write(f"    Bound Mean: {individual_zero_one_bounds[beta]['bound_mean']:.4f}\n")
            f.write(f"    Bound Std: {individual_zero_one_bounds[beta]['bound_std']:.4f}\n")
            f.write(f"    Bound Range: [{individual_zero_one_bounds[beta]['bound_min']:.4f}, {individual_zero_one_bounds[beta]['bound_max']:.4f}]\n")
            f.write(f"    Individual Bound Tightness: {individual_zero_one_bounds[beta]['bound_mean'] - gen_errors[beta]['zero_one_gen_error']:.4f} ± {individual_zero_one_bounds[beta]['bound_std']:.4f}\n")
            f.write(f"  KL Divergence Analysis:\n")
            f.write(f"    KL(train||test) Mean: {kl_analysis[beta]['kl_mean']:.4f} ± {kl_analysis[beta]['kl_std']:.4f}\n")
            f.write(f"    KL Bound Mean: {kl_analysis[beta]['kl_bound_mean']:.4f} ± {kl_analysis[beta]['kl_bound_std']:.4f}\n")
            f.write(f"    Test Error Bound (via invert_kl): {kl_analysis[beta]['test_bound_mean']:.4f} ± {kl_analysis[beta]['test_bound_std']:.4f}\n")
            f.write("\n")
    
        # Individual repetition data
        f.write("="*80 + "\n")
        f.write("INDIVIDUAL REPETITION DATA (Final Train/Test Errors)\n")
        f.write("="*80 + "\n\n")
        
        for beta in display_beta_values:
            f.write(f"Beta = {beta}:\n")
            f.write("-" * 40 + "\n")
            
            # Get raw data for this beta
            raw_train_bce = results[beta]['raw_train_bce']
            raw_test_bce = results[beta]['raw_test_bce']
            raw_train_01 = results[beta]['raw_train_01']
            raw_test_01 = results[beta]['raw_test_01']
            
            num_reps = len(raw_train_bce)
            f.write(f"Number of repetitions: {num_reps}\n\n")
            
            # Table header
            f.write(f"{'Rep':<4} {'Train BCE':<12} {'Test BCE':<12} {'Train 0-1':<12} {'Test 0-1':<12} {'BCE Gen.Err':<12} {'0-1 Gen.Err':<12}\n")
            f.write(f"{'-'*4} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}\n")
            
            # Individual repetition data
            for i in range(num_reps):
                train_bce = raw_train_bce[i]
                test_bce = raw_test_bce[i]
                train_01 = raw_train_01[i]
                test_01 = raw_test_01[i]
                bce_gen_err = test_bce - train_bce
                zo_gen_err = test_01 - train_01
                
                f.write(f"{i+1:<4} {train_bce:<12.6f} {test_bce:<12.6f} {train_01:<12.6f} {test_01:<12.6f} {bce_gen_err:<12.6f} {zo_gen_err:<12.6f}\n")
            
            # Summary statistics for this beta
            f.write(f"{'-'*4} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}\n")
            f.write(f"{'Mean':<4} {np.mean(raw_train_bce):<12.6f} {np.mean(raw_test_bce):<12.6f} {np.mean(raw_train_01):<12.6f} {np.mean(raw_test_01):<12.6f} {np.mean([raw_test_bce[i] - raw_train_bce[i] for i in range(num_reps)]):<12.6f} {np.mean([raw_test_01[i] - raw_train_01[i] for i in range(num_reps)]):<12.6f}\n")
            f.write(f"{'Std':<4} {np.std(raw_train_bce, ddof=1):<12.6f} {np.std(raw_test_bce, ddof=1):<12.6f} {np.std(raw_train_01, ddof=1):<12.6f} {np.std(raw_test_01, ddof=1):<12.6f} {np.std([raw_test_bce[i] - raw_train_bce[i] for i in range(num_reps)], ddof=1):<12.6f} {np.std([raw_test_01[i] - raw_train_01[i] for i in range(num_reps)], ddof=1):<12.6f}\n")
            f.write(f"{'Min':<4} {np.min(raw_train_bce):<12.6f} {np.min(raw_test_bce):<12.6f} {np.min(raw_train_01):<12.6f} {np.min(raw_test_01):<12.6f} {np.min([raw_test_bce[i] - raw_train_bce[i] for i in range(num_reps)]):<12.6f} {np.min([raw_test_01[i] - raw_train_01[i] for i in range(num_reps)]):<12.6f}\n")
            f.write(f"{'Max':<4} {np.max(raw_train_bce):<12.6f} {np.max(raw_test_bce):<12.6f} {np.max(raw_train_01):<12.6f} {np.max(raw_test_01):<12.6f} {np.max([raw_test_bce[i] - raw_train_bce[i] for i in range(num_reps)]):<12.6f} {np.max([raw_test_01[i] - raw_train_01[i] for i in range(num_reps)]):<12.6f}\n")
            f.write("\n")
        
        f.write("Notes:\n")
        f.write("- Rep: Repetition number\n")
        f.write("- BCE: Bounded Cross-Entropy Loss\n")
        f.write("- 0-1: Zero-One Loss (classification error rate)\n")  
        f.write("- Gen.Err: Generalization Error (Test - Train)\n")
        f.write("- All values are final errors after training completion\n")
        f.write("="*80 + "\n\n")
    
def generate_filename(beta_values, num_repetitions, num_epochs, a0, sigma_gauss_prior, 
                     dataset_type='synth', file_type='results', extension='txt'):
    """
    Generate a descriptive filename based on experimental parameters.
    
    Args:
        beta_values: List of beta values tested
        num_repetitions: Number of repetitions per beta
        num_epochs: Number of training epochs
        a0: Learning rate
        sigma_gauss_prior: Prior parameter
        dataset_type: Type of dataset ('synth', 'mnist', etc.)
        file_type: Type of file ('results', 'plot', etc.)
        extension: File extension ('txt', 'png', etc.)
    
    Returns:
        str: Descriptive filename
    """
    # Format beta range
    beta_min, beta_max = min(beta_values), max(beta_values)
    beta_str = f"beta{beta_min}-{beta_max}" if beta_min != beta_max else f"beta{beta_min}"
    
    # Format learning rate (a0)
    if isinstance(a0, dict):
        # For dict, show the range or summary
        a0_values = list(a0.values())
        if len(set(a0_values)) == 1:
            # All same a0
            a0_val = a0_values[0]
            if a0_val >= 1:
                lr_str = f"lr{a0_val:.0f}"
            elif a0_val >= 0.01:
                lr_str = f"lr{a0_val:.2f}".replace('.', 'p')
            else:
                lr_str = f"lr{a0_val:.1e}".replace('.', 'p').replace('-', 'n')
        else:
            # Variable a0
            min_a0 = min(a0_values)
            max_a0 = max(a0_values)
            lr_str = f"lr{min_a0:.3f}-{max_a0:.3f}".replace('.', 'p')
    elif callable(a0):
        # For callable, use a generic label
        lr_str = "lr_adaptive"
    elif isinstance(a0, (int, float)):
        if a0 >= 1:
            lr_str = f"lr{a0:.0f}"
        elif a0 >= 0.01:
            lr_str = f"lr{a0:.2f}".replace('.', 'p')
        else:
            lr_str = f"lr{a0:.1e}".replace('.', 'p').replace('-', 'n')
    else:
        lr_str = "lr_unknown"
    
    # Format sigma prior
    if sigma_gauss_prior >= 1:
        sigma_str = f"sigma{sigma_gauss_prior:.0f}"
    else:
        sigma_str = f"sigma{sigma_gauss_prior:.2f}".replace('.', 'p')
    
    # Format epochs
    if isinstance(num_epochs, dict):
        # For dict, show the range or summary
        epoch_values = list(num_epochs.values())
        if len(set(epoch_values)) == 1:
            # All same epochs
            epochs_val = epoch_values[0]
            if epochs_val >= 1000:
                epochs_str = f"ep{epochs_val//1000}k"
            else:
                epochs_str = f"ep{epochs_val}"
        else:
            # Variable epochs
            min_epochs = min(epoch_values)
            max_epochs = max(epoch_values)
            epochs_str = f"ep{min_epochs}-{max_epochs}"
    elif callable(num_epochs):
        # For callable, use a generic label
        epochs_str = "ep_adaptive"
    elif isinstance(num_epochs, int):
        if num_epochs >= 1000:
            epochs_str = f"ep{num_epochs//1000}k"
        else:
            epochs_str = f"ep{num_epochs}"
    else:
        epochs_str = "ep_unknown"
    
    # Create filename
    filename = f"sgld_{file_type}_{dataset_type}_{beta_str}_{lr_str}_{sigma_str}_{epochs_str}_rep{num_repetitions}.{extension}"
    
    return filename


def compute_individual_generalization_bounds(beta_values, results, n, loss_type='bce', delta=0.05, delta_prime=0.05):
    """
    Compute the generalization bound for each individual empirical loss, then average and compute std.
    
    This function computes bounds for each individual repetition's empirical loss rather than
    using the average empirical loss. This gives us a distribution of bounds from which we
    can compute mean and standard deviation.
    
    Args:
        beta_values: List of beta values (sorted) - these are the values to compute bounds for
        results: Dictionary containing experimental results (must include beta=0)
        n: Training set size (required)
        loss_type: 'bce' for bounded cross-entropy or 'zero_one' for zero-one loss
        delta: Confidence parameter for main bound (default: 0.05)
        delta_prime: Confidence parameter for integral bound (default: 0.05)
        
    Returns:
        Dictionary containing individual bounds statistics for each beta value
    """
    individual_bounds = {}
    
    # Validate that n is provided
    if n is None:
        raise ValueError("Training set size 'n' must be provided as an explicit argument")
    
    # Check that beta=0 is available in results for proper integral computation
    if 0.0 not in results and 0 not in results:
        raise ValueError("beta=0 is required in results for proper generalization bound computation. "
                        "This should be handled automatically by run_beta_experiments().")
    
    # Create extended beta list that includes all values from 0 to max(beta_values)
    # for proper integral computation
    all_available_betas = sorted([float(b) for b in results.keys()])
    max_requested_beta = max(beta_values)
    
    # Filter to only include betas from 0 up to the maximum requested beta
    integration_betas = [b for b in all_available_betas if 0.0 <= b <= max_requested_beta]
    
    # Choose the appropriate loss type
    if loss_type == 'bce':
        train_key = 'train_bce_mean'
        raw_key = 'raw_train_bce'
    else:  # zero_one
        train_key = 'train_01_mean'
        raw_key = 'raw_train_01'
    
    for current_beta in beta_values:
        # Get individual empirical losses for current beta
        individual_emp_losses = results[current_beta][raw_key]  # List of individual losses
        M = len(individual_emp_losses)  # Number of repetitions
        
        # Compute bound for each individual empirical loss using the unified function
        individual_bound_values = []
        
        for emp_loss in individual_emp_losses:
            bound_result = _compute_single_bound(
                emp_loss=emp_loss,
                current_beta=current_beta,
                results=results,
                integration_betas=integration_betas,
                train_key=raw_key,
                n=n,
                delta=delta,
                delta_prime=delta_prime,
                M=M
            )
            
            # Check for NaN in final bound
            if np.isnan(bound_result['generalization_bound']):
                print(f"WARNING: NaN generalization_bound for beta={current_beta}, loss_type={loss_type}, emp_loss={emp_loss}")
                bound_result['generalization_bound'] = 1.0  # Use fallback value
            
            individual_bound_values.append(bound_result['generalization_bound'])
        
        # Compute statistics over all individual bounds
        individual_bound_values = np.array(individual_bound_values)
        
        # Get average empirical loss for comparison
        average_emp_loss = results[current_beta][train_key]
        
        individual_bounds[current_beta] = {
            'individual_bounds': individual_bound_values.tolist(),
            'bound_mean': np.mean(individual_bound_values),
            'bound_std': np.std(individual_bound_values),
            'bound_var': np.var(individual_bound_values),
            'bound_min': np.min(individual_bound_values),
            'bound_max': np.max(individual_bound_values),
            'average_emp_loss': average_emp_loss,
            'predicted_test_loss_mean': average_emp_loss + np.mean(individual_bound_values),
            'predicted_test_loss_std': np.std(individual_bound_values),  # Std comes only from bound uncertainty
            'num_repetitions': M
        }
    
    return individual_bounds


def compute_kl_divergence_analysis(beta_values, results, n, loss_type='bce'):
    """
    Compute KL divergence between train and test error for each beta and repetition.
    Also compute bounds on KL divergence and use invert_kl to bound test error.
    
    Args:
        beta_values: List of beta values (sorted) - these are the values to compute for
        results: Dictionary containing experimental results (must include beta=0)
        n: Training set size (required)
        loss_type: 'bce' for bounded cross-entropy or 'zero_one' for zero-one loss
        
    Returns:
        Dictionary containing KL divergence statistics for each beta value
    """
    kl_analysis = {}
    
    # Validate that n is provided
    if n is None:
        raise ValueError("Training set size 'n' must be provided as an explicit argument")
    
    # Check that beta=0 is available in results for proper integral computation
    if 0.0 not in results and 0 not in results:
        raise ValueError("beta=0 is required in results for proper KL divergence bound computation.")
    
    # Create extended beta list that includes all values from 0 to max(beta_values)
    all_available_betas = sorted([float(b) for b in results.keys()])
    max_requested_beta = max(beta_values)
    
    # Filter to only include betas from 0 up to the maximum requested beta
    integration_betas = [b for b in all_available_betas if 0.0 <= b <= max_requested_beta]
    
    # Choose the appropriate loss type keys
    if loss_type == 'bce':
        train_key = 'train_bce_mean'
        raw_key = 'raw_train_bce'
        test_raw_key = 'raw_test_bce'
    else:  # zero_one
        train_key = 'train_01_mean'
        raw_key = 'raw_train_01'
        test_raw_key = 'raw_test_01'
    
    for current_beta in beta_values:
        # Get individual train and test errors for the specified loss type
        individual_train_losses = results[current_beta][raw_key]
        individual_test_losses = results[current_beta][test_raw_key]
        M = len(individual_train_losses)  # Number of repetitions
        
        # Compute KL divergence for each repetition
        individual_kl_values = []
        individual_kl_bounds = []
        individual_test_bounds = []
        
        for i in range(M):
            train_error = individual_train_losses[i]
            test_error = individual_test_losses[i]

            
            # Compute KL divergence between train and test
            try:
                # Add epsilon handling to prevent numerical issues with zero values
                eps = 1e-10
                kl_div = kl(train_error, test_error, eps)
                # Check if the result is valid (not NaN or infinite)
                if np.isnan(kl_div) or np.isinf(kl_div):
                    print(f"WARNING: Invalid KL divergence for beta={current_beta}, train={train_error}, test={test_error}")
                    individual_kl_values.append(0.0)
                else:
                    individual_kl_values.append(kl_div)
            except Exception as e:
                # Handle edge cases with more detailed error information
                print(f"WARNING: KL computation failed for beta={current_beta}, train={train_error}, test={test_error}: {e}")
                individual_kl_values.append(0.0)
                continue
            
            # Use the same unified bound computation approach as individual bounds
            bound_result = _compute_single_bound(
                emp_loss=train_error,
                current_beta=current_beta,
                results=results,
                integration_betas=integration_betas,
                train_key=train_key,
                n=n,
                delta=0.05,
                delta_prime=0.05,
                M=M
            )
            
            # Extract the KL bound from the inner term divided by n
            kl_bound = bound_result['inner_term'] / n
            kl_bound = max(kl_bound, 0.0)  # Ensure non-negative
            individual_kl_bounds.append(kl_bound)
            
            # Use invert_kl to bound the test error
            try:
                if kl_bound > 0:
                    eps = 1e-10
                    test_bound = invert_kl(train_error, kl_bound, eps)
                    # Check if the result is valid
                    if np.isnan(test_bound) or np.isinf(test_bound):
                        print(f"WARNING: Invalid test bound from invert_kl for beta={current_beta}, train={train_error}, kl_bound={kl_bound}")
                        test_bound = max(train_error, 1.0)  # Use reasonable fallback
                else:
                    test_bound = train_error  # If no bound, use train error
                individual_test_bounds.append(test_bound)
            except Exception as e:
                print(f"WARNING: invert_kl failed for beta={current_beta}, train={train_error}, kl_bound={kl_bound}: {e}")
                individual_test_bounds.append(max(train_error, 1.0))  # Fallback bound
        
        # Compute statistics over all repetitions
        individual_kl_values = np.array(individual_kl_values)
        individual_kl_bounds = np.array(individual_kl_bounds)
        individual_test_bounds = np.array(individual_test_bounds)
        
        kl_analysis[current_beta] = {
            'individual_kl_values': individual_kl_values.tolist(),
            'kl_mean': np.mean(individual_kl_values),
            'kl_std': np.std(individual_kl_values),
            'kl_min': np.min(individual_kl_values),
            'kl_max': np.max(individual_kl_values),
            'individual_kl_bounds': individual_kl_bounds.tolist(),
            'kl_bound_mean': np.mean(individual_kl_bounds),
            'kl_bound_std': np.std(individual_kl_bounds),
            'individual_test_bounds': individual_test_bounds.tolist(),
            'test_bound_mean': np.mean(individual_test_bounds),
            'test_bound_std': np.std(individual_test_bounds),
            'num_repetitions': M
        }
    
    return kl_analysis
