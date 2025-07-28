"""
Training and experiment orchestration for the Gibbs generalization bound experiments.

This module contains functions for training models with SGLD and running
experiments across different beta values with multiple repetitions.
"""

import numpy as np
import torch
import torch.optim as optim
import time
from losses import BoundedCrossEntropyLoss, ZeroOneLoss, TangentLoss
from models import SynthNN, MNISTNN, initialize_kaiming_and_get_prior_sigma
from sgld import SGLD
from dataset import (get_synth_dataloaders, get_synth_dataloaders_random_labels,
                    get_mnist_binary_dataloaders)


def transform_bce_to_unit_interval(bce_loss, l_max=2.0):
    """
    Transform BCE loss to [0,1] interval using the formula:
    (BCE + ln(1-exp(-l_max))) / (l_max + ln(1-exp(-l_max)))
    
    Args:
        bce_loss: BCE loss value (tensor or scalar)
        l_max: Maximum loss value parameter
    
    Returns:
        Transformed loss in [0,1] interval
    """
    ln_term = torch.log(1 - torch.exp(torch.tensor(-l_max)))
    numerator = bce_loss + ln_term
    denominator = l_max + ln_term
    return numerator / denominator


def train_sgld_model(model, train_loader, test_loader, num_epochs: int = 100, 
                     a0: float = 1e-3, b: float = 0.5, sigma_gauss_prior: float = 0.1, 
                     beta: float = 1.0, device: str = 'cpu', dataset_type: str = 'synth',
                     l_max: float = 2.0):
    """
    Train the neural network with SGLD and bounded cross entropy loss.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        a0: Initial learning rate (1e-5 for MNIST, 1e-3 for SYNTH)
        b: Decay rate (default: 0.5)
        sigma_gauss_prior: Gaussian prior sigma for weight decay
        beta: Inverse temperature parameter for SGLD noise scaling
        device: Device to run training on
        dataset_type: 'synth' or 'mnist' for proper loss computation
        l_max: Maximum loss value for transformation to [0,1] interval
        
    Returns:
        Tuple containing: (train_losses, test_losses, train_accuracies, test_accuracies, 
                          train_zero_one_losses, test_zero_one_losses, learning_rates)
    """
    model = model.to(device)
    criterion = BoundedCrossEntropyLoss(ell_max=l_max)
    # criterion = TangentLoss()
    zero_one_criterion = ZeroOneLoss()
    

    # Check if we're using BCE for optimization (to determine if transformation is needed)
    using_bce_for_optimization = isinstance(criterion, BoundedCrossEntropyLoss)
    
    # Initialize SGLD optimizer with inverse temperature
    optimizer = SGLD(model.parameters(), lr=a0, sigma_gauss_prior=sigma_gauss_prior, 
                     beta=beta, add_noise=True)
    
    # Learning rate scheduler: lr_t = a0 * t^(-b)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) ** (-b))
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    train_zero_one_losses = []
    test_zero_one_losses = []
    learning_rates = []
    
    print(f"Training with SGLD: a0={a0}, b={b}, sigma_gauss_prior={sigma_gauss_prior}, beta={beta}")
    print(f"Dataset type: {dataset_type}, Device: {device}")
    
    # Progress tracking
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_zero_one_total = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            # Use non_blocking=True for faster GPU transfer
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # Use more efficient zero_grad
            optimizer.zero_grad(set_to_none=True)
            
            # Standard precision training
            outputs = model(batch_x)
            
            # Handle different output shapes for SYNTH vs MNIST
            if dataset_type == 'synth':
                loss_for_optimization = criterion(outputs, batch_y)
                if using_bce_for_optimization:
                    loss_for_recording = transform_bce_to_unit_interval(loss_for_optimization, l_max)
                else:
                    loss_for_recording = loss_for_optimization
                predicted = (outputs.squeeze() > 0).float()
                zero_one_loss = zero_one_criterion(outputs, batch_y)
            else:
                loss_for_optimization = criterion(outputs, batch_y)
                if using_bce_for_optimization:
                    loss_for_recording = transform_bce_to_unit_interval(loss_for_optimization, l_max)
                else:
                    loss_for_recording = loss_for_optimization
                predicted = (outputs.squeeze() > 0).float()
                zero_one_loss = zero_one_criterion(outputs, batch_y)
            
            loss_for_optimization.backward()
            optimizer.step()
            
            # Record the loss (transformed if BCE was used for optimization, raw otherwise)
            train_loss_total += loss_for_recording.item()
            train_zero_one_total += zero_one_loss.item()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Step the learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        scheduler.step()
        
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_zero_one = train_zero_one_total / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_zero_one_losses.append(avg_train_zero_one)
        train_accuracies.append(train_accuracy)
        
        # Test/Evaluation phase
        model.eval()
        test_loss_total = 0.0
        test_zero_one_total = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                outputs = model(batch_x)
                
                if dataset_type == 'synth':
                    loss_for_optimization = criterion(outputs, batch_y)
                    if using_bce_for_optimization:
                        loss_for_recording = transform_bce_to_unit_interval(loss_for_optimization, l_max)
                    else:
                        loss_for_recording = loss_for_optimization
                    predicted = (outputs.squeeze() > 0).float()
                    zero_one_loss = zero_one_criterion(outputs, batch_y)
                else:
                    loss_for_optimization = criterion(outputs, batch_y)
                    if using_bce_for_optimization:
                        loss_for_recording = transform_bce_to_unit_interval(loss_for_optimization, l_max)
                    else:
                        loss_for_recording = loss_for_optimization
                    predicted = (outputs.squeeze() > 0).float()
                    zero_one_loss = zero_one_criterion(outputs, batch_y)
                
                test_loss_total += loss_for_recording.item()
                test_zero_one_total += zero_one_loss.item()
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        avg_test_loss = test_loss_total / len(test_loader)
        avg_test_zero_one = test_zero_one_total / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        test_losses.append(avg_test_loss)
        test_zero_one_losses.append(avg_test_zero_one)
        test_accuracies.append(test_accuracy)
        
        # More frequent progress reporting with time estimates
        if epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs - 1:
            elapsed_time = time.time() - start_time
            epochs_per_second = (epoch + 1) / elapsed_time if elapsed_time > 0 else 0
            eta_seconds = (num_epochs - epoch - 1) / epochs_per_second if epochs_per_second > 0 else 0
            eta_minutes = eta_seconds / 60
            
            print(f'Epoch [{epoch+1:>6}/{num_epochs}] '
                  f'Train: {avg_train_loss:.4f} Test: {avg_test_loss:.4f} '
                  f'Train0-1: {avg_train_zero_one:.4f} Test0-1: {avg_test_zero_one:.4f} '
                  f'LR: {current_lr:.2e} '
                  f'Speed: {epochs_per_second:.1f} ep/s '
                  f'ETA: {eta_minutes:.1f}min')
            
            # GPU memory monitoring (if using CUDA)
            if device == 'cuda':
                print(f'GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.0f}MB allocated, '
                      f'{torch.cuda.memory_reserved(0) / 1024**2:.0f}MB reserved')
    
    return train_losses, test_losses, train_accuracies, test_accuracies, train_zero_one_losses, test_zero_one_losses, learning_rates


def run_beta_experiments(beta_values, num_repetitions=50, num_epochs=10000, 
                         a0=1e-1, b=0.5, sigma_gauss_prior=1000000, 
                         device='cpu', dataset_type='synth', use_random_labels=False,
                         l_max=2.0, mnist_classes=None):
    """
    Run experiments across different beta values with multiple repetitions.
    
    Automatically includes beta=0 if not present, as it's required for proper
    generalization bound computation (beta=0 corresponds to pure noise).
    
    Args:
        beta_values: List of beta (inverse temperature) values to test
        num_repetitions: Number of repetitions for each beta value
        num_epochs: Number of training epochs per run. Can be:
                   - int: Same number of epochs for all beta values
                   - dict: {beta: epochs} mapping for different epochs per beta
                   - callable: function(beta) -> epochs
        a0: Initial learning rate. Can be:
           - float: Same learning rate for all beta values
           - dict: {beta: a0} mapping for different learning rates per beta
           - callable: function(beta) -> a0
        b: Learning rate decay parameter
        sigma_gauss_prior: Gaussian prior sigma
        device: Device to run on
        dataset_type: 'synth' or 'mnist'
        use_random_labels: If True, use random labels instead of linear relationship
        l_max: Maximum loss value for transformation to [0,1] interval
        
    Returns:
        Dictionary containing results for each beta value (including beta=0 if not present)
    """
    # Helper function to determine epochs for each beta
    def get_epochs_for_beta(beta, num_epochs):
        """Determine number of epochs based on beta value and num_epochs specification."""
        if isinstance(num_epochs, int):
            return num_epochs
        elif isinstance(num_epochs, dict):
            return num_epochs.get(beta, 1000)  # Default to 1000 if beta not in dict
        elif callable(num_epochs):
            return num_epochs(beta)
        else:
            raise ValueError(f"num_epochs must be int, dict, or callable, got {type(num_epochs)}")
    
    # Helper function to determine a0 for each beta
    def get_a0_for_beta(beta, a0):
        """Determine learning rate based on beta value and a0 specification."""
        if isinstance(a0, (int, float)):
            return float(a0)
        elif isinstance(a0, dict):
            return a0.get(beta, 1e-1)  # Default to 1e-1 if beta not in dict
        elif callable(a0):
            return a0(beta)
        else:
            raise ValueError(f"a0 must be int, float, dict, or callable, got {type(a0)}")
    
    # Ensure beta=0 is included for proper bound computation
    # Beta=0 corresponds to pure SGLD noise
    extended_beta_values = list(beta_values)
    if 0.0 not in extended_beta_values and 0 not in extended_beta_values:
        extended_beta_values = [0.0] + extended_beta_values
        print(f"Added beta=0 for proper generalization bound computation")
    
    results = {}
    
    print(f"\nConfiguration:")
    print(f"Epochs per beta:")
    for beta in sorted(extended_beta_values):
        epochs = get_epochs_for_beta(beta, num_epochs)
        print(f"  Beta {beta}: {epochs} epochs")
    
    print(f"Learning rate (a0) per beta:")
    for beta in sorted(extended_beta_values):
        current_a0 = get_a0_for_beta(beta, a0)
        print(f"  Beta {beta}: {current_a0}")
    print()
    
    for beta in sorted(extended_beta_values):
        current_epochs = get_epochs_for_beta(beta, num_epochs)
        current_a0 = get_a0_for_beta(beta, a0)
        print(f"\n{'='*60}")
        if beta == 0.0:
            print(f"Running experiments for beta = {beta} (Pure SGLD noise)")
        else:
            print(f"Running experiments for beta = {beta}")
        print(f"Training epochs: {current_epochs}")
        print(f"Learning rate (a0): {current_a0}")
        print(f"{'='*60}")
        
        # Storage for this beta value
        final_train_losses = []
        final_test_losses = []
        final_train_01_losses = []
        final_test_01_losses = []
        
        for rep in range(num_repetitions):
            print(f"Repetition {rep+1}/{num_repetitions} for beta = {beta}")
            
            # Create fresh dataset and model for each repetition
            if dataset_type == 'mnist':
                mnist_classes = [0, 1]
                train_loader, test_loader = get_mnist_binary_dataloaders(
                    classes=mnist_classes,
                    n_train_per_class=5000,
                    n_test_per_class=1000,
                    batch_size=128,
                    random_seed=rep,  # Different seed for each repetition
                    normalize=True
                )
            elif use_random_labels:
                train_loader, test_loader = get_synth_dataloaders_random_labels(
                    batch_size=10, 
                    random_seed=rep  # Different seed for each repetition
                )
            else:
                train_loader, test_loader = get_synth_dataloaders(
                    batch_size=10, 
                    random_seed=rep  # Different seed for each repetition
                )
            
            # Create fresh model 
            if dataset_type == 'mnist':
                model = MNISTNN(input_dim=28*28, hidden_dim=500, output_dim=1)
            else:
                # For SYNTH dataset, use the SynthNN model
                model = SynthNN(input_dim=4, hidden_dim=500) # TODO: Adjust hidden_dim as needed
            
            # Train the model
            train_losses, test_losses, _, _, train_01_losses, test_01_losses, _ = train_sgld_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=current_epochs,
                a0=current_a0,
                b=b,
                sigma_gauss_prior=sigma_gauss_prior,
                beta=beta,
                device=device,
                dataset_type=dataset_type,
                l_max=l_max
            )
            
            # Store final values (last epoch)
            final_train_losses.append(train_losses[-1])
            final_test_losses.append(test_losses[-1])
            final_train_01_losses.append(train_01_losses[-1])
            final_test_01_losses.append(test_01_losses[-1])
            
            if (rep + 1) % 10 == 0:
                print(f"  Completed {rep+1} repetitions for beta = {beta}")
        
        # Convert to numpy arrays for easier computation
        final_train_losses = np.array(final_train_losses)
        final_test_losses = np.array(final_test_losses)
        final_train_01_losses = np.array(final_train_01_losses)
        final_test_01_losses = np.array(final_test_01_losses)
        
        # Compute statistics
        results[beta] = {
            'train_bce_mean': np.mean(final_train_losses),
            'train_bce_var': np.var(final_train_losses),
            'train_bce_std': np.std(final_train_losses),
            'test_bce_mean': np.mean(final_test_losses),
            'test_bce_var': np.var(final_test_losses),
            'test_bce_std': np.std(final_test_losses),
            'train_01_mean': np.mean(final_train_01_losses),
            'train_01_var': np.var(final_train_01_losses),
            'train_01_std': np.std(final_train_01_losses),
            'test_01_mean': np.mean(final_test_01_losses),
            'test_01_var': np.var(final_test_01_losses),
            'test_01_std': np.std(final_test_01_losses),
            'raw_train_bce': final_train_losses.tolist(),
            'raw_test_bce': final_test_losses.tolist(),
            'raw_train_01': final_train_01_losses.tolist(),
            'raw_test_01': final_test_01_losses.tolist()
        }
        
        print(f"Beta {beta} completed:")
        print(f"  Train BCE: {results[beta]['train_bce_mean']:.4f} ± {results[beta]['train_bce_std']:.4f} (var: {results[beta]['train_bce_var']:.6f})")
        print(f"  Test BCE:  {results[beta]['test_bce_mean']:.4f} ± {results[beta]['test_bce_std']:.4f} (var: {results[beta]['test_bce_var']:.6f})")
        print(f"  Train 0-1: {results[beta]['train_01_mean']:.4f} ± {results[beta]['train_01_std']:.4f} (var: {results[beta]['train_01_var']:.6f})")
        print(f"  Test 0-1:  {results[beta]['test_01_mean']:.4f} ± {results[beta]['test_01_std']:.4f} (var: {results[beta]['test_01_var']:.6f})")
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Beta values tested: {sorted(results.keys())}")
    print(f"Original beta values: {beta_values}")
    if 0.0 in results and 0.0 not in beta_values:
        print(f"Note: Added beta=0 automatically for proper generalization bound computation")
    print(f"Total experiments completed: {len(results)} beta values × {num_repetitions} repetitions each")
    
    # Create a nice summary table
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Beta':<8} {'Train Error':<12} {'Test Error':<12} {'Min Train Error':<15}")
    print(f"{'(β)':<8} {'(Mean)':<12} {'(Mean)':<12} {'(Min per β)':<15}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*15}")
    
    # Print results for each beta, starting with beta=0 if present
    sorted_betas = sorted(results.keys())
    for beta in sorted_betas:
        train_error = results[beta]['train_bce_mean']
        test_error = results[beta]['test_bce_mean']
        
        # For this beta, find the minimum train error among all repetitions
        min_train_error_for_beta = min(results[beta]['raw_train_bce'])
        
        print(f"{beta:<8.1f} {train_error:<12.4f} {test_error:<12.4f} {min_train_error_for_beta:<15.4f}")
    
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*15}")
    print(f"Notes:")
    print(f"  - Train/Test Error: Bounded Cross-Entropy (BCE) Loss")
    print(f"  - Min Train Error: Lowest train error among all repetitions for each β")
    print(f"  - β=0: Pure SGLD noise")
    print(f"  - Higher β: More SGLD noise, potentially better generalization")
    
    return results


def create_beta_epochs_mapping(beta_epochs_pairs):
    """
    Create a beta -> epochs mapping from a list of (beta, epochs) pairs.
    
    Args:
        beta_epochs_pairs: List of (beta, epochs) tuples
        
    Returns:
        Dictionary mapping beta values to epochs
        
    Example:
        epochs_map = create_beta_epochs_mapping([
            (0, 100), (1, 100), (10, 1000), (50, 1000), (200, 1000)
        ])
    """
    return dict(beta_epochs_pairs)


def create_adaptive_epochs_function(low_beta_epochs=100, high_beta_epochs=1000, threshold=5.0):
    """
    Create a function that adaptively determines epochs based on beta value.
    
    Args:
        low_beta_epochs: Epochs for beta values <= threshold
        high_beta_epochs: Epochs for beta values > threshold  
        threshold: Beta threshold to switch between low and high epochs
        
    Returns:
        Function that takes beta and returns appropriate epochs
        
    Example:
        epochs_func = create_adaptive_epochs_function(100, 1000, 5.0)
        # epochs_func(1) -> 100, epochs_func(10) -> 1000
    """
    def epochs_function(beta):
        return low_beta_epochs if beta <= threshold else high_beta_epochs
    return epochs_function


def create_beta_a0_mapping(beta_a0_pairs):
    """
    Create a beta -> a0 mapping from a list of (beta, a0) pairs.
    
    Args:
        beta_a0_pairs: List of (beta, a0) tuples
        
    Returns:
        Dictionary mapping beta values to learning rates
        
    Example:
        a0_map = create_beta_a0_mapping([
            (0, 0.01), (1, 0.01), (10, 0.1), (50, 0.1), (200, 0.1)
        ])
    """
    return dict(beta_a0_pairs)


def create_adaptive_a0_function(low_beta_a0=0.01, high_beta_a0=0.1, threshold=5.0):
    """
    Create a function that adaptively determines learning rate based on beta value.
    
    Args:
        low_beta_a0: Learning rate for beta values <= threshold
        high_beta_a0: Learning rate for beta values > threshold  
        threshold: Beta threshold to switch between low and high learning rates
        
    Returns:
        Function that takes beta and returns appropriate learning rate
        
    Example:
        a0_func = create_adaptive_a0_function(0.01, 0.1, 5.0)
        # a0_func(1) -> 0.01, a0_func(10) -> 0.1
    """
    def a0_function(beta):
        return low_beta_a0 if beta <= threshold else high_beta_a0
    return a0_function
