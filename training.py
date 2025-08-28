"""
Training and experiment orchestration for the Gibbs generalization bound experiments.

This module contains functions for training models with SGLD and running
experiments across different beta values with multiple repetitions.
"""

import numpy as np
import torch
import torch.optim as optim
import time
import json
import os
import csv
from datetime import datetime
from losses import BoundedCrossEntropyLoss, ZeroOneLoss, TangentLoss
from models import SynthNN, MNISTNN, initialize_kaiming_and_get_prior_sigma
from sgld import SGLD
from dataset import (get_synth_dataloaders, get_synth_dataloaders_random_labels,
                    get_mnist_binary_dataloaders, get_mnist_binary_dataloaders_random_labels)
from torch.nn import BCEWithLogitsLoss

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


def save_output_label_products_to_csv(train_data, test_data, filename_prefix, beta_values):
    """
    Save output * batch_y values to CSV files for training and test data.
    
    Args:
        train_data: Dictionary with structure {beta: {rep: [output*batch_y values]}}
        test_data: Dictionary with structure {beta: {rep: [output*batch_y values]}}
        filename_prefix: Prefix for the CSV files (e.g., 'experiment_2024')
        beta_values: List of beta values for column ordering
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CSV directory if it doesn't exist
    os.makedirs('csv_outputs', exist_ok=True)
    
    # Save training data
    train_filename = f"csv_outputs/{filename_prefix}_train_output_label_products_{timestamp}.csv"
    save_data_to_csv(train_data, train_filename, beta_values, 'Training')
    
    # Save test data
    test_filename = f"csv_outputs/{filename_prefix}_test_output_label_products_{timestamp}.csv"
    save_data_to_csv(test_data, test_filename, beta_values, 'Test')
    
    print(f"📊 CSV files saved:")
    print(f"   Training data: {train_filename}")
    print(f"   Test data: {test_filename}")
    
    return train_filename, test_filename


def save_data_to_csv(data, filename, beta_values, data_type):
    """
    Save output*label product data to a CSV file.
    
    Args:
        data: Dictionary with structure {beta: {rep: [output*batch_y values]}}
        filename: Path to save the CSV file
        beta_values: List of beta values for column ordering
        data_type: 'Training' or 'Test' for logging purposes
    """
    # Determine the maximum number of repetitions and samples per repetition
    max_reps = 0
    max_samples = 0
    
    for beta in beta_values:
        if beta in data:
            max_reps = max(max_reps, len(data[beta]))
            for rep_idx, rep_data in data[beta].items():
                max_samples = max(max_samples, len(rep_data))
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header with metadata
        writer.writerow([f"{data_type} Data: Output * Label Products"])
        writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow([f"Beta values: {beta_values}"])
        writer.writerow([f"Max repetitions: {max_reps}"])
        writer.writerow([f"Max samples per repetition: {max_samples}"])
        writer.writerow([])  # Empty row for separation
        
        # Write column headers
        headers = ['Beta', 'Repetition', 'Sample_Count']
        for i in range(max_samples):
            headers.append(f'Output_Label_Product_{i+1}')
        writer.writerow(headers)
        
        # Write data rows
        for beta in sorted(beta_values):
            if beta in data:
                for rep_idx, rep_data in data[beta].items():
                    row = [beta, rep_idx + 1, len(rep_data)]
                    # Add the output*label products
                    for val in rep_data:
                        row.append(val)
                    # Pad with empty values if this repetition has fewer samples
                    while len(row) < len(headers):
                        row.append('')
                    writer.writerow(row)
    
    print(f"   {data_type} CSV saved: {filename} ({max_reps} reps × {max_samples} samples per rep)")


def collect_output_label_products(outputs, batch_y, dataset_type='synth'):
    """
    Compute output * batch_y products for the current batch.
    
    Args:
        outputs: Model outputs (logits)
        batch_y: True labels
        dataset_type: 'synth' or 'mnist' for proper handling
        
    Returns:
        List of output * label products for this batch
    """
    # Change the labels from {0,1} to {1,-1}
    batch_y = 2 * batch_y - 1
    # Handle different output shapes for SYNTH vs MNIST
    if dataset_type == 'synth':
        # For synth, outputs are typically (batch_size, 1) or (batch_size,)
        if outputs.dim() > 1:
            outputs = outputs.squeeze()
        products = outputs * batch_y
    else:
        # For mnist, similar handling
        if outputs.dim() > 1:
            outputs = outputs.squeeze()
        products = outputs * batch_y
    
    # Convert to list of float values
    return products.detach().cpu().numpy().tolist()


def train_sgld_model(model, train_loader, test_loader, num_epochs: int = 100, 
                     a0: float = 1e-3, b: float = 0.5, sigma_gauss_prior: float = 0.1, 
                     beta: float = 1.0, device: str = 'cpu', dataset_type: str = 'synth',
                     l_max: float = 2.0, collect_output_products: bool = False):
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
        collect_output_products: If True, collect output * label products
        
    Returns:
        Tuple containing: (train_losses, test_losses, train_accuracies, test_accuracies, 
                          train_zero_one_losses, test_zero_one_losses, learning_rates)
        If collect_output_products is True, also returns (train_output_products, test_output_products)
    """
    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    criterion = BoundedCrossEntropyLoss(ell_max=l_max)
    # criterion = TangentLoss()
    # criterion = BCEWithLogitsLoss()  # Use standard BCE for SGLD optimization
    zero_one_criterion = ZeroOneLoss()

    # Check if we're using BCE for optimization (to determine if transformation is needed)
    using_bce_for_optimization = isinstance(criterion, BoundedCrossEntropyLoss)
    
    # Initialize SGLD optimizer with inverse temperature
    optimizer = SGLD(model.parameters(), lr=a0, sigma_gauss_prior=sigma_gauss_prior, 
                     beta=beta, add_noise=True)
    
    # Learning rate scheduler with threshold: lr_t = max(a0 * t^(-b), 0.01)
    # This stops the decay when learning rate reaches 0.01
    lr_threshold = 0.005
    def lr_lambda_with_threshold(epoch):
        power_law_lr = (epoch + 1) ** (-b)
        # Convert to actual learning rate value and check threshold
        actual_lr = a0 * power_law_lr
        if actual_lr <= lr_threshold:
            return lr_threshold / a0  # Return the ratio that gives us the threshold
        else:
            return power_law_lr  # Return the normal power law ratio
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_with_threshold)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    train_zero_one_losses = []
    test_zero_one_losses = []
    learning_rates = []
    
    # Initialize output product collections if requested
    train_output_products = [] if collect_output_products else None
    test_output_products = [] if collect_output_products else None
    
    print(f"Training with SGLD: a0={a0}, b={b}, sigma_gauss_prior={sigma_gauss_prior}, beta={beta}")
    print(f"Learning rate scheduler: power law decay with threshold at {lr_threshold}")
    print(f"Dataset type: {dataset_type}, Device: {device}")
    if collect_output_products:
        print(f"Collecting output * label products for CSV export")
    
    # Progress tracking
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_zero_one_total = 0.0
        train_correct = 0
        train_total = 0
        epoch_train_products = [] if collect_output_products else None
        
        for batch_x, batch_y in train_loader:
            # Use non_blocking=True for faster GPU transfer
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # Use more efficient zero_grad
            optimizer.zero_grad(set_to_none=True)
            
            # Standard precision training
            outputs = model(batch_x)
            
            # Collect output * label products if requested
            if collect_output_products:
                batch_products = collect_output_label_products(outputs, batch_y, dataset_type)
                epoch_train_products.extend(batch_products)
             
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
        
        # Store epoch training products
        if collect_output_products:
            train_output_products.append(epoch_train_products)
        
        # Step the learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        # scheduler.step()
        
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
        epoch_test_products = [] if collect_output_products else None
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                outputs = model(batch_x)
                
                # Collect output * label products if requested
                if collect_output_products:
                    batch_products = collect_output_label_products(outputs, batch_y, dataset_type)
                    epoch_test_products.extend(batch_products)
                
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
        
        # Store epoch test products
        if collect_output_products:
            test_output_products.append(epoch_test_products)
        
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
    
    if collect_output_products:
        return (train_losses, test_losses, train_accuracies, test_accuracies, 
                train_zero_one_losses, test_zero_one_losses, learning_rates,
                train_output_products, test_output_products)
    else:
        return (train_losses, test_losses, train_accuracies, test_accuracies, 
                train_zero_one_losses, test_zero_one_losses, learning_rates)


def save_checkpoint(results, checkpoint_path, experiment_config, completed_repetitions):
    """
    Save experiment checkpoint to allow resuming interrupted runs.
    
    Args:
        results: Current results dictionary
        checkpoint_path: Path to save checkpoint file
        experiment_config: Configuration dictionary
        completed_repetitions: Number of completed repetitions
    """
    checkpoint_data = {
        'results': results,
        'experiment_config': experiment_config,
        'completed_repetitions': completed_repetitions,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save with temporary name first, then rename (atomic operation)
    temp_path = checkpoint_path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    os.rename(temp_path, checkpoint_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path):
    """
    Load experiment checkpoint to resume interrupted runs.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (results, experiment_config, completed_repetitions) or (None, None, 0) if no checkpoint
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None, None, 0
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        results = checkpoint_data['results']
        experiment_config = checkpoint_data['experiment_config'] 
        completed_repetitions = checkpoint_data['completed_repetitions']
        timestamp = checkpoint_data.get('timestamp', 'unknown')
        
        print(f"📂 Checkpoint loaded from {timestamp}")
        print(f"🔄 Resuming from repetition {completed_repetitions + 1}")
        
        return results, experiment_config, completed_repetitions
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None, None, 0


def generate_checkpoint_filename(experiment_config):
    """Generate a descriptive checkpoint filename based on experiment configuration."""
    beta_str = f"beta{min(experiment_config['beta_values'])}-{max(experiment_config['beta_values'])}"
    dataset_str = experiment_config['dataset_type']
    if experiment_config.get('use_random_labels', False):
        dataset_str += '_random'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"checkpoints/checkpoint_{dataset_str}_{beta_str}_rep{experiment_config['num_repetitions']}_{timestamp}.json"


def run_beta_experiments(beta_values, num_repetitions=50, num_epochs=10000, 
                         a0=1e-1, b=0.5, sigma_gauss_prior=1000000, 
                         device='cpu', dataset_type='synth', use_random_labels=False,
                         l_max=2.0, mnist_classes=None, train_loader=None, test_loader=None,
                         checkpoint_path=None, save_every=1, save_output_products_csv=False):
    """
    Run experiments across different beta values with multiple repetitions.
    
    Modified to run all betas for each repetition, then move to next repetition.
    Includes checkpoint saving/loading for resuming interrupted runs.
    
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
        mnist_classes: MNIST classes specification (only used if dataloaders not provided)
        train_loader: Pre-created training DataLoader (if None, will create based on other params)
        test_loader: Pre-created test DataLoader (if None, will create based on other params)
        checkpoint_path: Path to checkpoint file (if None, auto-generated)
        save_every: Save checkpoint every N repetitions (default: 1)
        save_output_products_csv: If True, save output * label products to CSV files
        
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
    extended_beta_values = list(beta_values)
    if 0.0 not in extended_beta_values and 0 not in extended_beta_values:
        extended_beta_values = [0.0] + extended_beta_values
        print(f"Added beta=0 for proper generalization bound computation")
    
    # Create experiment configuration for checkpointing
    experiment_config = {
        'beta_values': beta_values,
        'extended_beta_values': extended_beta_values,
        'num_repetitions': num_repetitions,
        'num_epochs': num_epochs,
        'a0': a0,
        'b': b,
        'sigma_gauss_prior': sigma_gauss_prior,
        'device': device,
        'dataset_type': dataset_type,
        'use_random_labels': use_random_labels,
        'l_max': l_max,
        'mnist_classes': mnist_classes,
        'train_dataset_size': len(train_loader.dataset) if train_loader else None,
        'test_dataset_size': len(test_loader.dataset) if test_loader else None
    }
    
    # Generate checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = generate_checkpoint_filename(experiment_config)
    
    # Try to load existing checkpoint
    results, loaded_config, completed_repetitions = load_checkpoint(checkpoint_path)
    
    if results is not None:
        print(f"🔄 Resuming experiment from repetition {completed_repetitions + 1}/{num_repetitions}")
        # Verify configuration compatibility
        if loaded_config['extended_beta_values'] != extended_beta_values:
            print("⚠️  Warning: Beta values in checkpoint don't match current configuration")
    else:
        print(f"🆕 Starting new experiment")
        completed_repetitions = 0
        # Initialize results structure
        results = {}
        for beta in extended_beta_values:
            results[beta] = {
                'raw_train_bce': [],
                'raw_test_bce': [],
                'raw_train_01': [],
                'raw_test_01': []
            }
    
    # Initialize output products storage if requested
    train_output_products_data = {} if save_output_products_csv else None
    test_output_products_data = {} if save_output_products_csv else None
    
    if save_output_products_csv:
        for beta in extended_beta_values:
            train_output_products_data[beta] = {}
            test_output_products_data[beta] = {}
        print(f"📊 Output * label products will be saved to CSV files")
    
    print(f"\nConfiguration:")
    print(f"Epochs per beta:")
    for beta in sorted(extended_beta_values):
        epochs = get_epochs_for_beta(beta, num_epochs)
        print(f"  Beta {beta}: {epochs} epochs")
    
    print(f"Learning rate (a0) per beta:")
    for beta in sorted(extended_beta_values):
        current_a0 = get_a0_for_beta(beta, a0)
        print(f"  Beta {beta}: {current_a0}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Save frequency: every {save_every} repetitions")
    print()
    
    # Main experiment loop: repetitions first, then betas
    for rep in range(completed_repetitions, num_repetitions):
        print(f"\n{'='*80}")
        print(f"REPETITION {rep+1}/{num_repetitions}")
        print(f"{'='*80}")
        
        # Run all beta values for this repetition
        for beta in sorted(extended_beta_values):
            current_epochs = get_epochs_for_beta(beta, num_epochs)
            current_a0 = get_a0_for_beta(beta, a0)
            
            print(f"\n--- Beta = {beta} (Rep {rep+1}) ---")
            print(f"Epochs: {current_epochs}, Learning rate: {current_a0}")
            
            # Use provided dataloaders or create fresh dataset for each repetition
            if train_loader is not None and test_loader is not None:
                # Use the pre-created dataloaders (same dataset for all repetitions)
                current_train_loader = train_loader
                current_test_loader = test_loader
            else:
                # Create fresh dataset for each repetition (legacy behavior)
                if dataset_type == 'mnist':
                    if use_random_labels:
                        current_train_loader, current_test_loader = get_mnist_binary_dataloaders_random_labels(
                            classes=mnist_classes if mnist_classes else [[0], [1]],
                            n_train_per_group=5000,
                            n_test_per_group=1000,
                            batch_size=128,
                            random_seed=rep,  # Different seed for each repetition
                            normalize=True
                        )
                    else:
                        current_train_loader, current_test_loader = get_mnist_binary_dataloaders(
                            classes=mnist_classes if mnist_classes else [[0], [1]],
                            n_train_per_group=5000,
                            n_test_per_group=1000,
                            batch_size=128,
                            random_seed=rep,  # Different seed for each repetition
                            normalize=True
                        )
                elif use_random_labels:
                    current_train_loader, current_test_loader = get_synth_dataloaders_random_labels(
                        batch_size=10, 
                        random_seed=rep  # Different seed for each repetition
                    )
                else:
                    current_train_loader, current_test_loader = get_synth_dataloaders(
                        batch_size=10, 
                        random_seed=rep  # Different seed for each repetition
                    )
            
            # Create fresh model for each beta-repetition combination
            if dataset_type == 'mnist':
                model = MNISTNN(input_dim=28*28, hidden_dim=500, output_dim=1)
            else:
                model = SynthNN(input_dim=4, hidden_dim=500)
            
            # Train the model
            training_results = train_sgld_model(
                model=model,
                train_loader=current_train_loader,
                test_loader=current_test_loader,
                num_epochs=current_epochs,
                a0=current_a0,
                b=b,
                sigma_gauss_prior=sigma_gauss_prior,
                beta=beta,
                device=device,
                dataset_type=dataset_type,
                l_max=l_max,
                collect_output_products=save_output_products_csv
            )
            
            # Unpack results based on whether output products were collected
            if save_output_products_csv:
                (train_losses, test_losses, _, _, train_01_losses, test_01_losses, _, 
                 train_output_products, test_output_products) = training_results
                
                # Store output products for this repetition and beta
                # We only store the final epoch's data (last element of each list)
                train_output_products_data[beta][rep] = train_output_products[-1] if train_output_products else []
                test_output_products_data[beta][rep] = test_output_products[-1] if test_output_products else []
            else:
                (train_losses, test_losses, _, _, train_01_losses, test_01_losses, _) = training_results
            
            # Store final values (last epoch)
            results[beta]['raw_train_bce'].append(train_losses[-1])
            results[beta]['raw_test_bce'].append(test_losses[-1])
            results[beta]['raw_train_01'].append(train_01_losses[-1])
            results[beta]['raw_test_01'].append(test_01_losses[-1])
            
            print(f"  Final - Train BCE: {train_losses[-1]:.4f}, Test BCE: {test_losses[-1]:.4f}, "
                  f"Train 0-1: {train_01_losses[-1]:.4f}, Test 0-1: {test_01_losses[-1]:.4f}")
        
        # Save checkpoint after completing this repetition
        if (rep + 1) % save_every == 0 or rep + 1 == num_repetitions:
            # Compute statistics for checkpoint
            checkpoint_results = {}
            for beta in extended_beta_values:
                raw_train_bce = np.array(results[beta]['raw_train_bce'])
                raw_test_bce = np.array(results[beta]['raw_test_bce'])
                raw_train_01 = np.array(results[beta]['raw_train_01'])
                raw_test_01 = np.array(results[beta]['raw_test_01'])
                
                checkpoint_results[beta] = {
                    'train_bce_mean': np.mean(raw_train_bce),
                    'train_bce_var': np.var(raw_train_bce),
                    'train_bce_std': np.std(raw_train_bce),
                    'test_bce_mean': np.mean(raw_test_bce),
                    'test_bce_var': np.var(raw_test_bce),
                    'test_bce_std': np.std(raw_test_bce),
                    'train_01_mean': np.mean(raw_train_01),
                    'train_01_var': np.var(raw_train_01),
                    'train_01_std': np.std(raw_train_01),
                    'test_01_mean': np.mean(raw_test_01),
                    'test_01_var': np.var(raw_test_01),
                    'test_01_std': np.std(raw_test_01),
                    'raw_train_bce': results[beta]['raw_train_bce'],
                    'raw_test_bce': results[beta]['raw_test_bce'],
                    'raw_train_01': results[beta]['raw_train_01'],
                    'raw_test_01': results[beta]['raw_test_01']
                }
            
            save_checkpoint(checkpoint_results, checkpoint_path, experiment_config, rep + 1)
            
            # Save output * label CSV files at the same time as checkpoint if requested
            if save_output_products_csv and train_output_products_data is not None:
                # Generate filename prefix based on experiment parameters
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_desc = f"{dataset_type}"
                if use_random_labels:
                    experiment_desc += "_random"
                experiment_desc += f"_beta{min(beta_values)}-{max(beta_values)}"
                experiment_desc += f"_rep{rep+1}-of-{num_repetitions}"
                
                filename_prefix = f"checkpoint_{experiment_desc}_{timestamp}"
                
                # Create a copy of data for the current repetitions completed
                current_train_data = {}
                current_test_data = {}
                for beta in extended_beta_values:
                    current_train_data[beta] = {k: v for k, v in train_output_products_data[beta].items() if k <= rep}
                    current_test_data[beta] = {k: v for k, v in test_output_products_data[beta].items() if k <= rep}
                
                # Save CSV files for current progress
                train_csv_path, test_csv_path = save_output_label_products_to_csv(
                    current_train_data, 
                    current_test_data, 
                    filename_prefix, 
                    sorted(extended_beta_values)
                )
                
                print(f"📊 Checkpoint CSV files saved:")
                print(f"   📁 Training data: {train_csv_path}")
                print(f"   📁 Test data: {test_csv_path}")
        
        print(f"\n✅ Completed repetition {rep+1}/{num_repetitions}")
        
        # Print interim summary
        print(f"\nCurrent results after {rep+1} repetitions:")
        for beta in sorted(extended_beta_values):
            n_reps = len(results[beta]['raw_train_bce'])
            if n_reps > 0:
                train_mean = np.mean(results[beta]['raw_train_bce'])
                test_mean = np.mean(results[beta]['raw_test_bce'])
                print(f"  Beta {beta}: Train BCE: {train_mean:.4f}, Test BCE: {test_mean:.4f} ({n_reps} reps)")
    
    # Compute final statistics
    final_results = {}
    for beta in extended_beta_values:
        raw_train_bce = np.array(results[beta]['raw_train_bce'])
        raw_test_bce = np.array(results[beta]['raw_test_bce'])
        raw_train_01 = np.array(results[beta]['raw_train_01'])
        raw_test_01 = np.array(results[beta]['raw_test_01'])
        
        final_results[beta] = {
            'train_bce_mean': np.mean(raw_train_bce),
            'train_bce_var': np.var(raw_train_bce),
            'train_bce_std': np.std(raw_train_bce),
            'test_bce_mean': np.mean(raw_test_bce),
            'test_bce_var': np.var(raw_test_bce),
            'test_bce_std': np.std(raw_test_bce),
            'train_01_mean': np.mean(raw_train_01),
            'train_01_var': np.var(raw_train_01),
            'train_01_std': np.std(raw_train_01),
            'test_01_mean': np.mean(raw_test_01),
            'test_01_var': np.var(raw_test_01),
            'test_01_std': np.std(raw_test_01),
            'raw_train_bce': results[beta]['raw_train_bce'],
            'raw_test_bce': results[beta]['raw_test_bce'],
            'raw_train_01': results[beta]['raw_train_01'],
            'raw_test_01': results[beta]['raw_test_01']
        }
    
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
        train_error = final_results[beta]['train_bce_mean']
        test_error = final_results[beta]['test_bce_mean']
        
        # For this beta, find the minimum train error among all repetitions
        min_train_error_for_beta = min(results[beta]['raw_train_bce'])
        
        print(f"{beta:<8.1f} {train_error:<12.4f} {test_error:<12.4f} {min_train_error_for_beta:<15.4f}")
    
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*15}")
    
    # Save output products to CSV if requested (final version)
    if save_output_products_csv:
        print(f"\n{'='*60}")
        print(f"SAVING FINAL OUTPUT * LABEL PRODUCTS TO CSV")
        print(f"{'='*60}")
        
        # Generate filename prefix based on experiment parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_desc = f"{dataset_type}"
        if use_random_labels:
            experiment_desc += "_random"
        experiment_desc += f"_beta{min(beta_values)}-{max(beta_values)}"
        experiment_desc += f"_rep{num_repetitions}_FINAL"
        
        filename_prefix = f"experiment_{experiment_desc}_{timestamp}"
        
        # Save CSV files
        train_csv_path, test_csv_path = save_output_label_products_to_csv(
            train_output_products_data, 
            test_output_products_data, 
            filename_prefix, 
            sorted(extended_beta_values)
        )
        
        print(f"✅ Final CSV files successfully created:")
        print(f"   📁 Training data: {train_csv_path}")
        print(f"   📁 Test data: {test_csv_path}")
    
    return final_results


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
