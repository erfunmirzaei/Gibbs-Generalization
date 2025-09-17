"""
Training and experiment orchestration for the Gibbs generalization bound experiments.

This module contains functions for training models with SGLD and running
experiments across different beta values with multiple repetitions.
"""
import math
import numpy as np
import torch
import torch.optim as optim
import time
import json
import os
import csv
from datetime import datetime
from losses import BoundedCrossEntropyLoss, ZeroOneLoss, TangentLoss, SavageLoss
from torch.nn import BCEWithLogitsLoss
from models import  initialize_nn_weights_gaussian, FCN1L,FCN2L
from sgld import SGLD

def transform_bce_to_unit_interval(bce_loss, l_max=2.0):
    """
    Transform BCE loss to [0,1] interval for bounded loss computation.
    
    Uses the formula: (BCE + ln(1-exp(-l_max))) / (l_max + ln(1-exp(-l_max)))
    This transformation ensures the loss is bounded in [0,1] as required for 
    generalization bound calculations.
    
    Args:
        bce_loss (torch.Tensor or float): BCE loss value to transform.
        l_max (float, optional): Maximum loss value parameter. Defaults to 2.0.
    
    Returns:
        torch.Tensor or float: Transformed loss in [0,1] interval.
    """
    ln_term = torch.log(1 - torch.exp(torch.tensor(-l_max)))
    numerator = bce_loss + ln_term
    denominator = l_max + ln_term
    return numerator / denominator

def save_moving_average_losses_to_csv(                 
        list_train_BCE_losses,
        list_test_BCE_losses,
        list_train_01_losses,
        list_test_01_losses,
        list_EMA_train_BCE_losses,
        list_EMA_test_BCE_losses,   
        list_EMA_train_01_losses,
        list_EMA_test_01_losses,
        filename_prefix,
        beta_values,
        sample_size,
        summary_string
 ):
    """
    Save experimental results with moving average losses to a CSV file.
    
    Creates a CSV file containing both raw and exponential moving average (EMA) 
    losses for training and test sets across different beta values.
    
    Args:
        list_train_BCE_losses (list): Training BCE losses for each beta.
        list_test_BCE_losses (list): Test BCE losses for each beta.
        list_train_01_losses (list): Training 0-1 losses for each beta.
        list_test_01_losses (list): Test 0-1 losses for each beta.
        list_EMA_train_BCE_losses (list): EMA of training BCE losses.
        list_EMA_test_BCE_losses (list): EMA of test BCE losses.
        list_EMA_train_01_losses (list): EMA of training 0-1 losses.
        list_EMA_test_01_losses (list): EMA of test 0-1 losses.
        filename_prefix (str): Prefix for the CSV filename.
        beta_values (list): List of beta values used in experiments.
        sample_size (int): Size of the training dataset.
        summary_string (str): Summary of experimental configuration.
    
    Returns:
        str: Path to the saved CSV file.
    """    
    # Create CSV directory if it doesn't exist
    os.makedirs('csv_EMA', exist_ok=True)
    
    # Save training data
    filename = f"csv_EMA/{filename_prefix}.csv"

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write column headers
        headers = ['Sample_size', 'Beta','BCE_Train', 'BCE_Test', '0-1_Train', '0-1_Test', 
                   'EMA_BCE_Train', 'EMA_BCE_Test', 'EMA_0-1_Train', 'EMA_0-1_Test', ]
        writer.writerow(headers)
        
        # Write data rows
        for i, beta in enumerate(sorted(beta_values)):
            row = [sample_size, beta, list_train_BCE_losses[i], list_test_BCE_losses[i], 
                   list_train_01_losses[i], list_test_01_losses[i],
                   list_EMA_train_BCE_losses[i], list_EMA_test_BCE_losses[i],
                   list_EMA_train_01_losses[i], list_EMA_test_01_losses[i]]
            writer.writerow(row)
        
        writer.writerow([])  # Empty row for separation
        writer.writerow(['Summary:', summary_string])
    print(f"   EMA CSV saved: {filename} ({len(beta_values)} beta values)")

    return filename

# Helper function to determine a0 for each beta
def get_a0_for_beta(beta, a0):
    """
    Determine learning rate based on beta value and a0 specification.
    
    Supports flexible learning rate specification: constant value, beta-specific
    mapping, or callable function.
    
    Args:
        beta (float): Beta (inverse temperature) value.
        a0 (int, float, dict, or callable): Learning rate specification.
            - int/float: Constant learning rate for all betas.
            - dict: {beta: learning_rate} mapping.
            - callable: Function that takes beta and returns learning rate.
    
    Returns:
        float: Learning rate for the given beta value.
    
    Raises:
        ValueError: If a0 type is not supported.
    """
    if isinstance(a0, (int, float)):
        return float(a0)
    elif isinstance(a0, dict):
        return a0.get(beta, 1e-1)  # Default to 1e-1 if beta not in dict
    elif callable(a0):
        return a0(beta)
    else:
        raise ValueError(f"a0 must be int, float, dict, or callable, got {type(a0)}")

def train_sgld_model(loss, model, train_loader, test_loader, min_epochs, 
                     a0, b, sigma_gauss_prior, 
                     beta, device, dataset_type,
                     l_max, alpha_average, alpha_stop,  eta, eps):
    """
    Train a neural network using Stochastic Gradient Langevin Dynamics (SGLD).
    
    Implements SGLD training with bounded cross-entropy loss, exponential moving
    average (EMA) convergence monitoring, and adaptive learning rate scheduling.
    For beta=0, samples from the prior distribution instead of training.
    
    Args:
        loss (str): Loss function name ('bce', 'tangent', or 'savage').
        model (nn.Module): Neural network model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        min_epochs (int): Minimum number of epochs to train before checking convergence.
        a0 (float): Initial learning rate.
        b (float): Learning rate decay exponent (lr_t = a0 * t^(-b)).
        sigma_gauss_prior (float): Standard deviation of Gaussian prior for weights.
        beta (float): Inverse temperature parameter. If 0, samples from prior.
        device (torch.device or str): Device for computation ('cpu' or 'cuda').
        dataset_type (str): Dataset type ('synth' or 'mnist') for loss computation.
        l_max (float): Maximum loss value for bounded transformation.
        alpha_average (float): EMA smoothing factor for loss averaging.
        alpha_stop (float): EMA smoothing factor for convergence detection.
        eta (float): Learning rate threshold parameter (lr >= eta/beta).
        eps (float): Convergence threshold for EMA training loss difference.
    
    Returns:
        tuple: Contains (train_losses, test_losses, train_accuracies, test_accuracies,
               train_zero_one_losses, test_zero_one_losses, learning_rates, EMA_train_losses,
               EMA_train_BCE_losses, EMA_test_BCE_losses, EMA_train_zero_one_losses, 
               EMA_test_zero_one_losses).
    """
    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    # Define loss function
    if loss.lower() == 'bbce':
        criterion = BoundedCrossEntropyLoss(ell_max=l_max)
    elif loss.lower() == 'tangent':
        criterion = TangentLoss()
    elif loss.lower() == 'savage':
        criterion = SavageLoss()
    else:
        criterion = BCEWithLogitsLoss()  # Use standard BCE for SGLD optimization
    zero_one_criterion = ZeroOneLoss()

    # Check if we're using BCE for optimization (to determine if transformation is needed)
    using_bce_for_optimization = isinstance(criterion, BoundedCrossEntropyLoss)
    
    # Initialize SGLD optimizer with inverse temperature
    optimizer = SGLD(model.parameters(), lr=a0, sigma_gauss_prior=sigma_gauss_prior, 
                     beta=beta, add_noise=True)
    
    # Learning rate scheduler with threshold: lr_t = max(a0 * t^(-b), 0.01)
    # This stops the decay when learning rate reaches 0.01
    lr_threshold = eta / beta if beta > 0 else 1e-5  # Avoid division by zero for beta=0
    def lr_lambda_with_threshold(epoch):
        power_law_lr = (epoch + 1) ** (-b)
        # Convert to actual learning rate value and check threshold
        actual_lr = a0 * power_law_lr
        if actual_lr <= lr_threshold:
            return lr_threshold / a0  # Return the ratio that gives us the threshold
        else:
            return power_law_lr  # Return the normal power law ratio
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_with_threshold)
    
    train_losses = [1.0]  # Initialize with 0.0 for epoch 0
    test_losses = [1.0]  # Initialize with 0.0 for epoch 0
    train_accuracies = [1.0]  # Initialize with 0.0 for epoch 0
    test_accuracies = [1.0]  # Initialize with 0.0 for epoch 0
    train_zero_one_losses = [1.0]  # Initialize with 0.0 for epoch 0
    test_zero_one_losses = [1.0]  # Initialize with 0.0 for epoch 0
    learning_rates = []
    EMA_train_losses = [0.0, 1.0]  # Initialize with 1.0 for epoch 0
    EMA_alpha = alpha_stop  # Smoothing factor for EMA of loss
    
    EMA_train_BCE_losses = [0.0, 1.0]  # Separate EMA for BCE losses if needed
    EMA_test_BCE_losses = [0.0, 1.0]
    EMA_train_zero_one_losses = [0.0, 1.0]
    EMA_test_zero_one_losses = [0.0, 1.0]
    EMA_alpha_BCE = alpha_average
    
    print(f"Training with SGLD: a0={a0}, b={b}, sigma_gauss_prior={sigma_gauss_prior}, beta={beta}")
    print(f"Learning rate scheduler: power law decay with threshold at {lr_threshold}")
    print(f"Dataset type: {dataset_type}, Device: {device}")
    
    # Progress tracking
    start_time = time.time()
    epoch = 0
    # if beta == 0.0, I have to sample from the prior many times and take the average both for train and test losses
    if beta == 0.0:
        num_prior_samples = 1000

        model_cpu = model.to('cpu')
        for i in range(num_prior_samples):
            learning_rates.append(0.0)  # No learning rate for beta=0 prior sampling
            # Reinitialize model weights from prior
            initialize_nn_weights_gaussian(model_cpu, sigma=sigma_gauss_prior, seed=i)
            # Compute train loss
            model_cpu.eval()
            with torch.no_grad():
                train_loss_total = 0.0
                zero_one_loss_total = 0.0
                for batch_x, batch_y in train_loader:
                    outputs = model_cpu(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    if using_bce_for_optimization:
                        loss = transform_bce_to_unit_interval(loss, l_max)
                    train_loss_total += loss.item()
                    zero_one_loss_total += zero_one_criterion(outputs, batch_y).item()
                train_zero_one_losses.append(zero_one_loss_total / len(train_loader))
                train_losses.append(train_loss_total / len(train_loader))

                # Compute test loss
                test_loss_total = 0.0
                zero_one_loss_total = 0.0
                for batch_x, batch_y in test_loader:
                    outputs = model_cpu(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    if using_bce_for_optimization:
                        loss = transform_bce_to_unit_interval(loss, l_max)
                    test_loss_total += loss.item()
                    zero_one_loss_total += zero_one_criterion(outputs, batch_y).item()
                test_losses.append(test_loss_total / len(test_loader))
                test_zero_one_losses.append(zero_one_loss_total / len(test_loader))

            print(f'Beta=0.0: Averaged over {num_prior_samples} prior samples - '
                f'Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, '
                f'EMA Train Loss: {EMA_train_BCE_losses[-1]:.4f}, EMA Test Loss: {EMA_test_BCE_losses[-1]:.4f}, '
                f'EMA Train Zero-One Loss: {EMA_train_zero_one_losses[-1]:.4f}, EMA Test Zero-One Loss: {EMA_test_zero_one_losses[-1]:.4f}'
                )
            EMA_train_BCE_losses.append(0.5 * EMA_alpha_BCE * train_losses[-1] + 0.5 * EMA_alpha_BCE * train_losses[-2] + (1 - EMA_alpha_BCE) * EMA_train_BCE_losses[-1])
            EMA_test_BCE_losses.append(0.5 * EMA_alpha_BCE * test_losses[-1] + 0.5 * EMA_alpha_BCE * test_losses[-2] + (1 - EMA_alpha_BCE) * EMA_test_BCE_losses[-1])
            EMA_train_zero_one_losses.append(0.5 * EMA_alpha_BCE * train_zero_one_losses[-1] + 0.5 * EMA_alpha_BCE * train_zero_one_losses[-2] + (1 - EMA_alpha_BCE) * EMA_train_zero_one_losses[-1])
            EMA_test_zero_one_losses.append(0.5 * EMA_alpha_BCE * test_zero_one_losses[-1] + 0.5 * EMA_alpha_BCE * test_zero_one_losses[-2] + (1 - EMA_alpha_BCE) * EMA_test_zero_one_losses[-1])
            epoch += 1

    while (EMA_train_losses[-1] - EMA_train_losses[-2] < eps or epoch <  min_epochs / len(train_loader)) and beta > 0.0:
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
                loss_for_optimization = criterion(outputs.squeeze(), batch_y)
                if using_bce_for_optimization:
                    loss_for_recording = transform_bce_to_unit_interval(loss_for_optimization, l_max)
                else:
                    loss_for_recording = loss_for_optimization
                predicted = (outputs.squeeze() > 0).float()
                zero_one_loss = zero_one_criterion(outputs, batch_y)
            else:
                loss_for_optimization = criterion(outputs.squeeze(), batch_y)
                if using_bce_for_optimization:
                    loss_for_recording = transform_bce_to_unit_interval(loss_for_optimization, l_max)
                else:
                    loss_for_recording = loss_for_optimization
                predicted = (outputs.squeeze() > 0).float()
                zero_one_loss = zero_one_criterion(outputs, batch_y)
            
            loss_for_optimization.backward()
            optimizer.step()
            
            # Record the loss (transformed if BCE was used for optimization, raw otherwise)
            bce_val = loss_for_recording.item()
            zeroOne_val = zero_one_loss.item()
            train_loss_total += loss_for_recording.item()
            train_zero_one_total += zero_one_loss.item()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            EMA_train_BCE_losses.append(EMA_alpha_BCE *  bce_val  + (1 - EMA_alpha_BCE) * EMA_train_BCE_losses[-1])
            EMA_train_zero_one_losses.append( EMA_alpha_BCE * zeroOne_val + (1 - EMA_alpha_BCE) * EMA_train_zero_one_losses[-1])
            EMA_train_losses.append( EMA_alpha * bce_val + (1 - EMA_alpha) * EMA_train_losses[-1])

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
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                outputs = model(batch_x)
                
                if dataset_type == 'synth':
                    loss_for_optimization = criterion(outputs.squeeze(), batch_y)
                    if using_bce_for_optimization:
                        loss_for_recording = transform_bce_to_unit_interval(loss_for_optimization, l_max)
                    else:
                        loss_for_recording = loss_for_optimization
                    predicted = (outputs.squeeze() > 0).float()
                    zero_one_loss = zero_one_criterion(outputs, batch_y)
                else:
                    loss_for_optimization = criterion(outputs.squeeze(), batch_y)
                    # if math.isnan(loss_for_optimization.item()):
                    #     raise ValueError("Loss is NaN")
                    if using_bce_for_optimization:
                        loss_for_recording = transform_bce_to_unit_interval(loss_for_optimization, l_max)
                    else:
                        loss_for_recording = loss_for_optimization
                    predicted = (outputs.squeeze() > 0).float()
                    zero_one_loss = zero_one_criterion(outputs, batch_y)
                
                bce_val = loss_for_recording.item()
                zeroOne_val = zero_one_loss.item()
                test_loss_total += loss_for_recording.item()
                test_zero_one_total += zero_one_loss.item()
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

                EMA_test_BCE_losses.append(EMA_alpha_BCE * bce_val + (1 - EMA_alpha_BCE) * EMA_test_BCE_losses[-1])
                EMA_test_zero_one_losses.append(EMA_alpha_BCE * zeroOne_val + (1 - EMA_alpha_BCE) * EMA_test_zero_one_losses[-1])
        
        avg_test_loss = test_loss_total / len(test_loader)
        avg_test_zero_one = test_zero_one_total / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        test_losses.append(avg_test_loss)
        test_zero_one_losses.append(avg_test_zero_one)
        test_accuracies.append(test_accuracy)
        
        # total_norm = 0.0
        # for p in model.parameters():
        #     param_norm = p.grad.detach().data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5

        # More frequent progress reporting with time estimates
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed_time = time.time() - start_time
            epochs_per_second = (epoch + 1) / elapsed_time if elapsed_time > 0 else 0

            
            print(f'Epoch [{epoch+1:>6}] '
                  f'Beta: {beta} '
                  f'EMA diff: {EMA_train_losses[-1] - EMA_train_losses[-2]:.6f} '
                  f'Train: {avg_train_loss:.4f} Test: {avg_test_loss:.4f} '
                  f'Train0-1: {avg_train_zero_one:.4f} Test0-1: {avg_test_zero_one:.4f} '
                  f'LR: {current_lr:.2e} '
                #   f'Speed: {epochs_per_second:.1f} ep/s '
                #   f'Norm of gradient: {total_norm:.4f} '
                  f'EMA Train BCE Loss: {EMA_train_BCE_losses[-1]:.4f} '
                  f'EMA Train Loss: {EMA_train_losses[-1]:.4f}'
                  )
            
            # GPU memory monitoring (if using CUDA)
            if device == 'cuda':
                print(f'GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.0f}MB allocated, '
                      f'{torch.cuda.memory_reserved(0) / 1024**2:.0f}MB reserved')
            
        
        epoch += 1
    print(f"Training completed in {time.time() - start_time:.1f} seconds over {epoch} epochs.")
    print(f"Final Training Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, EMA Diff: {EMA_train_losses[-1] - EMA_train_losses[-2]:.6f}, EMA Train Loss: {EMA_train_losses[-1]:.4f}")

    return (train_losses, test_losses, train_accuracies, test_accuracies,
            train_zero_one_losses, test_zero_one_losses, learning_rates, EMA_train_losses,
            EMA_train_BCE_losses, EMA_test_BCE_losses, EMA_train_zero_one_losses, EMA_test_zero_one_losses)

def run_beta_experiments(loss, beta_values, a0, b, sigma_gauss_prior, device,n_hidden_layers, width,
                         dataset_type, use_random_labels, l_max,  train_loader, test_loader,min_epochs,
                         alpha_average, alpha_stop, eta, eps, save_every=1):
    """
    Run SGLD experiments across multiple beta values for generalization bound computation.
    
    Trains neural networks with different inverse temperature (beta) values to compute
    Gibbs generalization bounds. Automatically includes beta=0 for prior sampling if
    not present. Saves results to CSV files with experimental metadata.
    
    Args:
        loss (str): Loss function name ('bce', 'tangent', or 'savage').
        beta_values (list): List of beta (inverse temperature) values to experiment with.
        a0 (float, dict, or callable): Initial learning rate specification.
        b (float): Learning rate decay exponent.
        sigma_gauss_prior (float): Standard deviation of Gaussian prior for weights.
        device (torch.device or str): Computation device ('cpu' or 'cuda').
        n_hidden_layers (int): Number of hidden layers in the network.
        width (int): Width (number of units) of hidden layers.
        dataset_type (str): Dataset type ('synth', 'mnist', or 'CIFAR10').
        use_random_labels (bool): Whether to use random labels instead of true labels.
        l_max (float): Maximum loss value for bounded transformation.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        min_epochs (int): Minimum number of epochs to train before checking convergence.
        alpha_average (float): EMA smoothing factor for loss averaging.
        alpha_stop (float): EMA smoothing factor for convergence detection.
        eta (float): Learning rate threshold parameter.
        eps (float): Convergence threshold for EMA training loss difference.
        save_every (int, optional): Save checkpoint frequency. Defaults to 1.
    
    Returns:
        None: Results are saved to CSV files with experimental metadata.
    
    Note:
        If beta=0 is not in beta_values, it will be automatically added for proper
        generalization bound computation through prior sampling.
    """

    
    # Ensure beta=0 is included for proper bound computation
    extended_beta_values = list(beta_values)
    if 0.0 not in extended_beta_values and 0 not in extended_beta_values:
        extended_beta_values = [0.0] + extended_beta_values
        print(f"Added beta=0 for proper generalization bound computation")    

    print(f"🆕 Starting new experiment")

    list_train_BCE_losses = []
    list_test_BCE_losses = []
    list_train_01_losses = []
    list_test_01_losses = []
    list_EMA_train_BCE_losses = []
    list_EMA_test_BCE_losses = []
    list_EMA_train_01_losses = []
    list_EMA_test_01_losses = []

    print(f"\nConfiguration:")    
    print(f"Learning rate (a0) per beta:")
    for beta in sorted(extended_beta_values):
        current_a0 = get_a0_for_beta(beta, a0)
        print(f"  Beta {beta}: {current_a0}")
    print(f"Save frequency: every {save_every} repetitions")
    

    print(f"{'='*80}")
    
    betas_experimented = []
    # Run all beta values for this repetition
    for beta in sorted(extended_beta_values):
        betas_experimented.append(beta)
        current_a0 = get_a0_for_beta(beta, a0)
        
        print(f"\n--- Beta = {beta} ---")
        print(f"Learning rate: {current_a0}")
        
        # Create fresh model for each beta-repetition combination
        if dataset_type == 'mnist':
            if n_hidden_layers == 1:
                model = FCN1L(input_dim=28*28, hidden_dim=width, output_dim=1)
            else:
                model = FCN2L(input_dim=28*28, hidden_dim=width, output_dim=1)

        elif dataset_type == 'cifar10':
            if n_hidden_layers == 1:
                model = FCN1L(input_dim=3*32*32, hidden_dim=width, output_dim=1)
            else:
                model = FCN2L(input_dim=3*32*32, hidden_dim=width, output_dim=1)

        # Train the model
        training_results = train_sgld_model(
            loss =loss,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            min_epochs=min_epochs,
            a0=current_a0,
            b=b,
            sigma_gauss_prior=sigma_gauss_prior,
            beta=beta,
            device=device,
            dataset_type=dataset_type,
            l_max=l_max,
            alpha_average=alpha_average,
            alpha_stop=alpha_stop,
            eta=eta,
            eps=eps
        )
        

        (train_losses, test_losses, _, _, train_01_losses, test_01_losses, _, EMA_train_losses,
                EMA_train_BCE_losses, EMA_test_BCE_losses, EMA_train_01_losses, EMA_test_01_losses) = training_results

        list_train_BCE_losses.append(train_losses[-50])
        list_test_BCE_losses.append(test_losses[-50])
        list_train_01_losses.append(train_01_losses[-50])
        list_test_01_losses.append(test_01_losses[-50])
        list_EMA_train_BCE_losses.append(EMA_train_BCE_losses[-1])
        list_EMA_test_BCE_losses.append(EMA_test_BCE_losses[-1])
        list_EMA_train_01_losses.append(EMA_train_01_losses[-1])
        list_EMA_test_01_losses.append(EMA_test_01_losses[-1])


        print(f"  Final - Train BCE: {train_losses[-1]:.4f}, Test BCE: {test_losses[-1]:.4f}, "
                f"Train 0-1: {train_01_losses[-1]:.4f}, Test 0-1: {test_01_losses[-1]:.4f}")
        

        # Save a csv file after each repetition if requested
        # Generate filename prefix based on experiment parameters
        filename_prefix = ""
        if dataset_type == 'mnist':
            filename_prefix = "M"
        elif dataset_type == 'CIFAR10':
            filename_prefix = "C"
        else:
            filename_prefix = "S"
        
        if use_random_labels:
            filename_prefix += "R"
        else:
            filename_prefix += "C"
        
        filename_prefix += f"L{n_hidden_layers}"
        filename_prefix += f"W{width}"
        if len(train_loader) == 1:
            filename_prefix += f"ULA"
        else:
            filename_prefix += f"SGLD"
        
        filename_prefix += f"{len(train_loader.dataset)/1000:.0f}k"
        filename_prefix += f"LR{current_a0}".replace('.', '')
        filename_prefix += f"{loss.upper()}"
        
        summary_string =  f"The LMC has been run with the following parameters:\n" \
        f"  - Device: {device}\n" \
        f"  - Loss function: {loss}\n" \
        f"  - l_max: {l_max}\n" \
        f"  - Network architecture: {model.__class__.__name__}\n" \
        f"  - Number of hidden layers: {n_hidden_layers}\n" \
        f"  - Width of hidden layers: {width}\n" \
        f"  - Dataset type: {dataset_type}\n" \
        f"  - Random labels: {use_random_labels}\n" \
        f"  - Training set size: {len(train_loader.dataset) if train_loader else 'N/A'}\n" \
        f"  - Test set size: {len(test_loader.dataset) if test_loader else 'N/A'}\n" \
        f"  - Minimum epochs: {min_epochs}\n" \
        f"  - Number of Batches: {len(train_loader) if train_loader else 'N/A'}\n" \
        f"  - Beta values: {sorted(betas_experimented)}\n" \
        f"  - Learning rate (a0): {current_a0}\n" \
        f"  - Learning rate decay (b): {b}\n" \
        f"  - Gaussian prior sigma: {sigma_gauss_prior}\n" \
        f" -  alpha_average: {alpha_average}\n" \
        f" -  alpha_stop: {alpha_stop}\n" \
        f" -  eta: {eta}\n" \
        f" -  eps: {eps}\n"

        csv_path = save_moving_average_losses_to_csv(
            list_train_BCE_losses,
            list_test_BCE_losses,
            list_train_01_losses,
            list_test_01_losses,
            list_EMA_train_BCE_losses,
            list_EMA_test_BCE_losses,   
            list_EMA_train_01_losses,
            list_EMA_test_01_losses,
            filename_prefix,
            sorted(betas_experimented),
            len(train_loader.dataset),
            summary_string
        )

    