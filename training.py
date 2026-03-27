"""
Training and experiment orchestration for the Gibbs generalization bound experiments.

This module contains functions for training models with SGLD and running
experiments across different beta values with multiple repetitions.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import csv
import random
from datetime import datetime
from losses import BoundedCrossEntropyLoss, ZeroOneLoss, TangentLoss, SavageLoss, PBBBoundedNLLLoss
from torch.nn import BCEWithLogitsLoss
from models import  initialize_nn_weights_gaussian, FCN1L, FCN2L, FCN3L, LeNet5, VGG16_CIFAR
from sgld import SGLD
from new_MALA import MALA, StepSizeTuner
from statistics import mean

# Optional PBB imports (for PBB model comparison)
try:
    from pbb_models import NNet4l, CNNet4l
    from pbb_prior import initialize_model_with_prior
    from pbb_truncated_prior import (
        initialize_prior_truncated_gaussian,
        build_layerwise_sigma_map,
        build_sgld_param_groups_from_sigma_map,
    )
    PBB_AVAILABLE = True
except ImportError:
    PBB_AVAILABLE = False

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


def build_sgld_optimizer(
    model,
    lr,
    sigma_gauss_prior,
    beta,
    add_noise=True,
    use_layerwise_prior_in_sgld=False,
    layerwise_prior_scale=1.0,
):
    """
    Build SGLD optimizer.

    Default behavior (existing experiments): single global sigma via sigma_gauss_prior.
    Optional behavior (PBB-style): layer-wise sigmas based on fan-in for per-parameter
    Gaussian prior regularization strength.
    """
    if use_layerwise_prior_in_sgld and PBB_AVAILABLE:
        sigma_map = build_layerwise_sigma_map(
            model,
            sigma_scale=layerwise_prior_scale,
            fallback_sigma=sigma_gauss_prior,
        )
        param_groups = build_sgld_param_groups_from_sigma_map(model, sigma_map, beta=beta)
        optimizer = SGLD(
            param_groups,
            lr=lr,
            sigma_gauss_prior=sigma_gauss_prior,
            beta=beta,
            add_noise=add_noise,
        )
        return optimizer

    return SGLD(
        model.parameters(),
        lr=lr,
        sigma_gauss_prior=sigma_gauss_prior,
        beta=beta,
        add_noise=add_noise,
    )

def train_sgld_model(loss, model, train_loader, test_loader, min_steps, 
                     a0, b, sigma_gauss_prior, 
                     beta, device, dataset_type,
                     l_max, alpha_average, alpha_stop,  eta, eps, add_noise=True,
                     prior_type='gaussian', use_layerwise_prior_in_sgld=False,
                     layerwise_prior_scale=1.0, max_epochs=None, pmin=None):
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
        min_steps (int): Minimum number of epochs to train before checking convergence.
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
        add_noise (bool): Whether to add Langevin noise during SGLD updates.
        max_epochs (int, optional): Maximum number of epochs (full passes through batch) to run.
                                   If None, uses convergence criterion only. If set, training stops
                                   when max_epochs is reached regardless of convergence.
    
    Returns:
        tuple: Contains (train_losses, test_losses, train_accuracies, test_accuracies,
               train_zero_one_losses, test_zero_one_losses, learning_rates, EMA_train_losses,
               EMA_train_BCE_losses, EMA_test_BCE_losses, EMA_train_zero_one_losses, 
               EMA_test_zero_one_losses, EMA_grad_norm).
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
    elif loss.lower() == 'nll':
        criterion = PBBBoundedNLLLoss(pmin=pmin) if pmin is not None else nn.NLLLoss()
    elif loss.lower() == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = BCEWithLogitsLoss()  
    zero_one_criterion = ZeroOneLoss()
    print(f"[training] criterion={criterion.__class__.__name__}")

    # Initialize SGLD optimizer with inverse temperature
    optimizer = build_sgld_optimizer(
        model=model,
        lr=a0,
        sigma_gauss_prior=sigma_gauss_prior,
        beta=beta,
        add_noise=add_noise,
        use_layerwise_prior_in_sgld=use_layerwise_prior_in_sgld,
        layerwise_prior_scale=layerwise_prior_scale,
    )
    # optimizer = MALA(model.parameters(), lr=a0, sigma_gauss_prior=sigma_gauss_prior, beta=beta)
    # Learning rate scheduler with threshold: lr_t = max(a0 * t^(-b), 0.01)
    # This stops the decay when learning rate reaches 0.01
    lr_threshold = eta / beta if beta != 0 else 0.01  # Avoid division by zero for beta=0
    def lr_lambda_with_threshold(epoch):
        power_law_lr = (epoch + 1) ** (-b)
        # Convert to actual learning rate value and check threshold
        actual_lr = a0 * power_law_lr
        if actual_lr <= lr_threshold:
            return lr_threshold / a0  # Return the ratio that gives us the threshold
        else:
            return power_law_lr  # Return the normal power law ratio
    
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_with_threshold)
    
    train_losses, test_losses, train_accuracies, test_accuracies, train_zero_one_losses, test_zero_one_losses, learning_rates = [], [], [], [], [], [], []
    EMA_train_losses = [0.0, 1.0]  # Initialize with 1.0 for epoch 0
    EMA_alpha = alpha_stop  # Smoothing factor for EMA of loss
    
    # EMA_alpha_BCE = alpha_average
    # EMA_train_BCE_losses = [1.0]  # Separate EMA for approximating the ergodic average
    # EMA_test_BCE_losses = [1.0]
    # EMA_train_zero_one_losses = [1.0]
    # EMA_test_zero_one_losses = [1.0]   
    # EMA_train_BCE_losses_sq = [1.0]  # For variance tracking if needed
    # EMA_test_BCE_losses_sq = [1.0]
    avg_train_BCE_losses, avg_test_BCE_losses, avg_train_zero_one_losses, avg_test_zero_one_losses = [], [], [], []
    avg_train_BCE_losses_sq, avg_test_BCE_losses_sq, avg_train_zero_one_losses_sq, avg_test_zero_one_losses_sq = [], [], [], []
    # EMA_grad_norm = [0.0]  # EMA for gradient norm if needed
    avg_grad_norm = []
    p_grad_norm = 2 # L-p norm for gradient norm tracking
    
    print(f"Training with SGLD: a0={a0}, b={b}, sigma_gauss_prior={sigma_gauss_prior}, beta={beta}")
    # print(f"Learning rate scheduler: power law decay with threshold at {lr_threshold}")
    print(f"Dataset type: {dataset_type}, Device: {device}")
    
    # Progress tracking
    start_time = time.time()
    epoch = 0
    # if beta == I have to sample from the prior many times and take the average both for train and test losses
    if beta == 0.0:
        num_prior_samples = 1000

        model_cpu = model.to('cpu')

        for i in range(num_prior_samples):
            learning_rates.append(0.0)  # No learning rate for beta=0 prior sampling
            optimizer.zero_grad(set_to_none=True)
            # Reinitialize model weights from prior
            if use_layerwise_prior_in_sgld and PBB_AVAILABLE and str(prior_type).lower() in ['truncated_gaussian', 'trunc_gaussian', 'truncnorm']:
                model_cpu = initialize_prior_truncated_gaussian(
                    model_cpu,
                    sigma_scale=layerwise_prior_scale,
                    truncation=2.0,
                    seed=42 + i * 1000,
                )
            else:
                model_cpu = initialize_nn_weights_gaussian(model_cpu, sigma=sigma_gauss_prior, seed=42+i*1000)
            # Use default initialization for prior sampling but with different seeds
            # torch.manual_seed(42+i*1000)
            # for layer in model_cpu.children():
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
            
            # Compute train loss
            # with torch.no_grad():
            train_loss_total = 0.0
            zero_one_loss_total = 0.0
            batch_x, batch_y = next(iter(train_loader))
            outputs = model_cpu(batch_x)
            loss_fn = criterion(outputs.squeeze(), batch_y)
            train_loss_total += loss_fn.item()

            loss_fn.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model_cpu.parameters(), float("inf"), norm_type=2).item()
            # EMA_grad_norm.append(grad_norm ** p_grad_norm)
            avg_grad_norm.append(grad_norm ** p_grad_norm)
            zero_one_loss_total += zero_one_criterion(outputs, batch_y).item()
            train_zero_one_losses.append(zero_one_loss_total )
            train_losses.append(train_loss_total)
            grad_dim = sum(p.grad.numel() for p in model_cpu.parameters() if p.grad is not None)

            # Compute test loss
            test_loss_total = 0.0
            zero_one_loss_total = 0.0
            with torch.no_grad():
                batch_x, batch_y = next(iter(test_loader))
                outputs = model_cpu(batch_x)
                loss_fn = criterion(outputs.squeeze(), batch_y)
                test_loss_total += loss_fn.item()
                zero_one_loss_total += zero_one_criterion(outputs, batch_y).item()
                test_losses.append(test_loss_total)
                test_zero_one_losses.append(zero_one_loss_total)

            # EMA_train_BCE_losses.append(EMA_alpha_BCE * train_losses[-1] + (1 - EMA_alpha_BCE) * EMA_train_BCE_losses[-1])
            # EMA_train_zero_one_losses.append(EMA_alpha_BCE * train_zero_one_losses[-1] + (1 - EMA_alpha_BCE) * EMA_train_zero_one_losses[-1])
            # EMA_test_BCE_losses.append(EMA_alpha_BCE * test_losses[-1]  + (1 - EMA_alpha_BCE) * EMA_test_BCE_losses[-1])
            # EMA_test_zero_one_losses.append(EMA_alpha_BCE * test_zero_one_losses[-1] + (1 - EMA_alpha_BCE) * EMA_test_zero_one_losses[-1])
            # EMA_train_BCE_losses_sq.append(EMA_alpha_BCE * (train_losses[-1]**2) + (1 - EMA_alpha_BCE) * EMA_train_BCE_losses_sq[-1])
            avg_train_BCE_losses.append(train_losses[-1])
            avg_train_zero_one_losses.append(train_zero_one_losses[-1])

            avg_test_BCE_losses.append(test_losses[-1])
            avg_test_zero_one_losses.append(test_zero_one_losses[-1])

            avg_train_BCE_losses_sq.append(train_losses[-1]**2)
            avg_test_BCE_losses_sq.append(test_losses[-1]**2)

            epoch += 1
            print(f'Beta=0.0: Averaged over {num_prior_samples} prior samples - '
                f'Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, '
                f'L-p grad norm: {(1/math.sqrt(grad_dim))*((sum(avg_grad_norm)/epoch)**(1/p_grad_norm)):.6f}, '
                f'grad norm: {grad_norm:.4f}, '
                f'EMA Train Loss: {mean(avg_train_BCE_losses):.4f}, EMA Test Loss: {mean(avg_test_BCE_losses):.4f}, '
                f'EMA Train Zero-One Loss: {mean(avg_train_zero_one_losses):.4f}, EMA Test Zero-One Loss: {mean(avg_test_zero_one_losses):.4f}'
                )

    while (EMA_train_losses[-1] - EMA_train_losses[-2] < eps or epoch <  min_steps / len(train_loader)) and beta > 0.0:
        # Check max_epochs limit if specified
        if max_epochs is not None and epoch >= max_epochs:
            print(f'   Reached maximum epochs limit: {max_epochs}. Stopping training.')
            break
    # while epoch < 20000 and beta > 0.0:  # Large max epoch, we will break based on EMA convergence
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
            
            # # ===== SGLD VERSION (commented out) =====
            # Use more efficient zero_grad
            optimizer.zero_grad(set_to_none=True)
            
            # Standard precision training
            outputs = model(batch_x)
             
            loss_fn = criterion(outputs.squeeze(), batch_y)
            predicted = (outputs.squeeze() > 0).float()
            zero_one_loss = zero_one_criterion(outputs, batch_y)
            
            loss_fn.backward()
            optimizer.step()
            # # ===== END SGLD VERSION =====
            
            # ===== MALA VERSION =====
            # Define closure for MALA (computes loss and gradients)
            # def closure():
            #     optimizer.zero_grad(set_to_none=True)
            #     outputs = model(batch_x)
            #     loss = criterion(outputs.squeeze(), batch_y)
            #     loss.backward()
            #     return loss.item()
            
            # # MALA step with closure
            # optimizer.step(closure)
            
            # # Compute metrics after the step (model is now at accepted/rejected state)
            # with torch.no_grad():
            #     outputs = model(batch_x)
            #     loss_fn = criterion(outputs.squeeze(), batch_y)
            #     predicted = (outputs.squeeze() > 0).float()
            #     zero_one_loss = zero_one_criterion(outputs, batch_y)
            # ===== END MALA VERSION =====
            
            # if add_grad_norm:
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), norm_type=2).item()
            # EMA_grad_norm.append(EMA_alpha_BCE * grad_norm + (1 - EMA_alpha_BCE) * EMA_grad_norm[-1])
            
            # TODO: Use the other approach to update EMA and check the difference
            try:
                EMA_train_losses.append( (EMA_alpha/2) * bce_val + (EMA_alpha/2) * loss_fn.item() + (1 - EMA_alpha) * EMA_train_losses[-1])
            except:
                EMA_train_losses.append( (EMA_alpha/2) * loss_fn.item() + (EMA_alpha/2) * loss_fn.item() + (1 - EMA_alpha) * EMA_train_losses[-1])
            bce_val = loss_fn.item()
            zeroOne_val = zero_one_loss.item()
            train_loss_total += bce_val
            train_zero_one_total += zeroOne_val
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            # EMA_train_BCE_losses.append(EMA_alpha_BCE *  bce_val  + (1 - EMA_alpha_BCE) * EMA_train_BCE_losses[-1])
            # EMA_train_zero_one_losses.append( EMA_alpha_BCE * zeroOne_val + (1 - EMA_alpha_BCE) * EMA_train_zero_one_losses[-1])
            # EMA_train_BCE_losses_sq.append(EMA_alpha_BCE * (bce_val**2) + (1 - EMA_alpha_BCE) * EMA_train_BCE_losses_sq[-1])

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

                loss_fn = criterion(outputs.squeeze(), batch_y)
                predicted = (outputs.squeeze() > 0).float()
                zero_one_loss = zero_one_criterion(outputs, batch_y)

                bce_val = loss_fn.item()
                zeroOne_val = zero_one_loss.item()
                test_loss_total += bce_val
                test_zero_one_total += zeroOne_val
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

                # EMA_test_BCE_losses.append(EMA_alpha_BCE * bce_val + (1 - EMA_alpha_BCE) * EMA_test_BCE_losses[-1])
                # EMA_test_zero_one_losses.append(EMA_alpha_BCE * zeroOne_val + (1 - EMA_alpha_BCE) * EMA_test_zero_one_losses[-1])
                # EMA_test_BCE_losses_sq.append(EMA_alpha_BCE * (bce_val**2) + (1 - EMA_alpha_BCE) * EMA_test_BCE_losses_sq[-1])

        avg_test_loss = test_loss_total / len(test_loader)
        avg_test_zero_one = test_zero_one_total / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        test_losses.append(avg_test_loss)
        test_zero_one_losses.append(avg_test_zero_one)
        test_accuracies.append(test_accuracy)


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
                #   f'Accept Rate: {optimizer.acceptance_rate:.2f} '
                #   f'Speed: {epochs_per_second:.1f} ep/s '
                #   f'Norm of gradient: {avg_grad_norm[-1]:.4f} '
                #   f'EMA Train BCE Loss: {mean(avg_train_BCE_losses):.4f} '
                  f'EMA Train Loss: {EMA_train_losses[-1]:.4f}'
                  )
            
            # GPU memory monitoring (if using CUDA)
            if device == 'cuda':
                print(f'GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.0f}MB allocated, '
                      f'{torch.cuda.memory_reserved(0) / 1024**2:.0f}MB reserved')
            
        epoch += 1
    print(f"First stage of training completed in {time.time() - start_time:.1f} seconds over {epoch} epochs.")
    if beta > 0.0:
        mixing_time_est = epoch
        epoch = 0 
        for i in range(mixing_time_est):
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
                
                # # ===== SGLD VERSION (commented out) =====
                # Use more efficient zero_grad
                optimizer.zero_grad(set_to_none=True)
                
                # Standard precision training
                outputs = model(batch_x)
                
                loss_fn = criterion(outputs.squeeze(), batch_y)
                predicted = (outputs.squeeze() > 0).float()
                zero_one_loss = zero_one_criterion(outputs, batch_y)
                
                loss_fn.backward()
                optimizer.step()
                # # ===== END SGLD VERSION =====
                
                # ===== MALA VERSION =====
                # Define closure for MALA (computes loss and gradients)
                # def closure():
                #     optimizer.zero_grad(set_to_none=True)
                #     outputs = model(batch_x)
                #     loss = criterion(outputs.squeeze(), batch_y)
                #     loss.backward()
                #     return loss.item()
                
                # # MALA step with closure
                # optimizer.step(closure)
                
                # # Compute metrics after the step (model is now at accepted/rejected state)
                # with torch.no_grad():
                #     outputs = model(batch_x)
                #     loss_fn = criterion(outputs.squeeze(), batch_y)
                #     predicted = (outputs.squeeze() > 0).float()
                #     zero_one_loss = zero_one_criterion(outputs, batch_y)
                # ===== END MALA VERSION =====
                
                # if add_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), norm_type=2).item()
                avg_grad_norm.append(grad_norm)

                bce_val = loss_fn.item()
                zeroOne_val = zero_one_loss.item()
                train_loss_total += bce_val
                train_zero_one_total += zeroOne_val
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

                avg_train_BCE_losses.append(bce_val)
                avg_train_zero_one_losses.append(zeroOne_val)
                avg_train_BCE_losses_sq.append(bce_val**2)

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

                    loss_fn = criterion(outputs.squeeze(), batch_y)
                    predicted = (outputs.squeeze() > 0).float()
                    zero_one_loss = zero_one_criterion(outputs, batch_y)

                    test_bce_val = loss_fn.item()
                    zeroOne_val = zero_one_loss.item()
                    test_loss_total += test_bce_val
                    test_zero_one_total += zeroOne_val
                    test_total += batch_y.size(0)
                    test_correct += (predicted == batch_y).sum().item()

                    avg_test_BCE_losses.append(test_bce_val)
                    avg_test_zero_one_losses.append(zeroOne_val)
                    avg_test_BCE_losses_sq.append(test_bce_val**2)

            avg_test_loss = test_loss_total / len(test_loader)
            avg_test_zero_one = test_zero_one_total / len(test_loader)
            test_accuracy = 100 * test_correct / test_total
            test_losses.append(avg_test_loss)
            test_zero_one_losses.append(avg_test_zero_one)
            test_accuracies.append(test_accuracy)


            # More frequent progress reporting with time estimates
            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed_time = time.time() - start_time
                epochs_per_second = (epoch + 1) / elapsed_time if elapsed_time > 0 else 0

                
                print(f'Epoch [{epoch+1:>6}] '
                    f'Beta: {beta} '
                    f'Train: {avg_train_loss:.4f} Test: {avg_test_loss:.4f} '
                    f'Train0-1: {avg_train_zero_one:.4f} Test0-1: {avg_test_zero_one:.4f} '
                    f'LR: {current_lr:.2e} '
                    # f'Accept Rate: {optimizer.acceptance_rate:.2f} '
                    #   f'Speed: {epochs_per_second:.1f} ep/s '
                    f'Norm of gradient: {avg_grad_norm[-1]:.4f} '
                    f'EMA Train BCE Loss: {mean(avg_train_BCE_losses):.4f} '
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
            avg_train_BCE_losses, avg_test_BCE_losses, avg_train_zero_one_losses, avg_test_zero_one_losses,
            avg_train_BCE_losses_sq, avg_test_BCE_losses_sq, avg_grad_norm, epoch)

def train_annealed_sgld_model(loss, model, train_loader, test_loader, min_steps, 
                     a0, b, sigma_gauss_prior, 
                     beta_values, device, dataset_type,
                     l_max, alpha_average, alpha_stop,  eta, eps, add_noise=True,
                     min_steps_first_beta=None, prior_type='gaussian',
                     use_layerwise_prior_in_sgld=False, layerwise_prior_scale=1.0,
                     pmin=None):
    """
    Train a neural network using Annealed Stochastic Gradient Langevin Dynamics (SGLD).
    
    Implements annealed SGLD training where the model is trained sequentially with 
    increasing beta values (inverse temperature). After convergence at one beta,
    the training continues with the next beta value, preserving the model state
    and EMA filters. For beta=0, samples from the prior distribution instead of training.
    
    Args:
        loss (str): Loss function name ('bce', 'tangent', or 'savage').
        model (nn.Module): Neural network model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        min_steps (int): Minimum number of epochs to train before checking convergence for subsequent betas.
        a0 (float): Initial learning rate.
        b (float): Learning rate decay exponent (lr_t = a0 * t^(-b)).
        sigma_gauss_prior (float): Standard deviation of Gaussian prior for weights.
        beta_values (list): List of inverse temperature values to anneal through.
        device (torch.device or str): Device for computation ('cpu' or 'cuda').
        dataset_type (str): Dataset type ('synth' or 'mnist') for loss computation.
        l_max (float): Maximum loss value for bounded transformation.
        alpha_average (float): EMA smoothing factor for loss averaging.
        alpha_stop (float): EMA smoothing factor for convergence detection.
        eta (float): Learning rate threshold parameter (lr >= eta/beta).
        eps (float): Convergence threshold for EMA training loss difference.
        add_noise (bool): Whether to add Langevin noise during SGLD updates.
        min_steps_first_beta (int, optional): Minimum steps for the first beta (> 0). 
            If None, uses min_steps for all betas. Typically should be larger than min_steps.
    
    Returns:
        tuple: Contains lists of results for each beta value - 
               (list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, 
               list_test_01_losses, list_EMA_train_BCE_losses, list_EMA_test_BCE_losses,   
               list_EMA_train_01_losses, list_EMA_test_01_losses, list_EMA_grad_norm).
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
    elif loss.lower() == 'nll':
        criterion = PBBBoundedNLLLoss(pmin=pmin) if pmin is not None else nn.NLLLoss()
    elif loss.lower() == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = BCEWithLogitsLoss()  
    zero_one_criterion = ZeroOneLoss()
    print(f"[training] criterion={criterion.__class__.__name__}")

    # Lists to store final results for each beta
    list_train_BCE_losses = []
    list_test_BCE_losses = []
    list_train_01_losses = []
    list_test_01_losses = []
    list_EMA_train_BCE_losses = []
    list_EMA_test_BCE_losses = []
    list_EMA_train_01_losses = []
    list_EMA_test_01_losses = []
    list_EMA_grad_norm = []
    list_num_epochs_per_beta = []

    # EMA variables that persist across beta values
    EMA_train_losses = [0.0, 1.0]  # For convergence detection
    # EMA_train_BCE_losses = [1.0]  # For ergodic average
    # EMA_test_BCE_losses = [1.0]
    # EMA_train_zero_one_losses = [1.0]
    # EMA_test_zero_one_losses = [1.0]    
    # EMA_grad_norm = [0.0]
    avg_train_BCE_losses, avg_test_BCE_losses, avg_train_zero_one_losses, avg_test_zero_one_losses = [], [], [], []
    avg_train_BCE_losses_sq, avg_test_BCE_losses_sq, avg_train_zero_one_losses_sq, avg_test_zero_one_losses_sq = [], [], [], []
    avg_grad_norm = []
    
    p_grad_norm = 2  # L-p norm for gradient norm tracking
    EMA_alpha = alpha_stop  # Smoothing factor for EMA of loss (for convergence)
    # EMA_alpha_BCE = alpha_average  # Smoothing factor for EMA of loss (for averaging)
    
    # Initialize optimizer (will be updated for each beta)
    optimizer = None
    
    # Set minimum steps for first beta vs subsequent betas
    if min_steps_first_beta is None:
        min_steps_first_beta = min_steps
    
    print(f"\n{'='*80}")
    print(f"Starting Annealed SGLD with {len(beta_values)} beta values: {sorted(beta_values)}")
    print(f"Model state and EMA filters will be preserved between beta transitions")
    print(f"Minimum steps for first beta (>0): {min_steps_first_beta}")
    print(f"Minimum steps for subsequent betas: {min_steps}")
    print(f"{'='*80}\n")
    
    # Track if we've trained the first beta > 0 yet
    first_nonzero_beta_trained = False

    for beta_idx, beta in enumerate(sorted(beta_values)):
        print(f"\n--- Beta {beta_idx + 1}/{len(beta_values)}: β = {beta} ---")

        optimizer = build_sgld_optimizer(
            model=model,
            lr=a0,
            sigma_gauss_prior=sigma_gauss_prior,
            beta=beta,
            add_noise=add_noise,
            use_layerwise_prior_in_sgld=use_layerwise_prior_in_sgld,
            layerwise_prior_scale=layerwise_prior_scale,
        )
        
        # Tracking for this beta
        local_epoch = 0
        start_time = time.time()
        
        # Special handling for beta=0 (prior sampling)
        if beta == 0.0:
            print("Beta=0: Sampling from prior distribution")
            num_prior_samples = 1000
            model_cpu = model.to('cpu')
            
            temp_train_losses = []
            temp_test_losses = []
            temp_train_01 = []
            temp_test_01 = []

            for i in range(num_prior_samples):
                optimizer.zero_grad(set_to_none=True)
                # Reinitialize model weights from prior
                if use_layerwise_prior_in_sgld and PBB_AVAILABLE and str(prior_type).lower() in ['truncated_gaussian', 'trunc_gaussian', 'truncnorm']:
                    model_cpu = initialize_prior_truncated_gaussian(
                        model_cpu,
                        sigma_scale=layerwise_prior_scale,
                        truncation=2.0,
                        seed=42 + i * 1000,
                    )
                else:
                    model_cpu = initialize_nn_weights_gaussian(model_cpu, sigma=sigma_gauss_prior, seed=42+i*1000)
                
                # Compute train loss
                batch_x, batch_y = next(iter(train_loader))
                outputs = model_cpu(batch_x)
                loss_fn = criterion(outputs.squeeze(), batch_y)
                zero_one_loss = zero_one_criterion(outputs, batch_y)
                
                temp_train_losses.append(loss_fn.item())
                temp_train_01.append(zero_one_loss.item())
                
                # Update averages
                # EMA_train_BCE_losses.append(EMA_alpha_BCE * temp_train_losses[-1] + (1 - EMA_alpha_BCE) * EMA_train_BCE_losses[-1])
                # EMA_train_zero_one_losses.append(EMA_alpha_BCE * temp_train_01[-1] + (1 - EMA_alpha_BCE) * EMA_train_zero_one_losses[-1])
                avg_train_BCE_losses.append(temp_train_losses[-1])
                avg_train_zero_one_losses.append(temp_train_01[-1])

                # Compute test loss
                with torch.no_grad():
                    batch_x, batch_y = next(iter(test_loader))
                    outputs = model_cpu(batch_x)
                    loss_fn = criterion(outputs.squeeze(), batch_y)
                    zero_one_loss = zero_one_criterion(outputs, batch_y)
                    
                    temp_test_losses.append(loss_fn.item())
                    temp_test_01.append(zero_one_loss.item())
                    
                    # EMA_test_BCE_losses.append(EMA_alpha_BCE * temp_test_losses[-1] + (1 - EMA_alpha_BCE) * EMA_test_BCE_losses[-1])
                    # EMA_test_zero_one_losses.append(EMA_alpha_BCE * temp_test_01[-1] + (1 - EMA_alpha_BCE) * EMA_test_zero_one_losses[-1])
                    avg_test_BCE_losses.append(temp_test_losses[-1])
                    avg_test_zero_one_losses.append(temp_test_01[-1])
            
            print(f'Beta=0.0: Sampled {num_prior_samples} times from prior - '
                  f'Avg Train BCE: {mean(avg_train_BCE_losses):.4f}, Avg Test BCE: {mean(avg_test_BCE_losses):.4f}, '
                  f'Avg Train 0-1: {mean(avg_train_zero_one_losses):.4f}, Avg Test 0-1: {mean(avg_test_zero_one_losses):.4f}')
            
            # Move model back to device
            model = model.to(device)
        
        # Training for beta > 0
        else:
            # Determine which min_steps to use
            if not first_nonzero_beta_trained:
                current_min_steps = min_steps_first_beta
                first_nonzero_beta_trained = True
                EMA_train_losses = [0.0, 1.0]  # For convergence detection
                # EMA_train_BCE_losses = [1.0]  # For ergodic average
                # EMA_test_BCE_losses = [1.0]
                # EMA_train_zero_one_losses = [1.0]
                # EMA_test_zero_one_losses = [1.0]    
                # EMA_grad_norm = [0.0]
                avg_train_BCE_losses = []
                avg_test_BCE_losses = []
                avg_train_zero_one_losses = []
                avg_test_zero_one_losses = []
                avg_grad_norm = []
                # Re initialize model from pytorch default initialization
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                
                print(f"Training with beta={beta}, min_steps={current_min_steps} (FIRST beta>0), a0={a0}")
            else:
                current_min_steps = min_steps
                print(f"Training with beta={beta}, min_steps={current_min_steps}, a0={a0}")
                print(f"Continuing from previous beta (model state and EMAs preserved)")
                EMA_train_losses = [0.0, bce_val]  # For convergence detection
                # EMA_train_BCE_losses = [bce_val]  # For ergodic average
                # EMA_test_BCE_losses = [test_bce_val]
                # EMA_train_zero_one_losses = [zero_one_val]
                # EMA_test_zero_one_losses = [test_zero_one_val]
                # EMA_grad_norm = [0.0]
                avg_train_BCE_losses = []
                avg_test_BCE_losses = []
                avg_train_zero_one_losses = []
                avg_test_zero_one_losses = []
                avg_grad_norm = []
                # EMA_train_losses[-1] = EMA_train_losses[-1] * 1.5  # Slightly increase to avoid false convergence
            # Train until convergence
            while (EMA_train_losses[-1] - EMA_train_losses[-2] < eps or 
                   local_epoch < current_min_steps / len(train_loader)):
                # Training phase
                model.train()
                train_loss_total = 0.0
                train_zero_one_total = 0.0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(batch_x)
                    
                    loss_fn = criterion(outputs.squeeze(), batch_y)
                    zero_one_loss = zero_one_criterion(outputs, batch_y)
                    
                    loss_fn.backward()
                    optimizer.step()
                    
                    # Update averages
                    bce_val = loss_fn.item()
                    zero_one_val = zero_one_loss.item()
                    train_loss_total += bce_val
                    train_zero_one_total += zero_one_val
                    
                    # EMA_train_BCE_losses.append(EMA_alpha_BCE * bce_val + (1 - EMA_alpha_BCE) * EMA_train_BCE_losses[-1])
                    # EMA_train_zero_one_losses.append(EMA_alpha_BCE * zero_one_val + (1 - EMA_alpha_BCE) * EMA_train_zero_one_losses[-1])
                    avg_train_BCE_losses.append(bce_val)
                    avg_train_zero_one_losses.append(zero_one_val)
                    EMA_train_losses.append(EMA_alpha * bce_val + (1 - EMA_alpha) * EMA_train_losses[-1])
                
                avg_train_loss = train_loss_total / len(train_loader)
                avg_train_zero_one = train_zero_one_total / len(train_loader)
                
                # Test/Evaluation phase
                model.eval()
                test_loss_total = 0.0
                test_zero_one_total = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x = batch_x.to(device, non_blocking=True)
                        batch_y = batch_y.to(device, non_blocking=True)
                        
                        outputs = model(batch_x)
                        loss_fn = criterion(outputs.squeeze(), batch_y)
                        zero_one_loss = zero_one_criterion(outputs, batch_y)
                        
                        test_bce_val = loss_fn.item()
                        test_zero_one_val = zero_one_loss.item()
                        test_loss_total += test_bce_val
                        test_zero_one_total += test_zero_one_val

                        # EMA_test_BCE_losses.append(EMA_alpha_BCE * test_bce_val + (1 - EMA_alpha_BCE) * EMA_test_BCE_losses[-1])
                        # EMA_test_zero_one_losses.append(EMA_alpha_BCE * test_zero_one_val + (1 - EMA_alpha_BCE) * EMA_test_zero_one_losses[-1])
                        avg_test_BCE_losses.append(test_bce_val)
                        avg_test_zero_one_losses.append(test_zero_one_val)
                
                avg_test_loss = test_loss_total / len(test_loader)
                avg_test_zero_one = test_zero_one_total / len(test_loader)
                
                # Progress reporting
                if (local_epoch + 1) % 10 == 0 or local_epoch == 0:
                    ema_diff = EMA_train_losses[-1] - EMA_train_losses[-2]
                    print(f'Epoch [{local_epoch+1:>6}] Beta: {beta} '
                          f'EMA diff: {ema_diff:.6f} '
                          f'Train BCE: {avg_train_loss:.4f} Test BCE: {avg_test_loss:.4f} '
                          f'Train 0-1: {avg_train_zero_one:.4f} Test 0-1: {avg_test_zero_one:.4f} '
                          f'Avg Train BCE: {mean(avg_train_BCE_losses):.4f}')
                
                local_epoch += 1
            
            elapsed_time = time.time() - start_time
            print(f"Beta={beta} training completed in {elapsed_time:.1f}s over {local_epoch} epochs")
            print(f"Final - Avg Train BCE: {mean(avg_train_BCE_losses):.4f}, Avg Test BCE: {mean(avg_test_BCE_losses):.4f}")
        
        # Store results for this beta
        list_train_BCE_losses.append(mean(avg_train_BCE_losses))
        list_test_BCE_losses.append(mean(avg_test_BCE_losses))
        list_train_01_losses.append(mean(avg_train_zero_one_losses))
        list_test_01_losses.append(mean(avg_test_zero_one_losses))
        list_EMA_train_BCE_losses.append(mean(avg_train_BCE_losses))
        list_EMA_test_BCE_losses.append(mean(avg_test_BCE_losses))
        list_EMA_train_01_losses.append(mean(avg_train_zero_one_losses))
        list_EMA_test_01_losses.append(mean(avg_test_zero_one_losses))
        list_EMA_grad_norm.append(mean(avg_grad_norm) if len(avg_grad_norm) > 0 else 0.0)
        list_num_epochs_per_beta.append(local_epoch)

    print(f"\n{'='*80}")
    print(f"Annealed SGLD completed for all {len(beta_values)} beta values")
    print(f"{'='*80}\n")
    
    # Return results in format compatible with non-annealing case
    return (list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,
            list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, 
            list_EMA_test_01_losses, list_EMA_grad_norm, list_num_epochs_per_beta)


import numpy as np

def compute_ess(series):
    """
    Estimate Effective Sample Size (ESS) for a 1D list/array.
    Uses a simplified autocorrelation sum method.
    """
    n = len(series)
    if n < 2: return 0
    
    series = np.array(series)
    
    # 1. Calculate Autocorrelation at various lags
    # (Using FFT for speed, but manual loop is fine for small thinned lists)
    mean = np.mean(series)
    var = np.var(series)
    if var == 0: return n # No variance means constant signal
    
    # Centered data
    centered = series - mean
    
    # Compute auto-covariance
    # This computes correlation for all possible lags
    full_corr = np.correlate(centered, centered, mode='full')
    
    # Normalize to get autocorrelation (rho)
    # We only care about the second half (positive lags)
    # rho[0] is always 1.0
    rho = full_corr[n-1:] / (var * n)
    
    # 2. Sum Rho until it becomes negative (standard Geyer's truncation)
    # We sum 2*rho because correlation is symmetric (lag -k == lag +k)
    sum_rho = 0
    for t in range(1, n):
        if rho[t] < 0.05: # Stop summing when correlation drops to noise level
            break
        sum_rho += rho[t]
        
    # 3. Calculate ESS
    # Formula: N / (1 + 2 * Sum_of_Correlations)
    ess = n / (1 + 2 * sum_rho)
    
    return max(1, ess)

# ---------------------------------------------------------

def check_stopping_criterion(thinned_loss_samples, rse_threshold=0.01):
    """
    Returns True if we should stop.
    """
    N = len(thinned_loss_samples)
    if N < 50: return False # Collect minimum baseline
    elif N % 10 != 0: return False # Check only every 10 samples
    
    # A. Calculate Statistics
    mu = np.mean(thinned_loss_samples)
    sigma = np.std(thinned_loss_samples, ddof=1)
    
    # B. Compute ESS on the thinned list
    ess = compute_ess(thinned_loss_samples)
    
    # C. Compute MCSE using ESS (The Corrected Formula)
    mcse = sigma / np.sqrt(ess)
    
    # D. Relative Standard Error
    rse = mcse / np.abs(mu)
    
    print(f"Samples: {N} | ESS: {ess:.1f} | RSE: {rse:.4f}")
    
    return rse < rse_threshold

def train_annealed_mala_model(loss, model, train_loader, test_loader, min_steps, 
                     a0, b, sigma_gauss_prior, 
                     beta_values, device, dataset_type,
                     l_max, alpha_average, alpha_stop,  eta, eps, pmin=None):
    """
    Train a neural network using Annealed MALA (Metropolis-Adjusted Langevin Algorithm).
    
    Implements annealed MALA training where the model is trained sequentially with 
    increasing beta values (inverse temperature). MALA requires full-batch training
    (batch_size == dataset_size) since it uses Metropolis-Hastings acceptance/rejection.
    For beta=0, samples from the prior distribution instead of training.
    
    Args:
        loss (str): Loss function name ('bce', 'tangent', or 'savage').
        model (nn.Module): Neural network model to train.
        train_loader (DataLoader): Training data loader (must have batch_size == dataset_size).
        test_loader (DataLoader): Test data loader (can have any batch size).
        min_steps (int): Minimum number of steps for warm-up phase to adapt step size.
        a0 (float): Initial learning rate (step size).
        b (float): Learning rate decay exponent (unused in MALA, kept for API compatibility).
        sigma_gauss_prior (float): Standard deviation of Gaussian prior for weights.
        beta_values (list): List of inverse temperature values to anneal through.
        device (torch.device or str): Device for computation ('cpu' or 'cuda').
        dataset_type (str): Dataset type ('synth' or 'mnist') for loss computation.
        l_max (float): Maximum loss value for bounded transformation.
        alpha_average (float): EMA smoothing factor for loss averaging.
        alpha_stop (float): EMA smoothing factor for convergence detection.
        eta (float): Learning rate threshold parameter (unused in MALA).
        eps (float): Convergence threshold (unused in MALA, uses RSE instead).
    
    Returns:
        tuple: Contains lists of results for each beta value - 
               (list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, 
               list_test_01_losses, list_EMA_train_BCE_losses, list_EMA_test_BCE_losses,   
               list_EMA_train_01_losses, list_EMA_test_01_losses, list_EMA_grad_norm).
    
    Raises:
        ValueError: If train_loader or test_loader have more than one batch.
    """
    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)
    
    # MALA requires full-batch training (batch_size == dataset_size)
    if len(train_loader) != 1:
        raise ValueError(
            f"MALA requires full-batch training (batch_size == dataset_size). "
            f"Got {len(train_loader)} batches. Please set batch_size equal to the training set size."
        )
    
    # Get the full training batch (since we verified there's only one batch)
    train_x, train_y = next(iter(train_loader))
    
    model = model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    
    # Define loss function
    if loss.lower() == 'bbce':
        criterion = BoundedCrossEntropyLoss(ell_max=l_max)
    elif loss.lower() == 'tangent':
        criterion = TangentLoss()
    elif loss.lower() == 'savage':
        criterion = SavageLoss()
    elif loss.lower() == 'nll':
        criterion = PBBBoundedNLLLoss(pmin=pmin) if pmin is not None else nn.NLLLoss()
    elif loss.lower() == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = BCEWithLogitsLoss()  
    zero_one_criterion = ZeroOneLoss()
    print(f"[training] criterion={criterion.__class__.__name__}")

    # Lists to store final results for each beta
    list_train_BCE_losses = []
    list_test_BCE_losses = []
    list_train_01_losses = []
    list_test_01_losses = []
    list_EMA_train_BCE_losses = []
    list_EMA_test_BCE_losses = []
    list_EMA_train_01_losses = []
    list_EMA_test_01_losses = []
    list_EMA_grad_norm = []
    list_num_epochs_per_beta = []

    # EMA variables that persist across beta values
    # EMA_train_losses = [0.0, 1.0]  # For convergence detection
    # EMA_train_BCE_losses = [1.0]  # For ergodic average
    # EMA_test_BCE_losses = [1.0]
    # EMA_train_zero_one_losses = [1.0]
    # EMA_test_zero_one_losses = [1.0]    
    # EMA_grad_norm = [0.0]
    avg_train_BCE_losses, avg_test_BCE_losses, avg_train_zero_one_losses, avg_test_zero_one_losses = [], [], [], []
    avg_train_BCE_losses_sq, avg_test_BCE_losses_sq, avg_train_zero_one_losses_sq, avg_test_zero_one_losses_sq = [], [], [], []
    avg_grad_norm = []
    
    p_grad_norm = 2  # L-p norm for gradient norm tracking
    EMA_alpha = alpha_stop  # Smoothing factor for EMA of loss (for convergence)
    # EMA_alpha_BCE = alpha_average  # Smoothing factor for EMA of loss (for averaging)
    
    # Initialize optimizer (will be updated for each beta)
    optimizer = None
    
    print(f"\n{'='*80}")
    print(f"Starting Annealed MALA with {len(beta_values)} beta values: {sorted(beta_values)}")
    print(f"Model state and EMA filters will be preserved between beta transitions")
    print(f"Minimum steps for subsequent betas: {min_steps}")
    print(f"{'='*80}\n")
    
    # Track if we've trained the first beta > 0 yet
    first_nonzero_beta_trained = False

    for beta_idx, beta in enumerate(sorted(beta_values)):
        print(f"\n--- Beta {beta_idx + 1}/{len(beta_values)}: β = {beta} ---")
        a0 = get_a0_for_beta(beta, a0)
        optimizer = MALA(model.parameters(), lr=a0, sigma_gauss_prior=sigma_gauss_prior, beta=beta)
        
        # Tracking for this beta
        local_epoch = 0
        start_time = time.time()
        
        # Special handling for beta=0 (prior sampling)
        if beta == 0.0:
            print("Beta=0: Sampling from prior distribution")
            num_prior_samples = 1000
            model_cpu = model.to('cpu')
            
            temp_train_losses = []
            temp_test_losses = []
            temp_train_01 = []
            temp_test_01 = []

            for i in range(num_prior_samples):
                optimizer.zero_grad(set_to_none=True)
                # Reinitialize model weights from prior
                model_cpu = initialize_nn_weights_gaussian(model_cpu, sigma=sigma_gauss_prior, seed=42+i*1000)
                
                # Compute train loss
                batch_x, batch_y = next(iter(train_loader))
                outputs = model_cpu(batch_x)
                loss_fn = criterion(outputs.squeeze(), batch_y)
                zero_one_loss = zero_one_criterion(outputs, batch_y)
                
                temp_train_losses.append(loss_fn.item())
                temp_train_01.append(zero_one_loss.item())
                
                # Update averages
                # EMA_train_BCE_losses.append(EMA_alpha_BCE * temp_train_losses[-1] + (1 - EMA_alpha_BCE) * EMA_train_BCE_losses[-1])
                # EMA_train_zero_one_losses.append(EMA_alpha_BCE * temp_train_01[-1] + (1 - EMA_alpha_BCE) * EMA_train_zero_one_losses[-1])
                avg_train_BCE_losses.append(temp_train_losses[-1])
                avg_train_zero_one_losses.append(temp_train_01[-1])

                # Compute test loss
                with torch.no_grad():
                    batch_x, batch_y = next(iter(test_loader))
                    outputs = model_cpu(batch_x)
                    loss_fn = criterion(outputs.squeeze(), batch_y)
                    zero_one_loss = zero_one_criterion(outputs, batch_y)
                    
                    temp_test_losses.append(loss_fn.item())
                    temp_test_01.append(zero_one_loss.item())
                    
                    # EMA_test_BCE_losses.append(EMA_alpha_BCE * temp_test_losses[-1] + (1 - EMA_alpha_BCE) * EMA_test_BCE_losses[-1])
                    # EMA_test_zero_one_losses.append(EMA_alpha_BCE * temp_test_01[-1] + (1 - EMA_alpha_BCE) * EMA_test_zero_one_losses[-1])
                    avg_test_BCE_losses.append(temp_test_losses[-1])
                    avg_test_zero_one_losses.append(temp_test_01[-1])
            
            print(f'Beta=0.0: Sampled {num_prior_samples} times from prior - '
                  f'Avg Train BCE: {mean(avg_train_BCE_losses):.4f}, Avg Test BCE: {mean(avg_test_BCE_losses):.4f}, '
                  f'Avg Train 0-1: {mean(avg_train_zero_one_losses):.4f}, Avg Test 0-1: {mean(avg_test_zero_one_losses):.4f}')
            
            # Move model back to device
            model = model.to(device)
        
        # Training for beta > 0
        else:
            thinning_interval = 100
            tuner = StepSizeTuner(initial_step_size=a0, target_accept=0.57)
            # EMA_train_BCE_losses = [1.0]  # For ergodic average
            # EMA_test_BCE_losses = [1.0]
            # EMA_train_zero_one_losses = [1.0]
            # EMA_test_zero_one_losses = [1.0]    
            # EMA_grad_norm = [0.0]
            avg_train_BCE_losses = []
            avg_test_BCE_losses = []
            avg_train_zero_one_losses = []
            avg_test_zero_one_losses = []
            avg_grad_norm = []

            if not first_nonzero_beta_trained:
                first_nonzero_beta_trained = True
                # Re initialize model from pytorch default initialization
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                
                print(f"Training with beta={beta}, min_steps={min_steps} (FIRST beta>0), a0={a0}")

            else:
                print(f"Training with beta={beta}, min_steps={min_steps}, a0={a0}")
                print(f"Continuing from previous beta (model state and EMAs preserved)")

            # Warm-up phase to adapt step size (no batch loop needed - full batch)
            for i in range(min_steps):
                model.train()
                
                # Define closure for MALA (computes loss and gradients)
                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(train_x)
                    loss = criterion(outputs.squeeze(), train_y)
                    loss.backward()
                    return loss.item()
                
                optimizer.step(closure)
                # Update Step Size
                new_step_size = tuner.update(optimizer.alpha)
                optimizer.set_step_size(new_step_size)
                if i % 10 == 0:
                    print(f"Warm-up Step {i+1}/{min_steps}, Adjusted Step Size: {new_step_size:.6f}, Acceptance Rate: {optimizer.acceptance_rate:.4f}, alpha: {optimizer.alpha:.4f}")
            final_step_size = tuner.step_size
            print(f"Optimal Step Size found: {final_step_size}")  

            # Sampling phase until convergence (no batch loop needed - full batch)
            while not check_stopping_criterion(avg_train_BCE_losses, rse_threshold=0.01):
                model.train()
                
                # Define closure for MALA (computes loss and gradients)
                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(train_x)
                    loss = criterion(outputs.squeeze(), train_y)
                    loss.backward()
                    return loss.item()
                
                optimizer.step(closure)
                
                # Compute metrics after the step (model is now at accepted state)
                with torch.no_grad():
                    outputs = model(train_x)
                    loss_fn = criterion(outputs.squeeze(), train_y)
                    zero_one_loss = zero_one_criterion(outputs, train_y)
                    bce_val = loss_fn.item()
                    zero_one_val = zero_one_loss.item()

                if (local_epoch + 1) % thinning_interval == 0 or local_epoch == 0:
                    # Update averages (thinned samples)
                    avg_train_BCE_losses.append(bce_val)
                    avg_train_zero_one_losses.append(zero_one_val)
                
                # Test/Evaluation phase (loop over test batches)
                model.eval()
                test_loss_total = 0.0
                test_zero_one_total = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x = batch_x.to(device, non_blocking=True)
                        batch_y = batch_y.to(device, non_blocking=True)
                        
                        outputs = model(batch_x)
                        loss_fn = criterion(outputs.squeeze(), batch_y)
                        zero_one_loss = zero_one_criterion(outputs, batch_y)
                        
                        test_loss_total += loss_fn.item()
                        test_zero_one_total += zero_one_loss.item()
                
                test_bce_val = test_loss_total / len(test_loader)
                test_zero_one_val = test_zero_one_total / len(test_loader)

                if local_epoch % thinning_interval == 0 or local_epoch == 0:
                    avg_test_BCE_losses.append(test_bce_val)
                    avg_test_zero_one_losses.append(test_zero_one_val)
                
                # Progress reporting
                if (local_epoch + 1) % thinning_interval == 0 or local_epoch == 0:
                    print(f'Epoch [{local_epoch+1:>6}] Beta: {beta} '
                          f'Train BCE: {bce_val:.4f} Test BCE: {test_bce_val:.4f} '
                          f'Train 0-1: {zero_one_val:.4f} Test 0-1: {test_zero_one_val:.4f} '
                          f'Avg Train BCE: {mean(avg_train_BCE_losses):.4f}'
                          f'Acceptance rate: {optimizer.acceptance_rate}')
                
                local_epoch += 1
            
            elapsed_time = time.time() - start_time
            print(f"Beta={beta} training completed in {elapsed_time:.1f}s over {local_epoch} epochs")
            print(f"Final - Avg Train BCE: {mean(avg_train_BCE_losses):.4f}, Avg Test BCE: {mean(avg_test_BCE_losses):.4f}")
        
        # Store results for this beta
        list_train_BCE_losses.append(mean(avg_train_BCE_losses))
        list_test_BCE_losses.append(mean(avg_test_BCE_losses))
        list_train_01_losses.append(mean(avg_train_zero_one_losses))
        list_test_01_losses.append(mean(avg_test_zero_one_losses))
        list_EMA_train_BCE_losses.append(mean(avg_train_BCE_losses))
        list_EMA_test_BCE_losses.append(mean(avg_test_BCE_losses))
        list_EMA_train_01_losses.append(mean(avg_train_zero_one_losses))
        list_EMA_test_01_losses.append(mean(avg_test_zero_one_losses))
        list_EMA_grad_norm.append(mean(avg_grad_norm) if len(avg_grad_norm) > 0 else 0.0)
        list_num_epochs_per_beta.append(local_epoch)

    print(f"\n{'='*80}")
    print(f"Annealed MALA completed for all {len(beta_values)} beta values")
    print(f"{'='*80}\n")

    # Return results in format compatible with non-annealing case
    return (list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,
            list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, 
            list_EMA_test_01_losses, list_EMA_grad_norm, list_num_epochs_per_beta)


def create_model_with_optional_pbb(dataset_type, n_hidden_layers, width, device,
                                    use_pbb_models=False, pbb_architecture=None,
                                    prior_type='gaussian', sigma_prior=1.0, seed=42):
    """
    Create a model, optionally using PBB architectures with data-free prior.
    
    Args:
        dataset_type (str): Dataset type ('mnist', 'cifar10', 'cifar100', 'synth')
        n_hidden_layers (int or str): Number of hidden layers or architecture name ('L', 'V')
        width (int): Width of hidden layers
        device (str): Device to place model on
        use_pbb_models (bool): Whether to use PBB models (NNet4l or CNNet4l)
        pbb_architecture (str): 'fc' for NNet4l or 'cnn' for CNNet4l (only if use_pbb_models=True)
        prior_type (str): 'gaussian' or 'laplace' for PBB prior
        sigma_prior (float): Scale of the prior
        seed (int): Random seed for reproducibility
        
    Returns:
        torch.nn.Module: The created model, moved to device
    """
    
    # Create PBB models if requested
    if use_pbb_models:
        if not PBB_AVAILABLE:
            raise ImportError("PBB mode requested but PBB modules are not available.")
        if dataset_type != 'mnist':
            raise ValueError(f"PBB mode currently supports only MNIST, got dataset_type='{dataset_type}'")
        if pbb_architecture is None:
            raise ValueError("PBB mode requested but pbb_architecture is None.")
        if sigma_prior is None:
            raise ValueError("PBB mode requested but sigma_prior is None.")

        print(f"\n📦 Using PBB Model: {pbb_architecture.upper()}")
        print(f"   Prior: {prior_type} with scale σ = {sigma_prior}")
        
        if pbb_architecture.lower() in ['fc', 'nnet4l']:
            model = NNet4l(num_classes=10, dropout_prob=0.0)
        elif pbb_architecture.lower() in ['cnn', 'cnnet4l']:
            model = CNNet4l(num_classes=10, dropout_prob=0.0)
        else:
            raise ValueError(f"Unknown PBB architecture: {pbb_architecture}")
        
        # Initialize model with data-free prior
        normalized_prior = str(prior_type).lower() if prior_type is not None else 'gaussian'
        if normalized_prior in ['truncated_gaussian', 'trunc_gaussian', 'truncnorm']:
            model = initialize_prior_truncated_gaussian(
                model,
                sigma_scale=sigma_prior,
                truncation=2.0,
                seed=seed,
            )
        else:
            model = initialize_model_with_prior(model, prior_type=normalized_prior,
                                               sigma_prior=sigma_prior, seed=seed)

        print(f"   ✓ Model initialized with data-free prior")
        
        return model.to(device)
    
    # Fall back to standard model creation if PBB is not used or not available
    if dataset_type == 'mnist':
        if n_hidden_layers == 1:
            model = FCN1L(input_dim=28*28, hidden_dim=width, output_dim=1)
        elif n_hidden_layers == 2:
            model = FCN2L(input_dim=28*28, hidden_dim=width, output_dim=1)
        elif n_hidden_layers == 3:
            model = FCN3L(input_dim=28*28, hidden_dim=width, output_dim=1)
        elif n_hidden_layers == 'L':
            model = LeNet5(num_classes=1)
    elif dataset_type in ('cifar10', 'cifar100'):
        if n_hidden_layers == 1:
            model = FCN1L(input_dim=3*32*32, hidden_dim=width, output_dim=1)
        elif n_hidden_layers == 2:
            model = FCN2L(input_dim=3*32*32, hidden_dim=width, output_dim=1)
        elif n_hidden_layers == 3:
            model = FCN3L(input_dim=3*32*32, hidden_dim=width, output_dim=1)
        elif n_hidden_layers == 'V':
            model = VGG16_CIFAR(num_classes=1)
    else:  # synth dataset
        if n_hidden_layers == 1:
            model = FCN1L(input_dim=4, hidden_dim=width, output_dim=1)
        elif n_hidden_layers == 2:
            model = FCN2L(input_dim=4, hidden_dim=width, output_dim=1)
        elif n_hidden_layers == 3:
            model = FCN3L(input_dim=4, hidden_dim=width, output_dim=1)
    
    return model.to(device)


def save_checkpoint(checkpoint_dir, seed, beta_values, completed_betas, 
                   list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,
                   list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses,
                   list_EMA_grad_norm, list_num_epochs_per_beta, list_EMA_var_train_BCE_losses, list_EMA_var_test_BCE_losses,
                   sample_size=None, config_dict=None):
    """
    Save checkpoint of experiment progress to resume later.
    
    Saves all necessary information to completely resume training and recreate CSV files.
    
    Args:
        checkpoint_dir (str): Directory to save checkpoints in.
        seed (int): Random seed for this experiment.
        beta_values (list): All beta values for this experiment.
        completed_betas (list): List of betas that have been completed.
        list_train_BCE_losses, list_test_BCE_losses, etc. (lists): Accumulated loss lists.
        sample_size (int, optional): Size of training dataset.
        config_dict (dict, optional): Configuration parameters for CSV summary recreation.
    
    Returns:
        str: Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_data = {
        'seed': seed,
        'beta_values': beta_values,
        'completed_betas': completed_betas,
        'list_train_BCE_losses': list_train_BCE_losses,
        'list_test_BCE_losses': list_test_BCE_losses,
        'list_train_01_losses': list_train_01_losses,
        'list_test_01_losses': list_test_01_losses,
        'list_EMA_train_BCE_losses': list_EMA_train_BCE_losses,
        'list_EMA_test_BCE_losses': list_EMA_test_BCE_losses,
        'list_EMA_train_01_losses': list_EMA_train_01_losses,
        'list_EMA_test_01_losses': list_EMA_test_01_losses,
        'list_EMA_grad_norm': list_EMA_grad_norm,
        'list_num_epochs_per_beta': list_num_epochs_per_beta,
        'list_EMA_var_train_BCE_losses': list_EMA_var_train_BCE_losses,
        'list_EMA_var_test_BCE_losses': list_EMA_var_test_BCE_losses,
        'sample_size': sample_size,
        'config_dict': config_dict or {},  # Store config for CSV recreation
    }
    
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_seed_{seed}.pt')
    torch.save(checkpoint_data, checkpoint_file)
    print(f"   ✓ Checkpoint saved: {checkpoint_file} (completed {len(completed_betas)} betas)")
    
    return checkpoint_file


def load_checkpoint(checkpoint_dir, seed):
    """
    Load experiment progress from checkpoint.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints.
        seed (int): Random seed for this experiment.
    
    Returns:
        dict: Checkpoint data if found, None otherwise.
    """
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_seed_{seed}.pt')
    
    if os.path.exists(checkpoint_file):
        checkpoint_data = torch.load(checkpoint_file)
        print(f"   ✓ Checkpoint loaded: {checkpoint_file}")
        print(f"   ✓ Resuming from {len(checkpoint_data['completed_betas'])} completed betas: {checkpoint_data['completed_betas']}")
        return checkpoint_data
    
    return None


def run_beta_experiments(loss, beta_values, a0, b, sigma_gauss_prior, device,n_hidden_layers, width,
                         dataset_type, use_random_labels, l_max,  train_loader, test_loader,min_steps,
                         alpha_average, alpha_stop, eta, eps, test_mode=False, add_grad_norm=False, 
                         sgld_num = 1, annealed = False, add_noise=True, save_every=1, min_steps_first_beta=None,
                         seed=42, selected_classes=None, use_pbb_models=False, pbb_architecture=None,
                         prior_type=None, sigma_prior=None, checkpoint_dir='checkpoints', resume_from_checkpoint=True,
                         max_epochs=None, pmin=None):
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
        use_random_labels (float): Percentage of randomly labelled data
        l_max (float): Maximum loss value for bounded transformation.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        min_steps (int): Minimum number of epochs to train before checking convergence 
            (for subsequent betas in annealing, or all betas in non-annealing).
        alpha_average (float): EMA smoothing factor for loss averaging.
        alpha_stop (float): EMA smoothing factor for convergence detection.
        eta (float): Learning rate threshold parameter.
        eps (float): Convergence threshold for EMA training loss difference.
        test_mode (bool, optional): If True, runs a quick test with fewer epochs. Defaults to False.
        add_grad_norm (bool, optional): Whether to track gradient norm EMA. Defaults to False.
        sgld_num (int, optional): SGLD or MALA. 1 mean SGLD and 0 means MALA
        annealed (bool, optional): Whether to use annealed SGLD. Defaults to False.
        add_noise (bool, optional): Whether to add Langevin noise during SGLD updates. Defaults to True.
        save_every (int, optional): Save checkpoint frequency. Defaults to 1.
        min_steps_first_beta (int, optional): Minimum steps for first beta in annealing (if annealed=True).
            If None, uses min_steps for all betas. Should be larger than min_steps.
        seed (int, optional): Random seed for reproducibility of training. Defaults to 42.
        selected_classes (list, optional): Chosen class ids/groups used to build the dataset.
        checkpoint_dir (str, optional): Directory to save/load checkpoints. Defaults to 'checkpoints'.
        resume_from_checkpoint (bool, optional): If True, resumes from checkpoint if it exists. Defaults to True.
        max_epochs (int, optional): Maximum number of epochs (full passes through batch) to train per beta.
                                   If None, uses convergence criterion only. If set, training stops when
                                   max_epochs is reached regardless of convergence (hard limit on training time).
    
    Returns:
        list: Paths to CSV files generated during the experiment.
    
    Note:
        If beta=0 is not in beta_values, it will be automatically added for proper
        generalization bound computation through prior sampling.
        
        Checkpoints are automatically saved after each beta is completed, allowing
        resumption from where the experiment left off (useful for long training on servers).
    """ 
    # Ensure beta=0 is included for proper bound computation
    extended_beta_values = list(beta_values)
    if 0.0 not in extended_beta_values and 0 not in extended_beta_values:
        extended_beta_values = [0.0] + extended_beta_values
        print(f"Added beta=0 for proper generalization bound computation")    

    print(f"🆕 Starting new experiment")

    # Try to load from checkpoint if resume is enabled (only for non-annealed mode)
    checkpoint_data = None
    completed_betas = []
    if resume_from_checkpoint and sgld_num == 1 and not annealed:
        checkpoint_data = load_checkpoint(checkpoint_dir, seed)
        if checkpoint_data is not None:
            extended_beta_values = checkpoint_data['beta_values']
            completed_betas = checkpoint_data['completed_betas']
            list_train_BCE_losses = checkpoint_data['list_train_BCE_losses']
            list_test_BCE_losses = checkpoint_data['list_test_BCE_losses']
            list_train_01_losses = checkpoint_data['list_train_01_losses']
            list_test_01_losses = checkpoint_data['list_test_01_losses']
            list_EMA_train_BCE_losses = checkpoint_data['list_EMA_train_BCE_losses']
            list_EMA_test_BCE_losses = checkpoint_data['list_EMA_test_BCE_losses']
            list_EMA_train_01_losses = checkpoint_data['list_EMA_train_01_losses']
            list_EMA_test_01_losses = checkpoint_data['list_EMA_test_01_losses']
            list_EMA_grad_norm = checkpoint_data['list_EMA_grad_norm']
            list_num_epochs_per_beta = checkpoint_data['list_num_epochs_per_beta']
            list_EMA_var_train_BCE_losses = checkpoint_data['list_EMA_var_train_BCE_losses']
            list_EMA_var_test_BCE_losses = checkpoint_data['list_EMA_var_test_BCE_losses']
        else:
            print("   ℹ️ No checkpoint found, starting fresh experiment")
            # Initialize empty lists for fresh start
            list_train_BCE_losses = []
            list_test_BCE_losses = []
            list_train_01_losses = []
            list_test_01_losses = []
            list_EMA_train_BCE_losses = []
            list_EMA_test_BCE_losses = []
            list_EMA_train_01_losses = []
            list_EMA_test_01_losses = []
            list_EMA_grad_norm = []
            list_num_epochs_per_beta = []
            list_EMA_var_train_BCE_losses = []
            list_EMA_var_test_BCE_losses = []
    else:
        # Initialize empty lists for non-checkpoint scenarios
        list_train_BCE_losses = []
        list_test_BCE_losses = []
        list_train_01_losses = []
        list_test_01_losses = []
        list_EMA_train_BCE_losses = []
        list_EMA_test_BCE_losses = []
        list_EMA_train_01_losses = []
        list_EMA_test_01_losses = []
        list_EMA_grad_norm = []
        list_num_epochs_per_beta = []
        list_EMA_var_train_BCE_losses = []
        list_EMA_var_test_BCE_losses = []

    # Set random seeds for reproducibility of training
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Training random seed set to: {seed}")

    saved_csv_paths = []

    print(f"\nConfiguration:")    
    print(f"Learning rate (a0) per beta:")
    for beta in sorted(extended_beta_values):
        current_a0 = get_a0_for_beta(beta, a0)
        print(f"  Beta {beta}: {current_a0}")
    print(f"Save frequency: every {save_every} repetitions")
    print(f"{'='*80}")        
    
    if add_noise == False:
        extended_beta_values = [extended_beta_values[-1]]  

    if dataset_type == 'cifar10':
        dataset_name = 'CIFAR-10'
    elif dataset_type == 'cifar100':
        dataset_name = 'CIFAR-100'
    elif dataset_type == 'mnist':
        dataset_name = 'MNIST'
    elif dataset_type == 'synth':
        dataset_name = 'SYNTH'
    else:
        dataset_name = str(dataset_type)

    selected_classes_str = str(selected_classes) if selected_classes is not None else 'N/A'

    normalized_prior_type = str(prior_type).lower() if prior_type is not None else 'gaussian'
    use_layerwise_prior_in_sgld = (
        use_pbb_models
        and normalized_prior_type in ['truncated_gaussian', 'trunc_gaussian', 'truncnorm']
        and dataset_type == 'mnist'
    )
    layerwise_prior_scale = sigma_prior if sigma_prior is not None else 1.0
    
    # Store all configuration for checkpoint (needed for CSV recreation)
    config_dict = {
        'device': str(device),
        'loss': loss,
        'l_max': l_max,
        'n_hidden_layers': n_hidden_layers,
        'width': width,
        'dataset_type': dataset_type,
        'dataset_name': dataset_name,
        'selected_classes_str': selected_classes_str,
        'use_random_labels': use_random_labels,
        'min_steps': min_steps,
        'alpha_average': alpha_average,
        'alpha_stop': alpha_stop,
        'eta': eta,
        'eps': eps,
        'b': b,
        'sigma_gauss_prior': sigma_gauss_prior,
        'pmin': pmin,
        'max_epochs': max_epochs,
    }

    if use_layerwise_prior_in_sgld:
        print(f"Using PBB-style layer-wise truncated Gaussian prior in SGLD (scale={layerwise_prior_scale})")

    if sgld_num == 1 and annealed:
        # In annealed mode, we use a single model and transition between betas
        print(f"\n🔥 Running ANNEALED SGLD with {len(extended_beta_values)} beta values")
        current_a0 = get_a0_for_beta(extended_beta_values[0], a0)  # Use a0 for first beta
        
        # Create model once for all betas (supports optional PBB setup)
        model = create_model_with_optional_pbb(
            dataset_type=dataset_type,
            n_hidden_layers=n_hidden_layers,
            width=width,
            device=device,
            use_pbb_models=use_pbb_models,
            pbb_architecture=pbb_architecture,
            prior_type=prior_type,
            sigma_prior=sigma_prior,
            seed=seed
        )
        
        # Run annealed training (returns results for all betas)
        (list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,
         list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, 
         list_EMA_test_01_losses, list_EMA_grad_norm, list_num_epochs_per_beta) = train_annealed_sgld_model(
            loss=loss,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            min_steps=min_steps,
            a0=current_a0,
            b=b,
            sigma_gauss_prior=sigma_gauss_prior,
            beta_values=extended_beta_values,
            device=device,
            dataset_type=dataset_type,
            l_max=l_max,
            alpha_average=alpha_average,
            alpha_stop=alpha_stop,
            eta=eta,
            eps=eps,
            add_noise=add_noise,
            min_steps_first_beta=min_steps_first_beta,
            prior_type=normalized_prior_type,
            use_layerwise_prior_in_sgld=use_layerwise_prior_in_sgld,
            layerwise_prior_scale=layerwise_prior_scale,
            pmin=pmin,
        )
        
        # Save results for annealed case
        filename_prefix = ""
        if dataset_type == 'mnist':
            filename_prefix = "M"
        elif dataset_type in ('cifar10', 'cifar100'):
            filename_prefix = "C"
        else:
            filename_prefix = "S"
        
        if use_random_labels == 1:
            filename_prefix += "R"
        elif use_random_labels == 0:
            filename_prefix += "C"
        else:
            filename_prefix += "P"
        
        filename_prefix += f"L{n_hidden_layers}"
        filename_prefix += f"W{width}"
        if len(train_loader) == 1:
            filename_prefix += "ULA" if add_noise else "GD"
        else:
            filename_prefix += "SGLD" if add_noise else "SGD"
        
        filename_prefix += f"{len(train_loader.dataset)/1000:.0f}k"
        filename_prefix += f"LR{str(current_a0).replace('.', '')}"
        filename_prefix += f"{loss.upper()}"
        filename_prefix += "_ANNEALED"  # Mark as annealed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix += f"_S{seed}"
        filename_prefix += f"_{timestamp}"
        if test_mode:
            filename_prefix += "_TEST"
        
        summary_string = f"Annealed LMC has been run with the following parameters:\n" \
        f"  - Device: {device}\n" \
        f"  - Loss function: {loss}\n" \
        f"  - l_max: {l_max}\n" \
        f"  - Network architecture: {model.__class__.__name__}\n" \
        f"  - Number of hidden layers: {n_hidden_layers}\n" \
        f"  - Width of hidden layers: {width}\n" \
        f"  - Dataset type: {dataset_type}\n" \
        f"  - Dataset name: {dataset_name}\n" \
        f"  - Selected classes/groups: {selected_classes_str}\n" \
        f"  - Random labels: {use_random_labels}\n" \
        f"  - Training set size: {len(train_loader.dataset)}\n" \
        f"  - Test set size: {len(test_loader.dataset)}\n" \
        f"  - Minimum steps per beta: {min_steps}\n" \
        f"  - Number of Batches: {len(train_loader)}\n" \
        f"  - Beta values (annealed): {sorted(extended_beta_values)}\n" \
        f"  - Learning rate (a0): {current_a0}\n" \
        f"  - Learning rate decay (b): {b}\n" \
        f"  - Gaussian prior sigma: {sigma_gauss_prior}\n" \
        f"  - alpha_average: {alpha_average}\n" \
        f"  - alpha_stop: {alpha_stop}\n" \
        f"  - eta: {eta}\n" \
        f"  - eps: {eps}\n" \
        f"  - Seed: {seed}\n" \
        f"  - Number of epochs per beta: {list_num_epochs_per_beta}\n"

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
            sorted(extended_beta_values),
            len(train_loader.dataset),
            summary_string
        )
        saved_csv_paths.append(csv_path)
        print(f"\n✅ Annealed SGLD completed! Results saved to: {csv_path}")

    elif sgld_num == 1 and not annealed:  
        betas_experimented = []
        list_num_epochs_per_beta = []
        # Run all beta values for this repetition
        for beta in sorted(extended_beta_values):
            # Skip if already completed in checkpoint
            if checkpoint_data is not None and beta in completed_betas:
                print(f"\n--- Beta = {beta} (SKIPPED - already completed) ---")
                betas_experimented.append(beta)
                continue
                
            betas_experimented.append(beta)
            current_a0 = get_a0_for_beta(beta, a0)
            
            print(f"\n--- Beta = {beta} ---")
            print(f"Learning rate: {current_a0}")
            
            # Create fresh model for each beta-repetition combination
            model = create_model_with_optional_pbb(
                dataset_type=dataset_type,
                n_hidden_layers=n_hidden_layers,
                width=width,
                device=device,
                use_pbb_models=use_pbb_models,
                pbb_architecture=pbb_architecture,
                prior_type=prior_type,
                sigma_prior=sigma_prior,
                seed=seed
            )

            # Train the model
            
            training_results = train_sgld_model(
                loss =loss,
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                min_steps=min_steps,
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
                eps=eps,
                add_noise=add_noise,
                prior_type=normalized_prior_type,
                use_layerwise_prior_in_sgld=use_layerwise_prior_in_sgld,
                layerwise_prior_scale=layerwise_prior_scale,
                max_epochs=max_epochs,
                pmin=pmin,
            )
            

            (train_losses, test_losses, _, _, train_01_losses, test_01_losses, _, EMA_train_losses,
            EMA_train_BCE_losses, EMA_test_BCE_losses, EMA_train_01_losses, EMA_test_01_losses,
            EMA_train_BCE_losses_sq, EMA_test_BCE_losses_sq, EMA_grad_norm, epoch) = training_results

            list_train_BCE_losses.append(train_losses[-50])
            list_test_BCE_losses.append(test_losses[-50])
            list_train_01_losses.append(train_01_losses[-50])
            list_test_01_losses.append(test_01_losses[-50])
            list_EMA_train_BCE_losses.append(mean(EMA_train_BCE_losses))
            list_EMA_test_BCE_losses.append(mean(EMA_test_BCE_losses))
            list_EMA_train_01_losses.append(mean(EMA_train_01_losses))
            list_EMA_test_01_losses.append(mean(EMA_test_01_losses))
            list_EMA_grad_norm.append(mean(EMA_grad_norm))
            list_num_epochs_per_beta.append(epoch)
            list_EMA_var_train_BCE_losses.append(mean(EMA_train_BCE_losses_sq) - mean(EMA_train_BCE_losses)**2)
            list_EMA_var_test_BCE_losses.append(mean(EMA_test_BCE_losses_sq) - mean(EMA_test_BCE_losses)**2)

            print(f"  Final - Train BCE: {train_losses[-1]:.4f}, Test BCE: {test_losses[-1]:.4f}, "
                    f"Train 0-1: {train_01_losses[-1]:.4f}, Test 0-1: {test_01_losses[-1]:.4f}")
        
            # Save checkpoint after each beta is completed
            completed_betas.append(beta)
            save_checkpoint(checkpoint_dir, seed, extended_beta_values, completed_betas,
                           list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,
                           list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses,
                           list_EMA_grad_norm, list_num_epochs_per_beta, list_EMA_var_train_BCE_losses, list_EMA_var_test_BCE_losses,
                           sample_size=len(train_loader.dataset), config_dict=config_dict)

            # Save a csv file after each repetition if requested
            # Generate filename prefix based on experiment parameters
            filename_prefix = ""
            if dataset_type == 'mnist':
                filename_prefix = "M"
            elif dataset_type in ('cifar10', 'cifar100'):
                filename_prefix = "C"
            else:
                filename_prefix = "S"
            
            if use_random_labels == 1:
                filename_prefix += "R"
            elif use_random_labels == 0:
                filename_prefix += "C"
            else: 
                filename_prefix += "P"
            
            filename_prefix += f"L{n_hidden_layers}"
            filename_prefix += f"W{width}"
            if len(train_loader) == 1:
                if add_noise == True:
                    filename_prefix += "ULA"
                else:
                    filename_prefix += "GD"
            else:
                if add_noise == True:
                    filename_prefix += "SGLD"
                else:
                    filename_prefix += "SGD"

            filename_prefix += f"{len(train_loader.dataset)/1000:.0f}k"
            filename_prefix += f"LR{current_a0}".replace('.', '')
            filename_prefix += f"{loss.upper()}"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename_prefix += f"_S{seed}"
            filename_prefix += f"_{timestamp}"
            if test_mode:
                filename_prefix += f"_TEST"
            
            summary_string =  f"The LMC has been run with the following parameters:\n" \
            f"  - Device: {device}\n" \
            f"  - Loss function: {loss}\n" \
            f"  - l_max: {l_max}\n" \
            f"  - Network architecture: {model.__class__.__name__}\n" \
            f"  - Number of hidden layers: {n_hidden_layers}\n" \
            f"  - Width of hidden layers: {width}\n" \
            f"  - Dataset type: {dataset_type}\n" \
            f"  - Dataset name: {dataset_name}\n" \
            f"  - Selected classes/groups: {selected_classes_str}\n" \
            f"  - Random labels: {use_random_labels}\n" \
            f"  - Training set size: {len(train_loader.dataset) if train_loader else 'N/A'}\n" \
            f"  - Test set size: {len(test_loader.dataset) if test_loader else 'N/A'}\n" \
            f"  - Minimum epochs: {min_steps}\n" \
            f"  - Number of Batches: {len(train_loader) if train_loader else 'N/A'}\n" \
            f"  - Beta values: {sorted(betas_experimented)}\n" \
            f"  - Learning rate (a0): {current_a0}\n" \
            f"  - Learning rate decay (b): {b}\n" \
            f"  - Gaussian prior sigma: {sigma_gauss_prior}\n" \
            f" -  alpha_average: {alpha_average}\n" \
            f" -  alpha_stop: {alpha_stop}\n" \
            f" -  eta: {eta}\n" \
            f" -  eps: {eps}\n" \
            f" -  Seed: {seed}\n" \
            f" -  Gradient norm: {list_EMA_grad_norm}\n" \
            f" -  Number of epochs per beta: {list_num_epochs_per_beta}\n" \
            f" -  Variance of EMA Train BCE losses: {list_EMA_var_train_BCE_losses}\n" \
            f" -  Variance of EMA Test BCE losses: {list_EMA_var_test_BCE_losses}\n"

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
            saved_csv_paths.append(csv_path)
    
    elif sgld_num == 0:
        # In annealed mode, we use a single model and transition between betas
        print(f"\n🔥 Running ANNEALED MALA with {len(extended_beta_values)} beta values")
        current_a0 = get_a0_for_beta(extended_beta_values[0], a0)  # Use a0 for first beta
        
        # Create model once for all betas (using helper function for optional PBB support)
        model = create_model_with_optional_pbb(
            dataset_type=dataset_type,
            n_hidden_layers=n_hidden_layers,
            width=width,
            device=device,
            use_pbb_models=use_pbb_models,
            pbb_architecture=pbb_architecture,
            prior_type=prior_type,
            sigma_prior=sigma_prior,
            seed=seed
        )
        
        # Initialize model weights from Gaussian prior
        # model = initialize_nn_weights_gaussian(model, sigma=sigma_gauss_prior, seed=seed)

        # Run annealed training (returns results for all betas)
        (list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,
         list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, 
         list_EMA_test_01_losses, list_EMA_grad_norm, list_num_epochs_per_beta) = train_annealed_mala_model(
            loss=loss,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            min_steps=min_steps,
            a0=current_a0,
            b=b,
            sigma_gauss_prior=sigma_gauss_prior,
            beta_values=extended_beta_values,
            device=device,
            dataset_type=dataset_type,
            l_max=l_max,
            alpha_average=alpha_average,
            alpha_stop=alpha_stop,
            eta=eta,
            eps=eps,
            pmin=pmin,
        )
        
        # Save results for annealed case
        filename_prefix = ""
        if dataset_type == 'mnist':
            filename_prefix = "M"
        
        elif dataset_type in ('cifar10', 'cifar100'):
            filename_prefix = "C"
        else:
            filename_prefix = "S"
        
        if use_random_labels == 1:
            filename_prefix += "R"
        elif use_random_labels == 0:
            filename_prefix += "C"
        else:
            filename_prefix += "P"
        
        filename_prefix += f"L{n_hidden_layers}"
        filename_prefix += f"W{width}"
        if len(train_loader) == 1:
            filename_prefix += "MULA" if add_noise else "GD"
        else:
            filename_prefix += "MALA" if add_noise else "SGD"
        
        filename_prefix += f"{len(train_loader.dataset)/1000:.0f}k"
        filename_prefix += f"LR{str(current_a0).replace('.', '')}"
        filename_prefix += f"{loss.upper()}"
        filename_prefix += "_ANNEALED"  # Mark as annealed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix += f"_S{seed}"
        filename_prefix += f"_{timestamp}"
        if test_mode:
            filename_prefix += "_TEST"
        
        summary_string = f"Annealed LMC has been run with the following parameters:\n" \
        f"  - Device: {device}\n" \
        f"  - Loss function: {loss}\n" \
        f"  - l_max: {l_max}\n" \
        f"  - Network architecture: {model.__class__.__name__}\n" \
        f"  - Number of hidden layers: {n_hidden_layers}\n" \
        f"  - Width of hidden layers: {width}\n" \
        f"  - Dataset type: {dataset_type}\n" \
        f"  - Dataset name: {dataset_name}\n" \
        f"  - Selected classes/groups: {selected_classes_str}\n" \
        f"  - Random labels: {use_random_labels}\n" \
        f"  - Training set size: {len(train_loader.dataset)}\n" \
        f"  - Test set size: {len(test_loader.dataset)}\n" \
        f"  - Minimum steps per beta: {min_steps}\n" \
        f"  - Number of Batches: {len(train_loader)}\n" \
        f"  - Beta values (annealed): {sorted(extended_beta_values)}\n" \
        f"  - Learning rate (a0): {current_a0}\n" \
        f"  - Learning rate decay (b): {b}\n" \
        f"  - Gaussian prior sigma: {sigma_gauss_prior}\n" \
        f"  - alpha_average: {alpha_average}\n" \
        f"  - alpha_stop: {alpha_stop}\n" \
        f"  - eta: {eta}\n" \
        f"  - eps: {eps}\n" \
        f"  - Seed: {seed}\n" \
        f"  - Number of epochs per beta: {list_num_epochs_per_beta}\n"

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
            sorted(extended_beta_values),
            len(train_loader.dataset),
            summary_string
        )
        saved_csv_paths.append(csv_path)
        print(f"\n✅ Annealed SGLD completed! Results saved to: {csv_path}")

    return saved_csv_paths


