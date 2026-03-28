"""
Training and experiment orchestration for multiclass Gibbs generalization experiments.

This module mirrors the structure of training.py, adapted for multiclass
classification.
"""

import csv
import math
import os
import random
import time
from datetime import datetime
from statistics import mean

import numpy as np
import torch
import torch.nn as nn

from losses import BoundedCrossEntropyLoss, MulticlassSavageLoss, ZeroOneLoss
from models import FCN1L, FCN2L, FCN3L, LeNet5, VGG16_CIFAR, initialize_nn_weights_gaussian
from sgld import SGLD


class NLLFromLogitsLoss(nn.Module):
    """Apply log_softmax then NLLLoss so raw logits are accepted safely."""

    def __init__(self):
        super().__init__()
        self._nll = nn.NLLLoss()

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        return self._nll(log_probs, targets)


def build_multiclass_criterion(loss_name, l_max):
    """Return a validated multiclass criterion and its normalized name.

    Supported multiclass losses:
    - 'ce': CrossEntropyLoss on logits.
    - 'bbce': bounded cross-entropy in [0, ell_max].
    - 'savage': Multiclass Savage loss (1 - p_y)^2.
    - 'nll': NLL on log-softmax(logits).
    """
    normalized = str(loss_name).lower().strip()
    if normalized == 'ce':
        return nn.CrossEntropyLoss(), normalized
    if normalized == 'bbce':
        return BoundedCrossEntropyLoss(ell_max=l_max), normalized
    if normalized == 'savage':
        return MulticlassSavageLoss(), normalized
    if normalized == 'nll':
        return NLLFromLogitsLoss(), normalized
    if normalized == 'tangent':
        raise ValueError(
            "'tangent' is a binary-margin loss and is not supported for multiclass in training_multiclass.py. "
            "Use one of: 'ce', 'bbce', 'savage', 'nll'."
        )

    raise ValueError(
        f"Unsupported multiclass loss '{loss_name}'. Use one of: 'ce', 'bbce', 'savage', 'nll'."
    )


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
    summary_string,
):
    """Save experiment metrics to csv_EMA."""
    os.makedirs('csv_EMA', exist_ok=True)
    filename = f"csv_EMA/{filename_prefix}.csv"

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Sample_size',
            'Beta',
            'BCE_Train',
            'BCE_Test',
            '0-1_Train',
            '0-1_Test',
            'EMA_BCE_Train',
            'EMA_BCE_Test',
            'EMA_0-1_Train',
            'EMA_0-1_Test',
        ])

        for i, beta in enumerate(sorted(beta_values)):
            writer.writerow([
                sample_size,
                beta,
                list_train_BCE_losses[i],
                list_test_BCE_losses[i],
                list_train_01_losses[i],
                list_test_01_losses[i],
                list_EMA_train_BCE_losses[i],
                list_EMA_test_BCE_losses[i],
                list_EMA_train_01_losses[i],
                list_EMA_test_01_losses[i],
            ])

        writer.writerow([])
        writer.writerow(['Summary:', summary_string])

    print(f"   EMA CSV saved: {filename} ({len(beta_values)} beta values)")
    return filename


def get_a0_for_beta(beta, a0):
    """Resolve learning rate for a given beta."""
    if isinstance(a0, (int, float)):
        return float(a0)
    if isinstance(a0, dict):
        return a0.get(beta, 1e-1)
    if callable(a0):
        return a0(beta)
    raise ValueError(f"a0 must be int, float, dict, or callable, got {type(a0)}")


def _to_class_indices(targets, num_classes_hint=None):
    """Convert targets to class indices when one-hot labels are provided."""
    if targets.ndim > 1:
        if targets.shape[-1] == 1:
            return targets.squeeze(-1).long()
        if num_classes_hint is not None and targets.shape[-1] == num_classes_hint:
            return torch.argmax(targets, dim=-1).long()
    return targets.long()


def _predict_from_outputs(outputs):
    """Return predictions for binary or multiclass outputs."""
    logits = outputs.squeeze()
    if logits.ndim <= 1:
        return (logits > 0).long()
    if logits.ndim == 2 and logits.size(1) == 1:
        return (logits.squeeze(1) > 0).long()
    return torch.argmax(logits, dim=1)


def _count_correct_predictions(predicted, targets):
    """Count correct predictions for class-index or one-hot targets."""
    if targets.ndim > 1:
        if targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        else:
            targets = torch.argmax(targets, dim=1)
    return (predicted == targets).sum().item()


def _bootstrap_ema_train_losses(model, train_loader, criterion, device):
    """Initialize EMA from first train batch to avoid arbitrary constants."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        batch_x, batch_y = next(iter(train_loader))
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_y = _to_class_indices(batch_y)
        outputs = model(batch_x)
        initial_loss = float(criterion(outputs.squeeze(), batch_y).item())
    if was_training:
        model.train()
    return [0.0, initial_loss]


def create_multiclass_model(dataset_type, n_hidden_layers, width, device, num_classes):
    """Create a model for multiclass experiments."""
    if dataset_type == 'mnist':
        if n_hidden_layers == 1:
            model = FCN1L(input_dim=28 * 28, hidden_dim=width, output_dim=num_classes)
        elif n_hidden_layers == 2:
            model = FCN2L(input_dim=28 * 28, hidden_dim=width, output_dim=num_classes)
        elif n_hidden_layers == 3:
            model = FCN3L(input_dim=28 * 28, hidden_dim=width, output_dim=num_classes)
        elif n_hidden_layers == 'L':
            model = LeNet5(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported n_hidden_layers for MNIST: {n_hidden_layers}")
    elif dataset_type in ('cifar10', 'cifar100'):
        if n_hidden_layers == 1:
            model = FCN1L(input_dim=3 * 32 * 32, hidden_dim=width, output_dim=num_classes)
        elif n_hidden_layers == 2:
            model = FCN2L(input_dim=3 * 32 * 32, hidden_dim=width, output_dim=num_classes)
        elif n_hidden_layers == 3:
            model = FCN3L(input_dim=3 * 32 * 32, hidden_dim=width, output_dim=num_classes)
        elif n_hidden_layers == 'V':
            model = VGG16_CIFAR(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported n_hidden_layers for CIFAR: {n_hidden_layers}")
    elif dataset_type == 'synth':
        if n_hidden_layers == 1:
            model = FCN1L(input_dim=4, hidden_dim=width, output_dim=num_classes)
        elif n_hidden_layers == 2:
            model = FCN2L(input_dim=4, hidden_dim=width, output_dim=num_classes)
        elif n_hidden_layers == 3:
            model = FCN3L(input_dim=4, hidden_dim=width, output_dim=num_classes)
        else:
            raise ValueError(f"Unsupported n_hidden_layers for synth: {n_hidden_layers}")
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    return model.to(device)


def validate_model_output_dim(model, train_loader, device, expected_num_classes):
    """Run one forward pass and verify the model output width matches class count."""
    model.eval()
    with torch.no_grad():
        batch_x, _ = next(iter(train_loader))
        batch_x = batch_x.to(device, non_blocking=True)
        outputs = model(batch_x)

    if outputs.ndim < 2:
        raise ValueError(
            f"Expected multiclass logits with shape [N, C], got shape {tuple(outputs.shape)}"
        )

    output_dim = int(outputs.shape[1])
    if output_dim != int(expected_num_classes):
        raise ValueError(
            f"Model output dim ({output_dim}) does not match number of classes ({expected_num_classes})"
        )


def train_sgld_model(
    loss,
    model,
    train_loader,
    test_loader,
    min_steps,
    a0,
    b,
    sigma_gauss_prior,
    beta,
    device,
    dataset_type,
    l_max,
    alpha_average,
    alpha_stop,
    eta,
    eps,
    add_noise=True,
    max_epochs=None,
    stopping_mode='ema',
):
    """Train one model for one beta using SGLD."""
    if isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)

    criterion, loss_name = build_multiclass_criterion(loss, l_max)

    zero_one_criterion = ZeroOneLoss()

    optimizer = SGLD(
        model.parameters(),
        lr=a0,
        sigma_gauss_prior=sigma_gauss_prior,
        beta=beta,
        add_noise=add_noise,
    )

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    train_zero_one_losses, test_zero_one_losses = [], []
    learning_rates = []

    EMA_train_losses = _bootstrap_ema_train_losses(model, train_loader, criterion, device)
    EMA_alpha = alpha_stop

    avg_train_BCE_losses, avg_test_BCE_losses = [], []
    avg_train_zero_one_losses, avg_test_zero_one_losses = [], []
    avg_train_BCE_losses_sq, avg_test_BCE_losses_sq = [], []
    avg_grad_norm = []
    p_grad_norm = 2

    print(f"Training with SGLD: a0={a0}, b={b}, sigma_gauss_prior={sigma_gauss_prior}, beta={beta}")
    print(f"Dataset type: {dataset_type}, Device: {device}")

    stopping_mode = str(stopping_mode).lower().strip()
    if stopping_mode not in {'ema', 'max_iter_only'}:
        raise ValueError("stopping_mode must be one of: 'ema', 'max_iter_only'")
    if stopping_mode == 'max_iter_only' and (max_epochs is None or max_epochs <= 0):
        raise ValueError("For stopping_mode='max_iter_only', max_epochs must be a positive integer")

    start_time = time.time()
    epoch = 0

    if beta == 0.0:
        num_prior_samples = 1000
        model_cpu = model.to('cpu')

        for i in range(num_prior_samples):
            learning_rates.append(0.0)
            optimizer.zero_grad(set_to_none=True)
            model_cpu = initialize_nn_weights_gaussian(model_cpu, sigma=sigma_gauss_prior, seed=42 + i * 1000)

            train_x, train_y = next(iter(train_loader))
            train_y = _to_class_indices(train_y)
            outputs = model_cpu(train_x)
            loss_fn = criterion(outputs.squeeze(), train_y)
            train_losses.append(loss_fn.item())
            train_zero_one_losses.append(float(zero_one_criterion(outputs, train_y).item()))

            loss_fn.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model_cpu.parameters(), float('inf'), norm_type=2).item()
            avg_grad_norm.append(grad_norm ** p_grad_norm)
            grad_dim = sum(p.grad.numel() for p in model_cpu.parameters() if p.grad is not None)

            with torch.no_grad():
                test_x, test_y = next(iter(test_loader))
                test_y = _to_class_indices(test_y)
                test_outputs = model_cpu(test_x)
                test_loss_fn = criterion(test_outputs.squeeze(), test_y)
                test_losses.append(test_loss_fn.item())
                test_zero_one_losses.append(float(zero_one_criterion(test_outputs, test_y).item()))

            avg_train_BCE_losses.append(train_losses[-1])
            avg_test_BCE_losses.append(test_losses[-1])
            avg_train_zero_one_losses.append(train_zero_one_losses[-1])
            avg_test_zero_one_losses.append(test_zero_one_losses[-1])
            avg_train_BCE_losses_sq.append(train_losses[-1] ** 2)
            avg_test_BCE_losses_sq.append(test_losses[-1] ** 2)

            epoch += 1
            print(
                f"Beta=0.0 sample {epoch}/{num_prior_samples} | "
                f"Train: {train_losses[-1]:.4f} Test: {test_losses[-1]:.4f} "
                f"EMA Train: {mean(avg_train_BCE_losses):.4f} EMA Test: {mean(avg_test_BCE_losses):.4f} "
                f"L-p grad norm: {(1 / math.sqrt(grad_dim)) * ((sum(avg_grad_norm) / epoch) ** (1 / p_grad_norm)):.6f}"
            )

    while beta > 0.0:
        if stopping_mode == 'max_iter_only':
            if epoch >= max_epochs:
                print(f"Reached maximum epochs limit: {max_epochs}. Stopping training.")
                break
        else:
            if not (EMA_train_losses[-1] - EMA_train_losses[-2] < eps or epoch < min_steps / len(train_loader)):
                break
            if max_epochs is not None and epoch >= max_epochs:
                print(f"Reached maximum epochs limit: {max_epochs}. Stopping training.")
                break

        model.train()
        train_loss_total = 0.0
        train_zero_one_total = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = _to_class_indices(batch_y.to(device, non_blocking=True))

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            loss_fn = criterion(outputs.squeeze(), batch_y)
            predicted = _predict_from_outputs(outputs)
            zero_one_loss = zero_one_criterion(outputs, batch_y)

            loss_fn.backward()
            optimizer.step()

            prev_loss = train_losses[-1] if train_losses else loss_fn.item()
            EMA_train_losses.append((EMA_alpha / 2) * prev_loss + (EMA_alpha / 2) * loss_fn.item() + (1 - EMA_alpha) * EMA_train_losses[-1])

            bce_val = loss_fn.item()
            train_loss_total += bce_val
            train_zero_one_total += zero_one_loss.item()
            train_total += batch_y.size(0)
            train_correct += _count_correct_predictions(predicted, batch_y)

        learning_rates.append(optimizer.param_groups[0]['lr'])

        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_zero_one = train_zero_one_total / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        train_losses.append(avg_train_loss)
        train_zero_one_losses.append(avg_train_zero_one)
        train_accuracies.append(train_accuracy)

        model.eval()
        test_loss_total = 0.0
        test_zero_one_total = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = _to_class_indices(batch_y.to(device, non_blocking=True))

                outputs = model(batch_x)
                loss_fn = criterion(outputs.squeeze(), batch_y)
                predicted = _predict_from_outputs(outputs)
                zero_one_loss = zero_one_criterion(outputs, batch_y)

                test_loss_total += loss_fn.item()
                test_zero_one_total += zero_one_loss.item()
                test_total += batch_y.size(0)
                test_correct += _count_correct_predictions(predicted, batch_y)

        avg_test_loss = test_loss_total / len(test_loader)
        avg_test_zero_one = test_zero_one_total / len(test_loader)
        test_accuracy = 100 * test_correct / test_total

        test_losses.append(avg_test_loss)
        test_zero_one_losses.append(avg_test_zero_one)
        test_accuracies.append(test_accuracy)

        avg_train_BCE_losses.append(avg_train_loss)
        avg_test_BCE_losses.append(avg_test_loss)
        avg_train_zero_one_losses.append(avg_train_zero_one)
        avg_test_zero_one_losses.append(avg_test_zero_one)
        avg_train_BCE_losses_sq.append(avg_train_loss ** 2)
        avg_test_BCE_losses_sq.append(avg_test_loss ** 2)

        epoch += 1

        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:>5} | Beta={beta} | "
                f"Train Loss={avg_train_loss:.4f} Test Loss={avg_test_loss:.4f} | "
                f"Train 0-1={avg_train_zero_one:.4f} Test 0-1={avg_test_zero_one:.4f} | "
                f"Train Acc={train_accuracy:.2f}% Test Acc={test_accuracy:.2f}% | "
                f"Elapsed={elapsed:.1f}s"
            )

    return (
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
        train_zero_one_losses,
        test_zero_one_losses,
        learning_rates,
        EMA_train_losses,
        avg_train_BCE_losses,
        avg_test_BCE_losses,
        avg_train_zero_one_losses,
        avg_test_zero_one_losses,
        avg_train_BCE_losses_sq,
        avg_test_BCE_losses_sq,
        avg_grad_norm,
        epoch,
    )


def save_checkpoint(
    checkpoint_dir,
    seed,
    beta_values,
    completed_betas,
    list_train_BCE_losses,
    list_test_BCE_losses,
    list_train_01_losses,
    list_test_01_losses,
    list_EMA_train_BCE_losses,
    list_EMA_test_BCE_losses,
    list_EMA_train_01_losses,
    list_EMA_test_01_losses,
    list_EMA_grad_norm,
    list_num_epochs_per_beta,
    list_EMA_var_train_BCE_losses,
    list_EMA_var_test_BCE_losses,
    sample_size=None,
    config_dict=None,
):
    """Save experiment progress for resumable runs."""
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
        'config_dict': config_dict or {},
    }
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_seed_{seed}.pt')
    torch.save(checkpoint_data, checkpoint_file)
    print(f"   Checkpoint saved: {checkpoint_file} (completed {len(completed_betas)} betas)")
    return checkpoint_file


def load_checkpoint(checkpoint_dir, seed):
    """Load experiment progress from checkpoint if available."""
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_seed_{seed}.pt')
    if os.path.exists(checkpoint_file):
        checkpoint_data = torch.load(checkpoint_file)
        print(f"   Checkpoint loaded: {checkpoint_file}")
        print(f"   Resuming from {len(checkpoint_data['completed_betas'])} completed betas: {checkpoint_data['completed_betas']}")
        return checkpoint_data
    return None


def run_beta_experiments(
    loss,
    beta_values,
    a0,
    b,
    sigma_gauss_prior,
    device,
    n_hidden_layers,
    width,
    dataset_type,
    use_random_labels,
    l_max,
    train_loader,
    test_loader,
    min_steps,
    alpha_average,
    alpha_stop,
    eta,
    eps,
    test_mode=False,
    add_grad_norm=False,
    sgld_num=1,
    annealed=False,
    add_noise=True,
    save_every=1,
    min_steps_first_beta=None,
    seed=42,
    selected_classes=None,
    checkpoint_dir='checkpoints',
    resume_from_checkpoint=True,
    max_epochs=None,
    stopping_mode='ema',
):
    """Run multiclass SGLD experiments across beta values."""
    if sgld_num != 1 or annealed:
        raise NotImplementedError("training_multiclass.py currently supports sgld_num=1 and annealed=False only.")

    extended_beta_values = list(beta_values)
    if 0.0 not in extended_beta_values and 0 not in extended_beta_values:
        extended_beta_values = [0.0] + extended_beta_values
        print("Added beta=0 for proper generalization bound computation")

    print("Starting new multiclass experiment")

    checkpoint_data = None
    completed_betas = []
    if resume_from_checkpoint:
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Training random seed set to: {seed}")

    if add_noise is False:
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
    num_classes = len(selected_classes) if isinstance(selected_classes, (list, tuple)) and len(selected_classes) > 1 else 10

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
        'max_epochs': max_epochs,
        'stopping_mode': stopping_mode,
    }

    betas_experimented = []
    saved_csv_paths = []

    for beta in sorted(extended_beta_values):
        if checkpoint_data is not None and beta in completed_betas:
            print(f"\n--- Beta = {beta} (SKIPPED - already completed) ---")
            betas_experimented.append(beta)
            continue

        betas_experimented.append(beta)
        current_a0 = get_a0_for_beta(beta, a0)

        print(f"\n--- Beta = {beta} ---")
        print(f"Learning rate: {current_a0}")

        model = create_multiclass_model(
            dataset_type=dataset_type,
            n_hidden_layers=n_hidden_layers,
            width=width,
            device=device,
            num_classes=num_classes,
        )
        validate_model_output_dim(
            model=model,
            train_loader=train_loader,
            device=device,
            expected_num_classes=num_classes,
        )

        training_results = train_sgld_model(
            loss=loss,
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
            max_epochs=max_epochs,
            stopping_mode=stopping_mode,
        )

        (
            train_losses,
            test_losses,
            _,
            _,
            train_01_losses,
            test_01_losses,
            _,
            _,
            EMA_train_BCE_losses,
            EMA_test_BCE_losses,
            EMA_train_01_losses,
            EMA_test_01_losses,
            EMA_train_BCE_losses_sq,
            EMA_test_BCE_losses_sq,
            EMA_grad_norm,
            epoch,
        ) = training_results

        summary_idx = -50 if len(train_losses) >= 50 else 0
        list_train_BCE_losses.append(train_losses[summary_idx])
        list_test_BCE_losses.append(test_losses[summary_idx])
        list_train_01_losses.append(train_01_losses[summary_idx])
        list_test_01_losses.append(test_01_losses[summary_idx])
        list_EMA_train_BCE_losses.append(mean(EMA_train_BCE_losses))
        list_EMA_test_BCE_losses.append(mean(EMA_test_BCE_losses))
        list_EMA_train_01_losses.append(mean(EMA_train_01_losses))
        list_EMA_test_01_losses.append(mean(EMA_test_01_losses))
        list_EMA_grad_norm.append(mean(EMA_grad_norm) if len(EMA_grad_norm) > 0 else 0.0)
        list_num_epochs_per_beta.append(epoch)
        list_EMA_var_train_BCE_losses.append(mean(EMA_train_BCE_losses_sq) - mean(EMA_train_BCE_losses) ** 2)
        list_EMA_var_test_BCE_losses.append(mean(EMA_test_BCE_losses_sq) - mean(EMA_test_BCE_losses) ** 2)

        print(
            f"  Final - Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, "
            f"Train 0-1: {train_01_losses[-1]:.4f}, Test 0-1: {test_01_losses[-1]:.4f}"
        )

        completed_betas.append(beta)
        save_checkpoint(
            checkpoint_dir,
            seed,
            extended_beta_values,
            completed_betas,
            list_train_BCE_losses,
            list_test_BCE_losses,
            list_train_01_losses,
            list_test_01_losses,
            list_EMA_train_BCE_losses,
            list_EMA_test_BCE_losses,
            list_EMA_train_01_losses,
            list_EMA_test_01_losses,
            list_EMA_grad_norm,
            list_num_epochs_per_beta,
            list_EMA_var_train_BCE_losses,
            list_EMA_var_test_BCE_losses,
            sample_size=len(train_loader.dataset),
            config_dict=config_dict,
        )

        filename_prefix = "M" if dataset_type == 'mnist' else ("C" if dataset_type in ('cifar10', 'cifar100') else "S")
        filename_prefix += "R" if use_random_labels == 1 else ("C" if use_random_labels == 0 else "P")
        filename_prefix += f"L{n_hidden_layers}"
        filename_prefix += f"W{width}"
        if len(train_loader) == 1:
            filename_prefix += "ULA" if add_noise else "GD"
        else:
            filename_prefix += "SGLD" if add_noise else "SGD"
        filename_prefix += f"{len(train_loader.dataset) / 1000:.0f}k"
        filename_prefix += f"LR{str(current_a0).replace('.', '')}"
        filename_prefix += f"{str(loss).upper()}"
        filename_prefix += f"_S{seed}"
        filename_prefix += f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if test_mode:
            filename_prefix += "_TEST"

        summary_string = (
            "The multiclass LMC run used:\n"
            f"  - Device: {device}\n"
            f"  - Loss function: {loss}\n"
            f"  - l_max: {l_max}\n"
            f"  - Network architecture: {model.__class__.__name__}\n"
            f"  - Number of hidden layers: {n_hidden_layers}\n"
            f"  - Width of hidden layers: {width}\n"
            f"  - Dataset type: {dataset_type}\n"
            f"  - Dataset name: {dataset_name}\n"
            f"  - Selected classes: {selected_classes_str}\n"
            f"  - Random labels: {use_random_labels}\n"
            f"  - Training set size: {len(train_loader.dataset)}\n"
            f"  - Test set size: {len(test_loader.dataset)}\n"
            f"  - Minimum epochs: {min_steps}\n"
            f"  - Number of batches: {len(train_loader)}\n"
            f"  - Beta values: {sorted(betas_experimented)}\n"
            f"  - Learning rate (a0): {current_a0}\n"
            f"  - Learning rate decay (b): {b}\n"
            f"  - Gaussian prior sigma: {sigma_gauss_prior}\n"
            f"  - alpha_average: {alpha_average}\n"
            f"  - alpha_stop: {alpha_stop}\n"
            f"  - eta: {eta}\n"
            f"  - eps: {eps}\n"
            f"  - Seed: {seed}\n"
            f"  - Gradient norm: {list_EMA_grad_norm}\n"
            f"  - Number of epochs per beta: {list_num_epochs_per_beta}\n"
            f"  - Variance of EMA Train losses: {list_EMA_var_train_BCE_losses}\n"
            f"  - Variance of EMA Test losses: {list_EMA_var_test_BCE_losses}\n"
        )

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
            summary_string,
        )
        saved_csv_paths.append(csv_path)

    return saved_csv_paths
