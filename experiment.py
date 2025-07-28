import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from typing import Generator, Iterable


def create_synth_dataset(
    n_train: int = 50,
    n_test: int = 100,
    input_dim: int = 4,
    random_seed: int = None
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create the SYNTH dataset as described in the paper.
    
    The SYNTH dataset consists of:
    - 50 training data and 100 heldout data
    - Each input is a 4-dimensional vector sampled independently from a 
      zero-mean Gaussian distribution with an identity covariance matrix
    - The true classifier is linear
    - The norm of the separating hyperplane is sampled from a standard normal
    
    Args:
        n_train (int): Number of training samples (default: 50)
        n_test (int): Number of test samples (default: 100)
        input_dim (int): Dimensionality of input vectors (default: 4)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[TensorDataset, TensorDataset]: Training and test datasets
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Generate the separating hyperplane
    # Sample hyperplane direction uniformly from unit sphere
    w_direction = np.random.randn(input_dim)
    w_direction = w_direction / np.linalg.norm(w_direction)
    
    # Sample the norm from standard normal distribution
    w_norm = np.abs(np.random.randn())  # Taking absolute to ensure positive norm
    
    # Create the weight vector
    w = w_norm * w_direction
    
    # Generate training data
    X_train = np.random.randn(n_train, input_dim)  # Zero-mean Gaussian with identity covariance
    y_train_scores = X_train @ w  # Linear scores
    y_train = (y_train_scores > 0).astype(np.float32)  # Binary classification
    
    # Generate test data
    X_test = np.random.randn(n_test, input_dim)
    y_test_scores = X_test @ w
    y_test = (y_test_scores > 0).astype(np.float32)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset


def get_synth_dataloaders(
    batch_size: int = 10,  # SYNTH uses batch size 10
    random_seed: int = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for the SYNTH dataset.
    
    Args:
        batch_size (int): Batch size for DataLoaders (default: 10 for SYNTH)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test DataLoaders
    """
    train_dataset, test_dataset = create_synth_dataset(random_seed=random_seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader


class BoundedCrossEntropyLoss(nn.Module):
    """
    Bounded cross entropy loss for differential privacy.
    
    The loss function remaps probabilities p -> ψ(p) where:
    ψ(p) = e^(-ℓ_max) + (1 - 2*e^(-ℓ_max)) * p
    
    This maps [0,1] to [e^(-ℓ_max), 1-e^(-ℓ_max)], removing extreme probability values.
    The cross entropy loss becomes: g((p_1, ..., p_K), y) = -ln(ψ(p_y))
    
    As a result, the loss is bounded in the interval [0, ℓ_max].
    """
    
    def __init__(self, l_max: float = 4.0):
        """
        Initialize the bounded cross entropy loss.
        
        Args:
            l_max (float): Maximum loss value (default: 4.0)
        """
        super(BoundedCrossEntropyLoss, self).__init__()
        self.l_max = l_max
        self.epsilon = torch.exp(torch.tensor(-l_max))  # e^(-ℓ_max)
        
    def psi(self, p: torch.Tensor) -> torch.Tensor:
        """
        Apply the affine transformation ψ(p) = e^(-ℓ_max) + (1 - 2*e^(-ℓ_max)) * p
        
        Args:
            p (torch.Tensor): Input probabilities
            
        Returns:
            torch.Tensor: Transformed probabilities
        """
        return self.epsilon + (1 - 2 * self.epsilon) * p
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the bounded cross entropy loss.
        
        Args:
            logits (torch.Tensor): Raw model outputs (before softmax)
            targets (torch.Tensor): Target labels
            
        Returns:
            torch.Tensor: Bounded cross entropy loss
        """
        # Small epsilon to prevent log(0)
        eps = 1e-8
        
        # For binary classification, we need to handle the case differently
        if logits.shape[-1] == 1:
            # Binary case: convert to probabilities for both classes
            p_pos = torch.sigmoid(logits.squeeze(-1))
            p_neg = 1 - p_pos
            
            # Apply transformation
            bounded_p_pos = self.psi(p_pos)
            bounded_p_neg = self.psi(p_neg)
            
            # Clamp to prevent log(0)
            bounded_p_pos = torch.clamp(bounded_p_pos, min=eps, max=1-eps)
            bounded_p_neg = torch.clamp(bounded_p_neg, min=eps, max=1-eps)
            
            # Compute loss: -ln(ψ(p_y))
            targets_int = targets.long()
            loss = -torch.where(targets_int == 1, 
                              torch.log(bounded_p_pos), 
                              torch.log(bounded_p_neg))
        else:
            # Multi-class case
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Apply the transformation ψ
            bounded_probs = self.psi(probs)
            
            # Clamp to prevent log(0)
            bounded_probs = torch.clamp(bounded_probs, min=eps, max=1-eps)
            
            targets_int = targets.long()
            # Gather the probabilities for the target classes
            target_probs = bounded_probs.gather(1, targets_int.unsqueeze(1)).squeeze(1)
            
            # Compute loss: -ln(ψ(p_y))
            loss = -torch.log(target_probs)
        
        return loss.mean()

class ZeroOneLoss(nn.Module):
    """
    Zero-one loss for evaluating model performance.
    
    The zero-one loss counts the fraction of misclassified examples:
    L_0-1 = (1/n) * sum(I[y_pred != y_true])
    
    This is useful for evaluating actual classification performance
    separate from the optimization objective.
    """
    
    def __init__(self):
        super(ZeroOneLoss, self).__init__()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, dataset_type: str = 'synth') -> torch.Tensor:
        """
        Compute the zero-one loss.
        
        Args:
            logits (torch.Tensor): Raw model outputs (before softmax)
            targets (torch.Tensor): Target labels
            dataset_type (str): 'synth' or 'mnist' for proper prediction handling
            
        Returns:
            torch.Tensor: Zero-one loss (fraction of misclassified examples)
        """
        with torch.no_grad():
            if dataset_type == 'synth':
                # For SYNTH: single output, threshold at 0
                if logits.dim() == 2 and logits.shape[1] == 1:
                    predicted = (logits.squeeze() > 0).float()
                else:
                    # Handle case where logits might be 2D with 2 classes
                    predicted = torch.argmax(logits, dim=1).float()
            else:
                # For MNIST: multi-class, argmax
                predicted = torch.argmax(logits, dim=1).float()
            
            # Count misclassifications
            misclassified = (predicted != targets).float()
            zero_one_loss = misclassified.mean()
            
        return zero_one_loss
    
class SynthNN(nn.Module):
    """
    Neural network for the SYNTH dataset following SGLD specifications.
    Architecture: 1 hidden layer with 100 units, input=4, output=1 (binary classification)
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 100):
        super(SynthNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Single output for binary classification
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, raw logits
        return x


class MNISTNN(nn.Module):
    """
    Neural network for MNIST following SGLD specifications.
    Architecture: 3 layers with 600 units each, input=784, output=10
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 600, num_classes: int = 10):
        super(MNISTNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten for MNIST
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SGLD(optimizer.Optimizer):
    """Stochastic Gradient Langevin Dynamics (SGLD).
    An algorithm for Bayesian learning from large scale datasets. 

    Weight decay is specified in terms of the Gaussian prior's sigma.
    Inverse temperature (beta) controls the noise level in the SGLD updates.

    Welling and Teh, 2011. Bayesian Learning via Stochastiv Gradient Langevin 
    Dynamics. Paper link: https://bit.ly/3ngnyRA

    Args:
        params (Iterable): an iterable of `torch.Tensor`s or
            `dict`s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
        lr (float): learning rate. 
        sigma_gauss_prior (float, optional): Defaults to 0.1.
        beta (float, optional): Inverse temperature parameter. Defaults to 1.0.
        add_noise (bool, optional): Defaults to True. 
    
    Attributes:
        param_group (OptimizerParamGroup): Stores parameters in the param_group
            and stores a pointer to the OptimizerOptions. 
            docs: https://preview.tinyurl.com/2y272xmv

    Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of 
        dictionaries.
    """ 

    def __init__(self, params: Iterable, lr: float, 
                 sigma_gauss_prior: float = 0.1, beta: float = 1.0, add_noise: bool = True):
        if isinstance(sigma_gauss_prior, (complex)):
            if sigma_gauss_prior.imag != 0:
                raise ValueError(f"sigma_gauss_prior must be a real number.")

        weight_decay = 1 / (sigma_gauss_prior * sigma_gauss_prior)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta=beta, add_noise=add_noise)
        super(SGLD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Updates neural network parameters. Called once the gradients are 
        computed using loss.backward(). Performs a single parameter update.

        Args:
            closure (callable, optional): A closure that reevaluates the 
                model and returns the loss. 
        This function should not modify the gradient field of the parameters, 
            i.e. `parameter.grad`. 
        """
        loss = None
        def params() -> Generator:
            for param_group in self.param_groups:
                weight_decay = param_group['weight_decay']
                beta = param_group['beta']
                for param in param_group['params']:
                    yield param, weight_decay, beta, param_group
        
        # 'loss' gets updated from the following loop (under the hood)
        for param, weight_decay, beta, param_group in params():
            if param.grad is None:
                continue
            gradient = param.grad.data
            if weight_decay != 0:
                gradient.add_(param.data, alpha=weight_decay)
            if param_group['add_noise']:
                # Langevin noise scaled by inverse temperature: N(0, 1) / sqrt(lr * beta)
                # Handle beta = 0 case to avoid division by zero
                if beta > 0:
                    langevin_noise = param.data.new(param.data.size()).normal_(
                        mean=0, std=1) / np.sqrt(param_group['lr'] * beta)
                    param.data.add_(0.5*gradient + langevin_noise, alpha=-param_group['lr'])
                else:
                    # When beta = 0,pure noise
                    param.data.add_(0.5*gradient, alpha=-param_group['lr'])
            else: # don't add noise
                param.data.add_(0.5*gradient, alpha=-param_group['lr'])
        return loss


def train_sgld_model(model, train_loader, test_loader, num_epochs: int = 100, 
                     a0: float = 1e-3, b: float = 0.5, sigma_gauss_prior: float = 0.1, 
                     beta: float = 1.0, device: str = 'cpu', dataset_type: str = 'synth'):
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
        
    Returns:
        Tuple containing: (train_losses, test_losses, train_accuracies, test_accuracies, 
                          train_zero_one_losses, test_zero_one_losses, learning_rates)
    """
    model = model.to(device)
    criterion = BoundedCrossEntropyLoss(l_max=4.0)
    zero_one_criterion = ZeroOneLoss()
    
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
    print(f"Dataset type: {dataset_type}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_zero_one_total = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # Handle different output shapes for SYNTH vs MNIST
            if dataset_type == 'synth':
                # For SYNTH: single output, convert to binary classification
                outputs_binary = torch.cat([1 - torch.sigmoid(outputs), torch.sigmoid(outputs)], dim=1)
                loss = criterion(outputs_binary, batch_y)
                # Training accuracy for SYNTH
                predicted = (outputs.squeeze() > 0).float()
                # Zero-one loss for training
                zero_one_loss = zero_one_criterion(outputs, batch_y, dataset_type)
            else:
                # For MNIST: multi-class classification
                loss = criterion(outputs, batch_y)
                # Training accuracy for MNIST
                predicted = torch.argmax(outputs, dim=1)
                # Zero-one loss for training
                zero_one_loss = zero_one_criterion(outputs, batch_y, dataset_type)
            
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
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
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                
                # Calculate test loss
                if dataset_type == 'synth':
                    outputs_binary = torch.cat([1 - torch.sigmoid(outputs), torch.sigmoid(outputs)], dim=1)
                    loss = criterion(outputs_binary, batch_y)
                    # Test accuracy for SYNTH
                    predicted = (outputs.squeeze() > 0).float()
                    # Zero-one loss for test
                    zero_one_loss = zero_one_criterion(outputs, batch_y, dataset_type)
                else:
                    loss = criterion(outputs, batch_y)
                    # Test accuracy for MNIST
                    predicted = torch.argmax(outputs, dim=1)
                    # Zero-one loss for test
                    zero_one_loss = zero_one_criterion(outputs, batch_y, dataset_type)
                
                test_loss_total += loss.item()
                test_zero_one_total += zero_one_loss.item()
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        avg_test_loss = test_loss_total / len(test_loader)
        avg_test_zero_one = test_zero_one_total / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        test_losses.append(avg_test_loss)
        test_zero_one_losses.append(avg_test_zero_one)
        test_accuracies.append(test_accuracy)
        
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, '
                  f'Train 0-1: {avg_train_zero_one:.4f}, Test 0-1: {avg_test_zero_one:.4f}, '
                #   f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, '
                  f'LR: {current_lr:.2e}')
    
    return train_losses, test_losses, train_accuracies, test_accuracies, train_zero_one_losses, test_zero_one_losses, learning_rates

def run_beta_experiments(beta_values, num_repetitions=50, num_epochs=10000, 
                         a0=1e-1, b=0.5, sigma_gauss_prior=1000000, 
                         device='cpu', dataset_type='synth'):
    """
    Run experiments across different beta values with multiple repetitions.
    
    Args:
        beta_values: List of beta (inverse temperature) values to test
        num_repetitions: Number of repetitions for each beta value
        num_epochs: Number of training epochs per run
        a0: Initial learning rate
        b: Learning rate decay parameter
        sigma_gauss_prior: Gaussian prior sigma
        device: Device to run on
        dataset_type: 'synth' or 'mnist'
        
    Returns:
        Dictionary containing results for each beta value
    """
    results = {}
    
    for beta in beta_values:
        print(f"\n{'='*60}")
        print(f"Running experiments for beta = {beta}")
        print(f"{'='*60}")
        
        # Storage for this beta value
        final_train_losses = []
        final_test_losses = []
        final_train_01_losses = []
        final_test_01_losses = []
        
        for rep in range(num_repetitions):
            print(f"Repetition {rep+1}/{num_repetitions} for beta = {beta}")
            
            # Create fresh dataset and model for each repetition
            train_loader, test_loader = get_synth_dataloaders(
                batch_size=10, 
                random_seed=rep  # Different seed for each repetition
            )
            
            # Create fresh model
            model = SynthNN(input_dim=4, hidden_dim=100)
            
            # Train the model
            train_losses, test_losses, _, _, train_01_losses, test_01_losses, _ = train_sgld_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                a0=a0,
                b=b,
                sigma_gauss_prior=sigma_gauss_prior,
                beta=beta,
                device=device,
                dataset_type=dataset_type
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
    
    return results


def compute_generalization_bound(beta_values, results, loss_type='bce', delta=0.05, delta_prime=0.05):
    """
    Compute the generalization bound for each beta value cumulatively.
    
    Args:
        beta_values: List of beta values (sorted)
        results: Dictionary containing experimental results
        loss_type: 'bce' for bounded cross-entropy or 'zero_one' for zero-one loss
        delta: Confidence parameter for main bound (default: 0.05)
        delta_prime: Confidence parameter for integral bound (default: 0.05)
        
    Returns:
        Dictionary containing bounds for each beta value
    """
    bounds = {}
    n = 50  # Training set size for SYNTH dataset
    
    # Choose the appropriate loss type
    if loss_type == 'bce':
        train_key = 'train_bce_mean'
        raw_key = 'raw_train_bce'
    else:  # zero_one
        train_key = 'train_01_mean'
        raw_key = 'raw_train_01'
    
    for i, current_beta in enumerate(beta_values):
        # Get current beta results
        current_results = results[current_beta]
        empirical_loss = current_results[train_key]  # L̂(h, x)
        M = len(current_results[raw_key])  # Number of repetitions
        
        # Debug: Check for NaN in empirical loss
        if np.isnan(empirical_loss):
            print(f"WARNING: NaN empirical_loss for beta={current_beta}, loss_type={loss_type}")
            empirical_loss = 0.0  # Use 0 as fallback
        
        # Compute integral bound using cumulative betas (including current beta)
        # The integral is from 0 to current_beta, so we need to handle the case where
        # beta_values doesn't start from 0
        integral_bound = 0.0
        variance_term = 0.0
        
        # Always start from 0 and integrate up to current_beta
        prev_beta = 0.0
        
        for j in range(i + 1):
            beta_k = beta_values[j]
            
            # Calculate beta difference from previous point
            beta_diff = beta_k - prev_beta
            
            # Average loss for this beta
            avg_loss_k = results[beta_k][train_key]
            
            # Check for NaN
            if np.isnan(avg_loss_k):
                print(f"WARNING: NaN avg_loss_k for beta={beta_k}, loss_type={loss_type}")
                avg_loss_k = 0.0
            
            # Add to integral approximation
            integral_bound += beta_diff * avg_loss_k
            variance_term += beta_diff ** 2
            
            prev_beta = beta_k
        
        # Add the variance/confidence term for the integral
        integral_confidence = np.sqrt((variance_term + np.log(1 / delta_prime)) / M)
        integral_upper_bound = integral_bound + integral_confidence
        
        # Compute the main generalization bound
        # Inner term: integral - β * L̂(h,x) + ln(2√n/δ)
        inner_term = integral_upper_bound - current_beta * empirical_loss + np.log(2 * np.sqrt(n) / delta)
        
        # Ensure inner_term is positive for square root
        inner_term = max(inner_term, 1e-10)
        
        # Square root term
        sqrt_term = np.sqrt((2 * empirical_loss * inner_term) / n)
        
        # Linear term
        linear_term = (2 * inner_term) / n
        
        # Total bound: L(h) - L̂(h,x) ≤ sqrt_term + linear_term
        generalization_bound = sqrt_term + linear_term
        
        # Debug: Check for NaN in final bound
        if np.isnan(generalization_bound):
            print(f"WARNING: NaN generalization_bound for beta={current_beta}, loss_type={loss_type}")
            generalization_bound = 1.0  # Use fallback value
        
        bounds[current_beta] = {
            'empirical_loss': empirical_loss,
            'integral_bound': integral_bound,
            'integral_upper_bound': integral_upper_bound,
            'generalization_bound': generalization_bound,
            'predicted_test_loss': empirical_loss + generalization_bound,  # L̂(h,x) + bound
            'sqrt_term': sqrt_term,
            'linear_term': linear_term,
            'inner_term': inner_term
        }
    
    return bounds


def compute_generalization_errors(beta_values, results):
    """
    Compute the actual generalization errors for both BCE and zero-one loss.
    
    Args:
        beta_values: List of beta values (sorted)
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


def plot_beta_results(results):
    """
    Plot the generalization errors with confidence intervals, generalization bounds, and individual train/test errors across different beta values.
    """
    beta_values = sorted(results.keys())
    
    # Compute generalization errors
    gen_errors = compute_generalization_errors(beta_values, results)
    
    # Extract generalization errors and their standard deviations
    bce_gen_errors = [gen_errors[beta]['bce_gen_error'] for beta in beta_values]
    bce_gen_error_stds = [gen_errors[beta]['bce_gen_error_std'] for beta in beta_values]
    zero_one_gen_errors = [gen_errors[beta]['zero_one_gen_error'] for beta in beta_values]
    zero_one_gen_error_stds = [gen_errors[beta]['zero_one_gen_error_std'] for beta in beta_values]
    
    # Extract individual train/test errors
    train_bce_means = [results[beta]['train_bce_mean'] for beta in beta_values]
    test_bce_means = [results[beta]['test_bce_mean'] for beta in beta_values]
    train_bce_stds = [results[beta]['train_bce_std'] for beta in beta_values]
    test_bce_stds = [results[beta]['test_bce_std'] for beta in beta_values]
    
    train_01_means = [results[beta]['train_01_mean'] for beta in beta_values]
    test_01_means = [results[beta]['test_01_mean'] for beta in beta_values]
    train_01_stds = [results[beta]['train_01_std'] for beta in beta_values]
    test_01_stds = [results[beta]['test_01_std'] for beta in beta_values]
    
    # Compute generalization bounds
    bounds = compute_generalization_bound(beta_values, results, loss_type='bce')
    theoretical_bounds = [bounds[beta]['generalization_bound'] for beta in beta_values]
    
    # Compute zero-one specific bounds
    zero_one_bounds = compute_generalization_bound(beta_values, results, loss_type='zero_one')
    zero_one_theoretical_bounds = [zero_one_bounds[beta]['generalization_bound'] for beta in beta_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define consistent colors for the same line types across both plots
    train_color = 'blue'
    test_color = 'orange'
    gen_error_color = 'green'
    bound_color = 'red'
    
    # BCE: Train/test losses AND generalization error in the same plot
    ax1.errorbar(beta_values, train_bce_means, yerr=train_bce_stds, 
                 fmt='o-', label='Train BCE (mean ± std)', linewidth=2, markersize=6, capsize=5, color=train_color)
    ax1.errorbar(beta_values, test_bce_means, yerr=test_bce_stds, 
                 fmt='s-', label='Test BCE (mean ± std)', linewidth=2, markersize=6, capsize=5, color=test_color)
    
    # Add generalization error on the same plot
    ax1.errorbar(beta_values, bce_gen_errors, yerr=bce_gen_error_stds, 
                 fmt='^-', label='BCE Gen. Error (mean ± std)', linewidth=2, markersize=6, capsize=5, color=gen_error_color)
    
    # Add theoretical generalization bound
    ax1.plot(beta_values, theoretical_bounds, 'v-', 
             label='Theoretical Upper Bound', linewidth=2, markersize=6, color=bound_color)
    
    ax1.set_xlabel('Beta (Inverse Temperature)')
    ax1.set_ylabel('Loss Value / Generalization Error')
    ax1.set_title('BCE: Train/Test Losses, Gen. Error & Bound')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Handle beta=0 case with symlog scale
    if 0 in beta_values:
        ax1.set_xscale('symlog', linthresh=1)
    else:
        ax1.set_xscale('log')
    
    # Zero-one: Train/test losses AND generalization error in the same plot
    # Use the SAME colors for the same line types
    ax2.errorbar(beta_values, train_01_means, yerr=train_01_stds, 
                 fmt='o-', label='Train 0-1 (mean ± std)', linewidth=2, markersize=6, capsize=5, color=train_color)
    ax2.errorbar(beta_values, test_01_means, yerr=test_01_stds, 
                 fmt='s-', label='Test 0-1 (mean ± std)', linewidth=2, markersize=6, capsize=5, color=test_color)
    
    # Add generalization error on the same plot
    ax2.errorbar(beta_values, zero_one_gen_errors, yerr=zero_one_gen_error_stds, 
                 fmt='^-', label='Zero-One Gen. Error (mean ± std)', linewidth=2, markersize=6, capsize=5, color=gen_error_color)
    
    # Add theoretical generalization bound for zero-one loss
    ax2.plot(beta_values, zero_one_theoretical_bounds, 'v-', 
             label='Theoretical Upper Bound', linewidth=2, markersize=6, color=bound_color)
    
    ax2.set_xlabel('Beta (Inverse Temperature)')
    ax2.set_ylabel('Loss Value / Generalization Error')
    ax2.set_title('Zero-One: Train/Test Losses & Gen. Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Handle beta=0 case with symlog scale
    if 0 in beta_values:
        ax2.set_xscale('symlog', linthresh=1)
    else:
        ax2.set_xscale('log')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/sgld_beta_experiments.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'results/sgld_beta_experiments.png'")
    
    plt.show()


def save_results_to_file(results, filename='results/sgld_beta_experiments.txt'):
    """
    Save the experimental results to a text file.
    """
    # Compute generalization bounds and errors
    beta_values = sorted(results.keys())
    bounds = compute_generalization_bound(beta_values, results, loss_type='bce')
    zero_one_bounds = compute_generalization_bound(beta_values, results, loss_type='zero_one')
    gen_errors = compute_generalization_errors(beta_values, results)
    
    with open(filename, 'w') as f:
        f.write("SGLD Beta Experiments Results\n")
        f.write("="*80 + "\n\n")
        
        # Summary table
        f.write("Summary Table:\n")
        f.write("Beta\tTrain_BCE\tTest_BCE\tBCE_GenErr\tBCE_Bound\tTrain_01\tTest_01\tZO_GenErr\tZO_Bound\n")
        for beta in beta_values:
            f.write(f"{beta}\t{results[beta]['train_bce_mean']:.4f}\t\t{results[beta]['test_bce_mean']:.4f}\t\t")
            f.write(f"{gen_errors[beta]['bce_gen_error']:.4f}\t\t{bounds[beta]['generalization_bound']:.4f}\t\t")
            f.write(f"{results[beta]['train_01_mean']:.4f}\t\t{results[beta]['test_01_mean']:.4f}\t\t")
            f.write(f"{gen_errors[beta]['zero_one_gen_error']:.4f}\t\t{zero_one_bounds[beta]['generalization_bound']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed results
        for beta in beta_values:
            f.write(f"Beta = {beta}:\n")
            f.write(f"  Bounded Cross-Entropy Loss:\n")
            f.write(f"    Train: mean={results[beta]['train_bce_mean']:.4f}, var={results[beta]['train_bce_var']:.6f}, std={results[beta]['train_bce_std']:.4f}\n")
            f.write(f"    Test:  mean={results[beta]['test_bce_mean']:.4f}, var={results[beta]['test_bce_var']:.6f}, std={results[beta]['test_bce_std']:.4f}\n")
            f.write(f"    Generalization Error: {gen_errors[beta]['bce_gen_error']:.4f} ± {gen_errors[beta]['bce_gen_error_std']:.4f}\n")
            f.write(f"  Theoretical Generalization Bound:\n")
            f.write(f"    Upper Bound: {bounds[beta]['generalization_bound']:.4f}\n")
            f.write(f"    Bound Tightness: {bounds[beta]['generalization_bound'] - gen_errors[beta]['bce_gen_error']:.4f}\n")
            f.write(f"    Sqrt Term: {bounds[beta]['sqrt_term']:.4f}, Linear Term: {bounds[beta]['linear_term']:.4f}\n")
            f.write(f"  Zero-One Loss:\n")
            f.write(f"    Train: mean={results[beta]['train_01_mean']:.4f}, var={results[beta]['train_01_var']:.6f}, std={results[beta]['train_01_std']:.4f}\n")
            f.write(f"    Test:  mean={results[beta]['test_01_mean']:.4f}, var={results[beta]['test_01_var']:.6f}, std={results[beta]['test_01_std']:.4f}\n")
            f.write(f"    Generalization Error: {gen_errors[beta]['zero_one_gen_error']:.4f} ± {gen_errors[beta]['zero_one_gen_error_std']:.4f}\n")
            f.write(f"  Zero-One Theoretical Generalization Bound:\n")
            f.write(f"    Upper Bound: {zero_one_bounds[beta]['generalization_bound']:.4f}\n")
            f.write(f"    Bound Tightness: {zero_one_bounds[beta]['generalization_bound'] - gen_errors[beta]['zero_one_gen_error']:.4f}\n")
            f.write(f"    Sqrt Term: {zero_one_bounds[beta]['sqrt_term']:.4f}, Linear Term: {zero_one_bounds[beta]['linear_term']:.4f}\n")
            f.write("\n")
    
    print(f"Results saved to {filename}")


# Test mode flag - set to False for full experiment
TEST_MODE = False

# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test the dataset creation (quick check)
    train_dataset, test_dataset = create_synth_dataset(random_seed=42)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Define beta values to test
    if TEST_MODE:
        print("\n" + "="*50)
        print("RUNNING IN TEST MODE")
        print("For full experiment, set TEST_MODE = False")
        print("="*50)
        beta_values = [1, 10, 50, 200]  # Reduced set for testing
        num_repetitions = 30  # Reduced for testing
        num_epochs = 100  # Reduced for testing
    else:
        beta_values = [1, 10, 30, 50, 70, 100, 200]  # Full experiment
        num_repetitions = 30  # Full experiment
        num_epochs = 10000  # Full experiment
    
    print(f"\n{'='*70}")
    print(f"SGLD BETA EXPERIMENTS")
    print(f"Beta values: {beta_values}")
    print(f"Repetitions per beta: {num_repetitions}")
    print(f"Epochs per training: {num_epochs}")
    print(f"{'='*70}")
    
    # Run the experiment
    results = run_beta_experiments(
        beta_values=beta_values,
        num_repetitions=num_repetitions,
        num_epochs=num_epochs,
        a0=1e-1,
        b=0.5,
        sigma_gauss_prior=10,
        device=device,
        dataset_type='synth'
    )
    
    # Print final summary
    print(f"\n{'='*90}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*90}")
    print("Beta\tTrain_BCE\tTest_BCE\tBCE_GenErr\tBCE_Bound\tBound_Gap\tTrain_01\tTest_01\tZO_GenErr\tZO_Bound\tZO_Gap")
    print("-" * 110)
    
    # Compute bounds and generalization errors for summary
    # Use the actual beta values from results (which may include beta=0)
    actual_beta_values = sorted(results.keys())
    bounds = compute_generalization_bound(actual_beta_values, results, loss_type='bce')
    zero_one_bounds = compute_generalization_bound(actual_beta_values, results, loss_type='zero_one')
    gen_errors = compute_generalization_errors(actual_beta_values, results)
    
    for beta in sorted(results.keys()):
        train_bce = results[beta]['train_bce_mean']
        test_bce = results[beta]['test_bce_mean']
        bce_gen_error = gen_errors[beta]['bce_gen_error']
        theo_bound = bounds[beta]['generalization_bound']
        
        # Bound gap: how much larger the theoretical bound is compared to actual generalization error
        # Positive means bound is valid, negative means bound is violated
        bound_gap = theo_bound - bce_gen_error
        
        train_01 = results[beta]['train_01_mean']
        test_01 = results[beta]['test_01_mean']
        zo_gen_error = gen_errors[beta]['zero_one_gen_error']
        zo_theo_bound = zero_one_bounds[beta]['generalization_bound']
        zo_bound_gap = zo_theo_bound - zo_gen_error
        
        print(f"{beta}\t{train_bce:.4f}\t\t{test_bce:.4f}\t\t{bce_gen_error:.4f}\t\t{theo_bound:.4f}\t\t{bound_gap:.4f}\t\t{train_01:.4f}\t\t{test_01:.4f}\t\t{zo_gen_error:.4f}\t\t{zo_theo_bound:.4f}\t\t{zo_bound_gap:.4f}")
    
    print(f"\nGeneralization Error = Test Loss - Train Loss")
    print(f"Bound Gap = Theoretical Bound - Actual Generalization Error")
    print(f"  > 0: Bound is valid (theoretical bound > actual generalization error)")
    print(f"  ≈ 0: Bound is tight")
    print(f"  < 0: Bound is violated (should not happen with high probability)")
    
    # Define experimental parameters for saving/plotting
    # Use original beta values for filename generation and plotting (excluding auto-added beta=0)
    experiment_params = {
        'beta_values': beta_values,
        'num_repetitions': num_repetitions,
        'num_epochs': num_epochs,
        'a0': 1e-1,
        'sigma_gauss_prior': 10,
        'dataset_type': 'synth'
    }
    
    # Save results to file with descriptive filename
    save_results_to_file(results, **experiment_params)
    
    # Plot the results with descriptive filename
    plot_beta_results(results, **experiment_params)
    
    # Get the generated filenames for display
    from bounds import generate_filename
    results_filename = generate_filename(file_type='results', extension='txt', **experiment_params)
    plot_filename = generate_filename(file_type='plot', extension='png', **experiment_params)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED!")
    print(f"Files generated:")
    print(f"  - results/{results_filename} (numerical results)")
    print(f"  - results/{plot_filename} (plots)")
    if TEST_MODE:
        print(f"\nTo run the full experiment:")
        print(f"  1. Set TEST_MODE = False in the script")
        print(f"  2. Re-run the script")
        print(f"  3. The full experiment will take much longer!")
    print(f"{'='*70}")
    
    # Test the bounded cross entropy loss directly (demonstration)
    print("\n" + "="*50)
    print("Testing Bounded Cross Entropy Loss (Quick Demo)")
    print("="*50)
    
    criterion = BoundedCrossEntropyLoss(l_max=4.0)
    
    # Test with single output (SYNTH style)
    test_logits_single = torch.randn(5, 1)  # 5 samples, 1 output
    test_targets = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)
    
    bounded_loss_single = criterion(test_logits_single, test_targets)
    print(f"Bounded CE Loss (single output): {bounded_loss_single.item():.4f}")
    print(f"Loss bounded in [0, {criterion.l_max}]: {0 <= bounded_loss_single.item() <= criterion.l_max}")


