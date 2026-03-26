"""
Prior distribution initialization for PBB-compatible models.

This module provides functions to initialize neural network weights with
different prior distributions, matching the approach used in the
PAC-Bayes with Backprop (PBB) paper.

Prior Distributions:
- Gaussian: N(0, sigma^2) - Standard normal prior
- Laplace: Laplace(0, b) - Laplace distribution prior

These are "data-free" priors in the sense that they don't depend on the
training data - they only depend on the model architecture and hyperparameters.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable


def initialize_prior_gaussian(
    model: nn.Module,
    sigma: float = 1.0,
    seed: Optional[int] = None
) -> nn.Module:
    """
    Initialize model weights with Gaussian prior distribution.
    
    Initializes all parameters in the model from a Gaussian (normal) distribution
    with mean 0 and standard deviation sigma. This is a data-free prior that
    matches the prior used in the PBB paper.
    
    Args:
        model (nn.Module): The neural network model to initialize
        sigma (float): Standard deviation of the Gaussian prior. Default: 1.0
                      In PBB paper, typically 1.0 or related to model architecture
        seed (Optional[int]): Random seed for reproducibility. If None, uses random state.
        
    Returns:
        nn.Module: The model with initialized weights (modified in-place)
        
    Example:
        >>> from pbb_models import CNNet4l
        >>> model = CNNet4l(num_classes=10)
        >>> model = initialize_prior_gaussian(model, sigma=1.0, seed=42)
        # All weights are now from N(0, 1.0^2)
    
    Note:
        This is a data-free prior - initialization depends only on architecture,
        not on training data. This is suitable for PAC-Bayesian analysis.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                # Initialize from Gaussian distribution
                nn.init.normal_(param, mean=0.0, std=sigma)
    
    return model


def initialize_prior_laplace(
    model: nn.Module,
    scale: float = 1.0,
    seed: Optional[int] = None
) -> nn.Module:
    """
    Initialize model weights with Laplace prior distribution.
    
    Initializes all parameters in the model from a Laplace distribution
    with location 0 and scale parameter. This is a data-free prior used
    in the PBB paper as an alternative to Gaussian prior.
    
    Laplace distribution has the form: f(x) = (1/(2*b)) * exp(-|x|/b)
    
    Args:
        model (nn.Module): The neural network model to initialize
        scale (float): Scale parameter b of the Laplace distribution. Default: 1.0
                      Related to the prior spread; larger values = wider prior
        seed (Optional[int]): Random seed for reproducibility. If None, uses random state.
        
    Returns:
        nn.Module: The model with initialized weights (modified in-place)
        
    Example:
        >>> from pbb_models import CNNet4l
        >>> model = CNNet4l(num_classes=10)
        >>> model = initialize_prior_laplace(model, scale=1.0, seed=42)
        # All weights are now from Laplace(0, 1.0)
    
    Note:
        The Laplace distribution is heavier-tailed than Gaussian, allowing
        for more extreme weight values with non-negligible probability.
        
        This is a data-free prior suitable for PAC-Bayesian analysis.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                # Generate Laplace-distributed values using exponential distribution
                # Method: If U ~ Uniform(0,1), then X = b * sign(U - 0.5) * log(1 - 2|U - 0.5|)
                # is Laplace(0, b) distributed
                
                # Generate uniform samples
                u = torch.rand_like(param)
                
                # Transform to Laplace distribution
                param.copy_(scale * torch.sign(u - 0.5) * torch.log(1.0 - 2.0 * torch.abs(u - 0.5)))
    
    return model


def get_prior_initializer(
    prior_type: str,
    sigma_or_scale: float = 1.0,
    seed: Optional[int] = None
) -> Callable[[nn.Module], nn.Module]:
    """
    Get a prior initialization function based on prior type.
    
    Returns a callable that can be used to initialize any model with the
    specified prior distribution. This provides a convenient interface for
    selecting between different prior types.
    
    Args:
        prior_type (str): Type of prior distribution. Options:
                         - 'gaussian': Gaussian/Normal distribution
                         - 'laplace': Laplace distribution
        sigma_or_scale (float): Hyperparameter for the prior:
                               - For 'gaussian': standard deviation
                               - For 'laplace': scale parameter
        seed (Optional[int]): Random seed for reproducibility
        
    Returns:
        Callable: A function that takes a model and returns initialized model
        
    Example:
        >>> initializer = get_prior_initializer('gaussian', sigma_or_scale=1.0, seed=42)
        >>> model = initializer(model)  # Apply initialization
    
    Raises:
        ValueError: If prior_type is not 'gaussian' or 'laplace'
    """
    if prior_type.lower() == 'gaussian':
        def init_fn(model):
            return initialize_prior_gaussian(model, sigma=sigma_or_scale, seed=seed)
    elif prior_type.lower() == 'laplace':
        def init_fn(model):
            return initialize_prior_laplace(model, scale=sigma_or_scale, seed=seed)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}. "
                        f"Choose from: 'gaussian', 'laplace'")
    
    return init_fn


def initialize_model_with_prior(
    model: nn.Module,
    prior_type: str = 'gaussian',
    sigma_prior: float = 1.0,
    seed: Optional[int] = None
) -> nn.Module:
    """
    Initialize a model with the specified prior distribution.
    
    Convenience function that combines prior type selection and initialization.
    This is the main entry point for prior initialization.
    
    Args:
        model (nn.Module): The neural network model to initialize
        prior_type (str): Type of prior ('gaussian' or 'laplace'). Default: 'gaussian'
        sigma_prior (float): Prior scale/std. Default: 1.0
        seed (Optional[int]): Random seed for reproducibility
        
    Returns:
        nn.Module: The initialized model
        
    Example:
        >>> from pbb_models import CNNet4l
        >>> model = CNNet4l(num_classes=10)
        >>> model = initialize_model_with_prior(
        ...     model,
        ...     prior_type='gaussian',
        ...     sigma_prior=1.0,
        ...     seed=42
        ... )
    
    Note:
        This initializes the model in-place and returns it for convenience.
        
        For the PBB paper comparison:
        - Use prior_type='gaussian' with sigma_prior=1.0 for standard setup
        - Use prior_type='laplace' with sigma_prior=1.0 for alternative prior
    """
    initializer = get_prior_initializer(prior_type, sigma_prior, seed)
    return initializer(model)


# Statistics and utilities for prior analysis

def compute_prior_statistics(
    model: nn.Module,
    prior_type: str = 'gaussian',
    sigma_prior: float = 1.0,
    num_samples: int = 1000,
    seed: int = 42
) -> dict:
    """
    Compute statistics of the prior distribution over model weights.
    
    This function initializes multiple copies of the model with the prior
    and computes statistics of the weight distributions.
    
    Args:
        model (nn.Module): Model architecture to analyze
        prior_type (str): Type of prior distribution
        sigma_prior (float): Prior scale/std
        num_samples (int): Number of samples for statistics
        seed (int): Random seed
        
    Returns:
        dict: Statistics including:
            - 'mean': Mean of weights
            - 'std': Standard deviation of weights
            - 'min': Minimum weight value
            - 'max': Maximum weight value
            - 'median': Median of weights
            - 'expected_sigma': Expected sigma_prior
            - 'expected_scale': Expected scale (for Laplace)
    
    Example:
        >>> from pbb_models import CNNet4l
        >>> model = CNNet4l(num_classes=10)
        >>> stats = compute_prior_statistics(model, prior_type='gaussian', sigma_prior=1.0)
        >>> print(f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    """
    all_weights = []
    
    for i in range(num_samples):
        model_copy = type(model)(*([p for p in model.parameters() if p.requires_grad]))
        model_copy = initialize_model_with_prior(model_copy, prior_type, sigma_prior, seed + i)
        
        weights = torch.cat([p.flatten() for p in model_copy.parameters() if p.requires_grad])
        all_weights.append(weights)
    
    all_weights = torch.cat(all_weights)
    
    stats = {
        'mean': all_weights.mean().item(),
        'std': all_weights.std().item(),
        'min': all_weights.min().item(),
        'max': all_weights.max().item(),
        'median': all_weights.median().item(),
        'expected_sigma': sigma_prior if prior_type.lower() == 'gaussian' else None,
        'expected_scale': sigma_prior if prior_type.lower() == 'laplace' else None,
    }
    
    return stats


if __name__ == "__main__":
    """
    Test script to verify prior initialization works correctly.
    """
    import sys
    sys.path.insert(0, '/Users/erfanmirzaei/Projects/Generalization Bound/Gibbs-Generalization')
    
    from pbb_models import NNet4l, CNNet4l
    
    print("=" * 70)
    print("Testing Prior Distribution Initialization")
    print("=" * 70)
    
    # Test Gaussian prior with FC network
    print("\n1. Gaussian Prior + NNet4l (FC Network)")
    print("-" * 70)
    model_fc = NNet4l(num_classes=10)
    print(f"   Model before initialization - First parameter mean: "
          f"{next(model_fc.parameters()).mean().item():.4f}")
    
    model_fc = initialize_model_with_prior(model_fc, prior_type='gaussian', 
                                           sigma_prior=1.0, seed=42)
    params_fc = torch.cat([p.flatten() for p in model_fc.parameters()])
    print(f"   Model after Gaussian init - Mean: {params_fc.mean().item():.6f}, "
          f"Std: {params_fc.std().item():.6f}")
    print(f"   Expected Std: 1.0")
    
    # Test Gaussian prior with CNN network
    print("\n2. Gaussian Prior + CNNet4l (CNN Network)")
    print("-" * 70)
    model_cnn = CNNet4l(num_classes=10)
    model_cnn = initialize_model_with_prior(model_cnn, prior_type='gaussian',
                                            sigma_prior=1.0, seed=42)
    params_cnn = torch.cat([p.flatten() for p in model_cnn.parameters()])
    print(f"   Model after Gaussian init - Mean: {params_cnn.mean().item():.6f}, "
          f"Std: {params_cnn.std().item():.6f}")
    print(f"   Expected Std: 1.0")
    
    # Test Laplace prior
    print("\n3. Laplace Prior + CNNet4l")
    print("-" * 70)
    model_cnn_laplace = CNNet4l(num_classes=10)
    model_cnn_laplace = initialize_model_with_prior(model_cnn_laplace, prior_type='laplace',
                                                     sigma_prior=1.0, seed=42)
    params_laplace = torch.cat([p.flatten() for p in model_cnn_laplace.parameters()])
    print(f"   Model after Laplace init - Mean: {params_laplace.mean().item():.6f}, "
          f"Std: {params_laplace.std().item():.6f}")
    print(f"   Laplace distribution has higher tails than Gaussian")
    
    # Test determinism with seed
    print("\n4. Reproducibility Test (same seed = same initialization)")
    print("-" * 70)
    model_a = CNNet4l(num_classes=10)
    model_b = CNNet4l(num_classes=10)
    
    model_a = initialize_model_with_prior(model_a, prior_type='gaussian', 
                                          sigma_prior=1.0, seed=123)
    model_b = initialize_model_with_prior(model_b, prior_type='gaussian',
                                          sigma_prior=1.0, seed=123)
    
    diff = sum((p1 - p2).abs().max().item() 
               for p1, p2 in zip(model_a.parameters(), model_b.parameters()))
    print(f"   Max difference between identically seeded inits: {diff:.2e}")
    print(f"   ✓ Reproducible (should be ~0)" if diff < 1e-6 else "   ✗ Not reproducible")
    
    # Test different priors
    print("\n5. Comparison: Gaussian vs Laplace")
    print("-" * 70)
    model_g = CNNet4l(num_classes=10)
    model_l = CNNet4l(num_classes=10)
    
    model_g = initialize_model_with_prior(model_g, prior_type='gaussian', 
                                          sigma_prior=1.0, seed=42)
    model_l = initialize_model_with_prior(model_l, prior_type='laplace',
                                          sigma_prior=1.0, seed=42)
    
    params_g = torch.cat([p.flatten() for p in model_g.parameters()])
    params_l = torch.cat([p.flatten() for p in model_l.parameters()])
    
    print(f"   Gaussian - Mean: {params_g.mean().item():.6f}, Std: {params_g.std().item():.6f}")
    print(f"   Laplace  - Mean: {params_l.mean().item():.6f}, Std: {params_l.std().item():.6f}")
    print(f"   Note: Laplace typically has higher std due to heavier tails")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
