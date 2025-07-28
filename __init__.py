"""
Gibbs Generalization Bound Experiments

A modular implementation of SGLD experiments for computing PAC-Bayesian
generalization bounds on synthetic datasets.

Modules:
    - dataset: Dataset creation and data loading utilities
    - models: Neural network models
    - sgld: SGLD optimizer implementation
    - losses: Loss functions (bounded cross-entropy, zero-one loss)
    - training: Training procedures and experiment orchestration
    - bounds: Generalization bound computation and analysis
    - plot_utils: Plotting and visualization utilities
    - main: Main experiment script
"""

__version__ = "1.0.0"
__author__ = "Erfan Mirzaei"

# Import main components for easy access
from .dataset import create_synth_dataset, get_synth_dataloaders
from .models import SynthNN, MNISTNN
from .sgld import SGLD
from .losses import BoundedCrossEntropyLoss, ZeroOneLoss
from .training import train_sgld_model, run_beta_experiments
from .bounds import (
    compute_generalization_bound, 
    compute_generalization_errors, 
    save_results_to_file
)
from .plot_utils import (
    plot_beta_results,
    plot_training_curves,
    plot_bound_comparison,
    plot_bound_tightness
)

__all__ = [
    # Dataset utilities
    'create_synth_dataset',
    'get_synth_dataloaders',
    
    # Models and optimizers
    'SynthNN',
    'MNISTNN', 
    'SGLD',
    
    # Loss functions
    'BoundedCrossEntropyLoss',
    'ZeroOneLoss',
    
    # Training functions
    'train_sgld_model',
    'run_beta_experiments',
    
    # Bounds analysis
    'compute_generalization_bound',
    'compute_generalization_errors',
    'save_results_to_file',
    
    # Plotting utilities
    'plot_beta_results',
    'plot_training_curves',
    'plot_bound_comparison',
    'plot_bound_tightness',
]
