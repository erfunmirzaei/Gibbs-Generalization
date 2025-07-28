# Gibbs Generalization Bound Experiments

A modular implementation of SGLD (Stochastic Gradient Langevin Dynamics) experiments for computing PAC-Bayesian generalization bounds on synthetic datasets.

## Project Structure

The code has been organized into the following modules for better maintainability and reusability:

### Core Modules

- **`dataset.py`** - Dataset creation and data loading utilities
  - `create_synth_dataset()` - Creates the synthetic dataset as described in the paper
  - `get_synth_dataloaders()` - Creates PyTorch DataLoaders for training

- **`models.py`** - Neural network architectures
  - `SynthNN` - Neural network for SYNTH dataset (1 hidden layer, 100 units)
  - `MNISTNN` - Neural network for MNIST dataset (3 layers, 600 units each)

- **`sgld.py`** - SGLD optimizer implementation
  - `SGLD` - Stochastic Gradient Langevin Dynamics optimizer with inverse temperature control

- **`losses.py`** - Loss functions
  - `BoundedCrossEntropyLoss` - Bounded cross-entropy loss for PAC-Bayesian analysis
  - `ZeroOneLoss` - Zero-one loss for classification error evaluation

- **`training.py`** - Training procedures and experiment orchestration
  - `train_sgld_model()` - Train a single model with SGLD
  - `run_beta_experiments()` - Run experiments across different beta values
  - **Automatic Beta=0 Inclusion**: Automatically adds beta=0 (pure gradient descent) when not present for proper bound computation

- **`bounds.py`** - Generalization bound computation and analysis
  - `compute_generalization_bound()` - Compute PAC-Bayesian generalization bounds
    - **Internal Beta=0 Handling**: Automatically uses beta=0 from results for proper integral computation
    - **Original Beta Values**: Returns bounds only for the requested beta values
    - **Mathematical Correctness**: Ensures proper integral computation from 0 to each beta value
  - `compute_generalization_errors()` - Compute actual generalization errors
  - `save_results_to_file()` - Save experimental results to text files

- **`plot_utils.py`** - Plotting and visualization utilities
  - `plot_beta_results()` - Main results plot with bounds and errors
  - `plot_training_curves()` - Training curves for individual experiments
  - `plot_bound_comparison()` - Compare theoretical bounds vs actual errors
  - `plot_bound_tightness()` - Visualize bound tightness across beta values

### Main Scripts

- **`main.py`** - Main experiment script
  - Orchestrates the complete experiment
  - Configurable for test mode vs full experiment
  - Generates plots and saves results

- **`__init__.py`** - Package initialization
  - Imports all main components for easy access
  - Defines the public API

### Results Directory

- **`results/`** - Output directory for all experimental results
  - Contains generated plots, result files, and experimental data
  - All output files are automatically saved here

## Output Files

All results are saved in the `results/` folder with descriptive filenames that include experimental parameters:

### Filename Convention
Files are automatically named based on experimental parameters:
```
sgld_{type}_{dataset}_beta{min}-{max}_lr{learning_rate}_sigma{prior}_ep{epochs}_rep{repetitions}.{ext}
```

### Example Filenames
- `sgld_results_synth_beta1-200_lr0p10_sigma10_ep10k_rep30.txt` - Full experiment results
- `sgld_plot_synth_beta1-200_lr0p10_sigma10_ep10k_rep30.png` - Corresponding plot
- `sgld_results_synth_beta1-200_lr0p10_sigma10_ep100_rep5.txt` - Test mode results

### File Contents
- **Results files (`.txt`)**: Experimental parameters, summary tables, detailed statistics, and generalization bounds
- **Plot files (`.png`)**: Visualization of generalization errors, bounds, and training curves

### Benefits of Descriptive Filenames
- **Easy Identification**: Quickly identify experimental parameters from filename
- **No Overwrites**: Different experiments automatically generate different filenames
- **Organization**: Results from multiple experiments are clearly separated
- **Reproducibility**: Filename contains all key parameters for reproduction

## Usage

### Quick Test
```python
# Run a quick test of the system
from main import test_loss_functions
test_loss_functions()
```

### Full Experiment
```python
# Run the main experiment
python main.py
```

### Individual Components
```python
# Use individual components
from dataset import create_synth_dataset
from models import SynthNN
from sgld import SGLD
from losses import BoundedCrossEntropyLoss

# Create dataset
train_dataset, test_dataset = create_synth_dataset(random_seed=42)

# Create model and optimizer
model = SynthNN()
optimizer = SGLD(model.parameters(), lr=1e-3, beta=1.0)

# Train with bounded loss
criterion = BoundedCrossEntropyLoss(l_max=4.0)
```

### Custom Experiments
```python
from training import run_beta_experiments
from plot_utils import plot_beta_results

# Run experiments with custom parameters
results = run_beta_experiments(
    beta_values=[1, 10, 50],
    num_repetitions=10,
    num_epochs=1000
)

# Plot results
plot_beta_results(results)
```

## Key Features

1. **Modular Design**: Each component is in its own file for easy maintenance
2. **Configurable**: Test mode for quick validation, full mode for complete experiments
3. **Comprehensive Plotting**: Multiple visualization options for different aspects of the results
4. **PAC-Bayesian Bounds**: Implements the generalization bound computation from the paper
5. **SGLD Implementation**: Full implementation of Stochastic Gradient Langevin Dynamics
6. **Automatic Beta=0 Baseline**: Automatically includes beta=0 (pure gradient descent) as a baseline for proper generalization bound computation

## Beta Parameter Interpretation

- **Beta = 0**: Pure gradient descent (no SGLD noise) - serves as a deterministic baseline
- **Beta > 0**: SGLD with noise - higher beta means less noise (closer to deterministic)
- **Beta → ∞**: Approaches pure gradient descent again
- The system automatically includes beta=0 when not specified to ensure proper mathematical bounds

## Experiment Configuration

The main experiment can be configured by modifying the `TEST_MODE` flag in `main.py`:

- `TEST_MODE = True`: Quick test with fewer beta values, repetitions, and epochs
- `TEST_MODE = False`: Full experiment as described in the paper

## Dependencies

- PyTorch
- NumPy
- Matplotlib

## References

Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11) (pp. 681-688).

## Beta=0 Handling and Bounds Computation

### Mathematical Requirement
The PAC-Bayesian generalization bound requires computing an integral from β=0 to the target β value:

```
∫₀^β L̂(h_t, S) dt
```

This integral is approximated using the trapezoidal rule and requires the average loss at β=0 (pure gradient descent) as the starting point.

### Automatic Beta=0 Inclusion
- **`run_beta_experiments()`** automatically adds β=0 to the experiment if not present
- This ensures the mathematical requirement is satisfied
- β=0 corresponds to pure gradient descent (no SGLD noise)

### Bounds Computation Behavior
- **Input**: Pass only the original β values you want bounds for
- **Internal Handling**: The bounds functions automatically use β=0 from results for proper integral computation
- **Output**: Returns bounds only for the requested β values (not β=0)

```python
# Example: Request bounds for [1, 10, 50]
bounds = compute_generalization_bound([1, 10, 50], results, loss_type='bce')
# Returns: bounds for [1, 10, 50] only
# But internally uses β=0 from results for proper integral computation
```

### Display and Plotting
- **Results Display**: Shows only the originally requested β values
- **Plots**: Plot only the originally requested β values
- **Files**: Save results for only the originally requested β values
- **Beta=0 is Used**: Internally for bounds computation but not shown in output
