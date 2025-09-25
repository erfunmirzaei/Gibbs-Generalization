# Gibbs Generalization Bound Experiments

A comprehensive implementation of SGLD (Stochastic Gradient Langevin Dynamics) and ULA (Unadjusted Langevin Algorithm) experiments for computing PAC-Bayesian generalization bounds on MNIST and CIFAR-10 datasets.

## Project Structure

The code has been organized into the following modules for better maintainability and reusability:

### Core Modules

- **`dataset.py`** - Dataset creation and data loading utilities
  - `get_mnist_binary_dataloaders()` - Creates MNIST binary classification datasets
  - `get_cifar10_binary_dataloaders()` - Creates CIFAR-10 binary classification datasets
  - `get_mnist_binary_dataloaders_random_labels()` - MNIST with random labels for generalization studies
  - `get_cifar10_binary_dataloaders_random_labels()` - CIFAR-10 with random labels for generalization studies
  - Support for both individual class pairs and grouped classes (e.g., even vs odd digits)

- **`models.py`** - Neural network architectures
  - `FCN1L` - 1-layer fully connected network
  - `FCN2L` - 2-layer fully connected network  
  - `FCN3L` - 3-layer fully connected network
  - `LeNet5` - LeNet-5 architecture for MNIST
  - `VGG16_CIFAR` - VGG-16 architecture adapted for CIFAR-10
  - `initialize_nn_weights_gaussian()` - Gaussian weight initialization for Bayesian analysis

- **`sgld.py`** - SGLD/ULA optimizer implementation
  - `SGLD` - Stochastic Gradient Langevin Dynamics optimizer with inverse temperature (β) control
  - Supports both SGLD (with gradient noise) and ULA modes
  - Configurable noise injection and step size scheduling

- **`losses.py`** - Loss functions for PAC-Bayesian analysis
  - `BoundedCrossEntropyLoss` - Bounded cross-entropy loss (BBCE) for generalization bounds
  - `ZeroOneLoss` - Zero-one loss for classification error evaluation  
  - `TangentLoss` - Tangent loss implementation
  - `SavageLoss` - Savage loss for robust learning

- **`training.py`** - Training procedures and experiment orchestration
  - `train_sgld_model()` - Train individual models with SGLD/ULA
  - `run_beta_experiments()` - Run comprehensive experiments across multiple β values
  - `save_moving_average_losses_to_csv()` - Export results to CSV format
  - **Automatic Beta=0 Inclusion**: Automatically adds β=0 (pure gradient descent) for proper bound computation
  - **Exponential Moving Average (EMA)** tracking for stable loss estimation

- **`plot.py`** - Plotting and visualization utilities
  - `create_plots_from_csv()` - Generate plots from CSV experimental results
  - `klbounds()` - Compute KL-divergence based PAC-Bayesian bounds  
  - `invert_kl()` - Numerical inversion of KL divergence for bound computation
  - Support for multiple bound types (KL, Hoeffding, Bernstein)
  - Automatic plot generation with generalization bounds vs actual test errors

### Main Scripts

- **`main.py`** - Main experiment script
  - Orchestrates complete SGLD/ULA experiments on MNIST or CIFAR-10
  - Configurable for test mode vs full experiment
  - Support for both normal and random label experiments
  - Automatic dataset selection and parameter configuration

- **`table_MNIST.py`** - MNIST results analysis and table generation
  - Processes experimental results from CSV files
  - Generates publication-ready tables and plots
  - Computes and displays PAC-Bayesian bounds

- **`table_CIFAR.py`** - CIFAR-10 results analysis and table generation  
  - Similar functionality to table_MNIST.py but for CIFAR-10 experiments
  - Handles CIFAR-10 specific experimental parameters

- **`__init__.py`** - Package initialization
  - Imports all main components for easy access
  - Defines the public API with version information

### Output Directories

- **`csv_EMA/`** - CSV files with experimental results including EMA (Exponential Moving Average) tracking
  - Contains loss trajectories, generalization errors, and experimental metadata
  - Organized by experiment parameters (dataset, β values, learning rates, etc.)
  
- **`newplots/`** - Generated visualization outputs
  - Loss curves, generalization bounds, and comparative analysis plots
  - Automatic filename generation based on experimental parameters

## Output Files and Naming Convention

### CSV Results Format
Experimental results are saved in `csv_EMA/` with descriptive filenames:

```
{Dataset}{Label_Type}{Network_Type}W{Width}{Algorithm}{Steps}LR{Learning_Rate}{Loss_Type}.csv
```

**Components:**
- `{Dataset}`: M (MNIST), C (CIFAR-10)
- `{Label_Type}`: CL (Correct Labels), RL (Random Labels) 
- `{Network_Type}`: 1, 2, 3 (FCN layers), L (LeNet), V (VGG)
- `{Width}`: Network width (e.g., 500, 1000, 2000)
- `{Algorithm}`: SGLD, ULA, SGD
- `{Steps}`: Training steps (e.g., 2k, 8k)
- `{Learning_Rate}`: Learning rate (e.g., 0005 for 0.005)
- `{Loss_Type}`: BBCE, SAVAGE, etc.

**Example Filenames:**
- `MCL2W1000SGLD8kLR001BBCE.csv` - MNIST, Correct Labels, 2-layer FCN, width 1000, SGLD, 8k steps, lr=0.01, BBCE loss
- `CRL3W500ULA2kLR0005SAVAGE.csv` - CIFAR-10, Random Labels, 3-layer FCN, width 500, ULA, 2k steps, lr=0.005, Savage loss

### CSV File Structure
Each CSV contains per-iteration results:
- Sample size, Beta value, iteration number
- Training/test losses (BCE and 0-1 error)
- Exponential Moving Averages (EMA) of all metrics
- Timestamp and experimental metadata

### Plot Outputs
Generated plots are saved in `newplots/` with corresponding names:
- `{ExperimentName}_01.png` - Generalization error (0-1 loss) plots with bounds
- `{ExperimentName}_loss.png` - Loss trajectory plots
- `{ExperimentName}_KL.png` - KL-divergence bound visualizations

## Usage

### Quick Start
```bash
# Run the main experiment (configured in main.py)
python main.py
```

### Configuration Options
Edit `main.py` to customize experiments:

```python
# Basic configuration flags
TEST_MODE = False           # Set True for quick testing, False for full experiment
USE_RANDOM_LABELS = True    # Set True for random label experiments  
DATASET_TYPE = 'cifar10'    # Choose 'mnist' or 'cifar10'

# Class selection for binary classification
MNIST_CLASSES = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]  # Even vs Odd digits
CIFAR10_CLASSES = [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]  # Vehicles vs Animals

# Beta values (inverse temperature parameter)
beta_values = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
```

### Individual Components Usage
```python
# Dataset creation
from dataset import get_mnist_binary_dataloaders, get_cifar10_binary_dataloaders

train_loader, test_loader = get_mnist_binary_dataloaders(
    classes=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],  # Even vs odd
    n_train_per_group=1000,
    batch_size=50
)

# Model initialization  
from models import FCN2L, VGG16_CIFAR
model = FCN2L(input_dim=784, hidden_dim=1000)  # For MNIST
model = VGG16_CIFAR()  # For CIFAR-10

# SGLD optimizer
from sgld import SGLD
optimizer = SGLD(model.parameters(), lr=0.01, beta=1000.0, add_noise=True)

# Loss functions
from losses import BoundedCrossEntropyLoss, SavageLoss
criterion = BoundedCrossEntropyLoss(ell_max=4.0)
```

### Generate Results Tables and Plots
```python
# For MNIST results
python table_MNIST.py

# For CIFAR-10 results  
python table_CIFAR.py
```

## Key Features

1. **Comprehensive Datasets**: Support for both MNIST and CIFAR-10 with flexible class groupings
2. **Multiple Architectures**: FCN (1-3 layers), LeNet-5, and VGG-16 implementations
3. **Advanced Optimization**: SGLD, ULA, and SGD with configurable noise injection
4. **Multiple Loss Functions**: BBCE, Savage, Tangent losses for robust PAC-Bayesian analysis  
5. **Random Label Experiments**: Built-in support for generalization studies with random labels
6. **PAC-Bayesian Bounds**: Complete implementation of KL-divergence based generalization bounds
7. **Automatic Result Management**: CSV export, plot generation, and organized file naming
8. **EMA Tracking**: Exponential Moving Average tracking for stable convergence assessment
9. **Configurable Experiments**: Test mode for quick validation, full mode for publication results

## Algorithm and Parameter Guide

### Beta (β) Parameter - Inverse Temperature
- **β = 0**: Pure gradient descent (deterministic, no noise injection)
- **β > 0**: SGLD with decreasing noise as β increases
- **Higher β**: Less exploration, more exploitation (approaches deterministic optimization)
- **Lower β**: More exploration, better generalization (at cost of training accuracy)

### Supported Algorithms
- **SGLD**: Stochastic Gradient Langevin Dynamics with Gaussian noise injection
- **ULA**: Unadjusted Langevin Algorithm (continuous-time limit of SGLD)  
- **SGD**: Standard Stochastic Gradient Descent (β → ∞ limit)

### Loss Functions for PAC-Bayesian Analysis
- **BBCE (Bounded Binary Cross Entropy)**: Primary loss for generalization bounds [0,1] bounded
- **Savage Loss**: Robust alternative with different theoretical properties
- **Tangent Loss**: Additional robust loss option

## Experiment Configuration Guide

### Test vs Full Mode
Configure experiments by modifying flags in `main.py`:

```python
TEST_MODE = False  # True: Quick test (single β), False: Full experiment (8 β values)
USE_RANDOM_LABELS = True   # True: Random labels, False: Correct labels
DATASET_TYPE = 'cifar10'   # 'mnist' or 'cifar10'
```

### Typical Experimental Setup

**MNIST Experiments:**
- Binary classification: Even digits vs Odd digits
- Network: 2-layer FCN with width 1000  
- Training samples: 2000 (1000 per class)
- β values: [125, 250, 500, 1000, 2000, 4000, 8000, 16000]

**CIFAR-10 Experiments:**
- Binary classification: Vehicles vs Animals
- Network: VGG-16 or 3-layer FCN with width 500
- Training samples: 8000 (4000 per class)  
- β values: [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]

## Installation and Dependencies

### Required Dependencies
```bash
pip install torch torchvision numpy matplotlib csv
```

### Optional Dependencies
For extended analysis:
```bash
pip install pandas seaborn scipy
```

### Conda Environment (Recommended)
```bash
conda create -n gibbs-gen python=3.8
conda activate gibbs-gen
conda install pytorch torchvision numpy matplotlib
```

## References and Theoretical Background

This implementation is based on the following key papers:

1. **SGLD Algorithm**: Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *Proceedings of the 28th International Conference on Machine Learning (ICML-11)*, 681-688.

2. **PAC-Bayesian Theory**: McAllester, D. A. (1999). PAC-Bayesian model averaging. *Proceedings of the 12th Annual Conference on Computational Learning Theory*, 164-170.

3. **Generalization Bounds**: Catoni, O. (2007). *PAC-Bayesian supervised classification: the thermodynamics of statistical learning*. Institute of Mathematical Statistics.

### Theoretical Framework
The experiments implement PAC-Bayesian generalization bounds of the form:

```
P(R(h) ≤ R̂(h) + √((KL(Q||P) + ln(2√n/δ))/(2n))) ≥ 1-δ
```

Where:
- `R(h)`: True generalization error
- `R̂(h)`: Empirical training error  
- `KL(Q||P)`: KL divergence between posterior Q and prior P
- `n`: Sample size, `δ`: Confidence parameter

## Mathematical Details

### Beta=0 Baseline and Bounds Computation
The PAC-Bayesian generalization bound computation requires a baseline at β=0 (pure SGD):

- **Automatic β=0 inclusion**: `run_beta_experiments()` automatically adds β=0 when not specified
- **Integral approximation**: Bounds computed using trapezoidal rule from β=0 to target β
- **Mathematical correctness**: Ensures proper KL-divergence computation for generalization bounds

### Bound Computation Process
1. Train models at β=0 (SGD) and target β values (SGLD/ULA)
2. Compute empirical losses and KL divergences between posterior and prior
3. Apply PAC-Bayesian bound formula with confidence parameter δ
4. Generate bounds that hold with probability ≥ 1-δ

### EMA (Exponential Moving Average) Tracking
Stable loss estimation using EMA with decay factor α:
```
EMA_t = α × loss_t + (1-α) × EMA_{t-1}
```
Default α = 0.01 for stable convergence assessment.

## File Organization and Reproducibility

### Systematic File Naming
All outputs use consistent naming conventions encoding:
- Dataset type and label configuration
- Network architecture and parameters  
- Optimization algorithm and hyperparameters
- Loss function and experimental settings

### Reproducibility Features
- **Fixed Random Seeds**: Consistent dataset generation across runs
- **Parameter Tracking**: All hyperparameters saved in CSV metadata
- **Version Control Ready**: Organized structure for git tracking
- **Automated Outputs**: No manual file management required

### Result Verification
Use `table_MNIST.py` and `table_CIFAR.py` to:
- Load and validate experimental results
- Generate publication-ready tables  
- Create comparative visualizations
- Verify bound computation accuracy

## Contributing

This research code is provided for reproducibility and further research. When using or extending this code:

1. **Cite the relevant papers** listed in the References section
2. **Maintain the modular structure** for easier maintenance and extension
3. **Follow the naming conventions** for consistency with existing outputs
4. **Test with `TEST_MODE=True`** before running full experiments
5. **Document any modifications** for reproducibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite the original SGLD paper:

```bibtex
@inproceedings{welling2011bayesian,
  title={Bayesian learning via stochastic gradient Langevin dynamics},
  author={Welling, Max and Teh, Yee Whye},
  booktitle={Proceedings of the 28th international conference on machine learning (ICML-11)},
  pages={681--688},
  year={2011}
}
```

## Contact

For questions about the implementation or to report issues, please open an issue in the repository.
