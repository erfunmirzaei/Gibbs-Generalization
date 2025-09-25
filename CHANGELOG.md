# Changelog

## [1.0.0] - 2025-09-25

### Added
- Comprehensive SGLD/ULA implementation for MNIST and CIFAR-10 datasets
- Multiple neural network architectures: FCN (1-3 layers), LeNet-5, VGG-16
- Support for both correct labels and random label experiments
- PAC-Bayesian generalization bound computation with KL-divergence
- Multiple loss functions: BBCE, Savage, Tangent, Zero-one loss
- Exponential Moving Average (EMA) tracking for stable convergence assessment
- Automatic CSV result export with descriptive filenames
- Plot generation with bounds visualization
- Configurable test mode for quick validation
- Automatic beta=0 baseline inclusion for proper bound computation

### Project Structure
- `main.py` - Main experiment orchestration
- `dataset.py` - MNIST/CIFAR-10 dataset handling with flexible class groupings  
- `models.py` - Neural network architectures
- `sgld.py` - SGLD/ULA optimizer implementation
- `losses.py` - Loss functions for PAC-Bayesian analysis
- `training.py` - Training procedures with EMA tracking
- `plot.py` - Visualization and bound computation utilities
- `table_MNIST.py` / `table_CIFAR.py` - Results analysis scripts
- `csv_EMA/` - Experimental results storage
- `newplots/` - Generated visualization outputs

### Key Features
- Binary classification on both MNIST (even/odd digits) and CIFAR-10 (vehicles/animals)
- Inverse temperature (β) parameter controlling exploration vs exploitation
- Systematic file naming convention for reproducibility
- Built-in support for publication-ready tables and plots
- Theoretical grounding in PAC-Bayesian learning theory

### Dependencies
- PyTorch >= 1.9.0
- torchvision >= 0.10.0  
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0

### Usage
```bash
# Quick test
python main.py  # with TEST_MODE = True

# Full experiment  
python main.py  # with TEST_MODE = False

# Generate analysis tables
python table_MNIST.py
python table_CIFAR.py
```
