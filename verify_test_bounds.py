#!/usr/bin/env python3
"""
Quick verification that test bounds are included in both plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_beta_results

# Create mock data with test bounds
beta_values = [0.0, 0.1, 1.0, 10.0]  # Include beta=0 for bounds computation
mock_results = {}

for beta in beta_values:
    # Basic results that contain all required data for plotting
    mock_results[beta] = {
        'train_bce_mean': 0.5 + 0.1 * np.random.randn(),
        'test_bce_mean': 0.6 + 0.1 * np.random.randn(),
        'train_bce_std': 0.05,
        'test_bce_std': 0.05,
        'train_01_mean': 0.3 + 0.1 * np.random.randn(),
        'test_01_mean': 0.4 + 0.1 * np.random.randn(),
        'train_01_std': 0.03,
        'test_01_std': 0.03,
        'train_losses': [0.5 + 0.1 * np.random.randn() for _ in range(5)],  # Mock 5 repetitions
        'test_losses': [0.6 + 0.1 * np.random.randn() for _ in range(5)],
        'train_zero_one_losses': [0.3 + 0.1 * np.random.randn() for _ in range(5)],
        'test_zero_one_losses': [0.4 + 0.1 * np.random.randn() for _ in range(5)],
        # Raw data needed for bounds computation
        'raw_train_bce': [0.5 + 0.1 * np.random.randn() for _ in range(5)],
        'raw_test_bce': [0.6 + 0.1 * np.random.randn() for _ in range(5)],
        'raw_train_01': [0.3 + 0.1 * np.random.randn() for _ in range(5)],
        'raw_test_01': [0.4 + 0.1 * np.random.randn() for _ in range(5)],
    }

print("Creating plot with test bounds included...")

# Create plot - the function will compute bounds and KL analysis internally
plot_beta_results(
    results=mock_results,
    beta_values=beta_values,
    num_repetitions=5,
    num_epochs=1000,
    a0=0.1,
    sigma_gauss_prior=1.0,
    filename='results/verify_test_bounds_plot.png'
)

print("✅ Plot created with test bounds!")
print("Check 'results/verify_test_bounds_plot.png' to verify both plots include:")
print("  - Train losses")
print("  - Test losses") 
print("  - Test Bound (via KL) - NEW!")
print("  - KL divergence")
print("  - KL bounds")
