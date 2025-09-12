
import csv
import math
import os
import torch
from losses import BoundedCrossEntropyLoss, ZeroOneLoss
from bounds import invert_kl, ln, kl
from training import transform_bce_to_unit_interval
import numpy as np


def read_csv_2_lists(csv_file_path):
    """
    Read CSV file and return data organized by beta.
    
    Args:
        csv_file_path: Path to the CSV file
    Returns:
        list 
    """     
    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        return []
    
    # Initialize data structure
    n_samples = []
    beta_values = []
    list_train_BCE_losses = []
    list_test_BCE_losses = []
    list_train_01_losses = []
    list_test_01_losses = []
    list_EMA_train_BCE_losses = []
    list_EMA_test_BCE_losses = []
    list_EMA_train_01_losses = []
    list_EMA_test_01_losses = []
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        n_samples_idx = headers.index("Sample_size")
        beta_idx = headers.index("Beta")
        train_BCE_idx = headers.index("BCE_Train")
        test_BCE_idx = headers.index("BCE_Test")
        train_01_idx = headers.index("0-1_Train")
        test_01_idx = headers.index("0-1_Test")
        EMA_train_BCE_idx = headers.index("EMA_BCE_Train")
        EMA_test_BCE_idx = headers.index("EMA_BCE_Test")
        EMA_train_01_idx = headers.index("EMA_0-1_Train")
        EMA_test_01_idx = headers.index("EMA_0-1_Test")
        for row in reader:
            if len(row) > 1 and row[0] != 'Summary:':  # Skip empty lines
                n_samples.append(int(row[n_samples_idx])) 
                beta_values.append(float(row[beta_idx]))
                list_train_BCE_losses.append(float(row[train_BCE_idx]))
                list_test_BCE_losses.append(float(row[test_BCE_idx]))
                list_train_01_losses.append(float(row[train_01_idx]))
                list_test_01_losses.append(float(row[test_01_idx]))
                list_EMA_train_BCE_losses.append(float(row[EMA_train_BCE_idx]))
                list_EMA_test_BCE_losses.append(float(row[EMA_test_BCE_idx]))
                list_EMA_train_01_losses.append(float(row[EMA_train_01_idx]))
                list_EMA_test_01_losses.append(float(row[EMA_test_01_idx]))

            elif len(row) > 1 and row[0] == 'Summary:':
                summary_string = ', '.join(row[1:])
                print(f"Summary from CSV: {summary_string}")
    print(f"   Loaded {len(beta_values)} rows")
    return beta_values, list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses, list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses, n_samples

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # TODO: THESE NAMES SHOULD BE CHANGED WITH THE TRUE NAMES OF THE FILES
    # # True label
    csv_filename = "checkpoint_mnist_beta250-16000_20250911_113247.csv"
    #"checkpoint_mnist_random_beta250-16000_20250911_034846.csv"
    #"checkpoint_mnist_beta250-16000_20250910_233202.csv"
    #"checkpoint_mnist_random_beta250-16000_20250910_211119.csv"
    #"checkpoint_mnist_random_beta250-4000_20250910_170038.csv" 
    #"checkpoint_mnist_random_beta250-4000_20250910_151654.csv"
    #"checkpoint_mnist_random_beta250-4000_20250910_140918.csv"
    #"checkpoint_mnist_random_beta250-4000_20250910_105653.csv"
    #"checkpoint_mnist_random_beta250-4000_20250909_231851.csv"
    #"checkpoint_mnist_random_beta250-4000_20250909_172655.csv" 
    #"checkpoint_mnist_random_beta250-4000_20250909_163533.csv"

    # Random label
    # csv_filename = "experiment_mnist_random_beta250-16000_rep10_20250821_054118_test_output_label_products_20250821_054118.csv"

    # TODO: HERE I AM ASSUMED THAT CSV FILES ARE SAVED IN A SUBDIRECTORY NAMED 'csv_outputs' BUT IF IT IS NOT THE CASE THAT EXTRA ARGUMENT COULD EASILY BE OMITTED
    csv_path = os.path.join(script_dir, "csv_EMA", csv_filename)

    # Read the CSV files
    beta_values, list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,\
    list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses, n_samples = read_csv_2_lists(csv_path)

    print("Train Data:")
    for beta, train_bce, test_bce, train_01, test_01, EMA_train_bce, EMA_test_bce, EMA_train_01, EMA_test_01 in zip(    
        beta_values, list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,
        list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses):
        print(f"Beta: {beta}, Train BCE: {train_bce}, Test BCE: {test_bce}, Train 0-1: {train_01}, Test 0-1: {test_01}, EMA Train BCE: {EMA_train_bce}, EMA Test BCE: {EMA_test_bce}, EMA Train 0-1: {EMA_train_01}, EMA Test 0-1: {EMA_test_01}")

 
    lmax = 4
    def bce (v):
        psi = math.exp (-lmax)+(1-2*math.exp(-lmax))/(1+math.exp(-v))
        return (-ln(psi)-math.exp(-lmax))/(lmax-math.exp(-lmax))

    # f(x) = max(0, min(1, (1-x)/gamma))
    def f(x, gamma):
        if x <= 0:
            return 1
        elif x >= gamma:
            return 0
        else:
            return 1 - (x / gamma)


    kl_upper_bounds = []
    BCE_test_bounds = []
    BCE_train_test_kl = []
    BCE_tests = []
    zeroOne_test_bounds = []
    zeroOne_train_test_kl = []
    zeroOne_tests = []

    for j, beta in enumerate(beta_values):

        # Filter integration betas to only include those up to current_beta
        relevant_betas = [b for b in beta_values if b <= beta]

        # Always start from 0 and integrate up to current_beta
        prev_beta_index = 0
        prev_beta = 0.0
        integral_bound = 0.0
        for i, beta_k in enumerate(relevant_betas):
            # Calculate beta difference from previous point
            beta_diff = beta_k - prev_beta
            
            # Add to integral approximation
            integral_bound += beta_diff * list_EMA_train_BCE_losses[prev_beta_index]

            prev_beta = beta_k
            prev_beta_index = i
        integral_upper_bound = integral_bound #TODO: Change to use: + integral_confidence

        # Compute the main generalization bound
        # Inner term: integral - β * L̂(h,x) + ln(2√n/δ)
        delta = 0.05  # Confidence level
        n = n_samples[-1]
        kl_upper_bound = (integral_upper_bound - beta * list_EMA_train_BCE_losses[j] + math.log(2 * math.sqrt(n) / delta)) / n
        test_bound = invert_kl(list_EMA_train_BCE_losses[j], kl_upper_bound, eps=1e-10)
        zeroOne_test_bound = invert_kl(list_EMA_train_01_losses[j], kl_upper_bound, eps=1e-10)
        kl_upper_bounds.append(kl_upper_bound)
        BCE_test_bounds.append(test_bound)
        BCE_train_test_kl.append(kl(list_EMA_train_BCE_losses[j], list_EMA_test_BCE_losses[j]))
        BCE_tests.append(list_EMA_test_BCE_losses[j])
        zeroOne_test_bounds.append(zeroOne_test_bound)
        zeroOne_train_test_kl.append(kl(list_EMA_train_01_losses[j], list_EMA_test_01_losses[j]))
        zeroOne_tests.append(list_EMA_test_01_losses[j])

    return kl_upper_bounds, BCE_test_bounds, BCE_train_test_kl, BCE_tests, zeroOne_test_bounds, zeroOne_train_test_kl, zeroOne_tests, beta_values, list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses, n_samples, csv_filename

if __name__ == "__main__":
    kl_upper_bounds, BCE_test_bounds, BCE_train_test_kl, BCE_tests, zeroOne_test_bounds, zeroOne_train_test_kl, zeroOne_tests,\
          beta_values, list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses, n_samples, csv_filename = main()

    import matplotlib.pyplot as plt
    import numpy as np


    # Create comprehensive plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define consistent colors
    train_color = 'blue'
    test_color = 'orange'
    gen_error_color = 'green'
    bound_color = 'red'
    individual_bound_color = 'purple'
    kl_color = 'brown'
    
    # Plot 1: BCE Train/Test + KL Analysis (no generalization bounds/errors)
    ax1.errorbar(beta_values[1:], list_EMA_train_BCE_losses[1:], fmt='o-', label='Train BCE', linewidth=2, markersize=5, capsize=3, color=train_color)
    ax1.errorbar(beta_values[1:], list_EMA_test_BCE_losses[1:], fmt='s-', label='Test BCE', linewidth=2, markersize=5, capsize=3, color=test_color)
    ax1.errorbar(beta_values[1:], BCE_test_bounds[1:], fmt='p-', label='Test Bound', linewidth=2, markersize=5, capsize=3, color=kl_color)
    ax1.errorbar(beta_values[1:], BCE_train_test_kl[1:], fmt='h-', label='KL(train||test)', linewidth=2, markersize=4, capsize=3, color='darkblue')
    ax1.errorbar(beta_values[1:], kl_upper_bounds[1:], fmt='*-', label='KL Bound', linewidth=2, markersize=4, capsize=3, color='gray')

    ax1.set_xlabel('Beta (Inverse Temperature)')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('BCE: Train/Test Losses & KL Analysis')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    # Use linear scale for x-axis
    # ax1.set_xscale('linear')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    # Plot 2: Zero-One Train/Test + KL Analysis (no generalization bounds/errors)
    ax2.errorbar(beta_values[1:], list_EMA_train_01_losses[1:], fmt='o-', label='Train 0-1', linewidth=2, markersize=5, capsize=3, color=train_color)
    ax2.errorbar(beta_values[1:], list_EMA_test_01_losses[1:], fmt='s-', label='Test 0-1', linewidth=2, markersize=5, capsize=3, color=test_color)
    ax2.errorbar(beta_values[1:], zeroOne_test_bounds[1:], fmt='p-', label='Test Bound (via KL)', linewidth=2, markersize=5, capsize=3, color=kl_color)
    ax2.errorbar(beta_values[1:], zeroOne_train_test_kl[1:], fmt='h-', label='KL(train||test)', linewidth=2, markersize=4, capsize=3, color='darkblue')
    ax2.errorbar(beta_values[1:], kl_upper_bounds[1:], fmt='*-', label='KL Bound', linewidth=2, markersize=4, capsize=3, color='gray')

    ax2.set_xlabel('Beta (Inverse Temperature)')
    ax2.set_ylabel('Loss Value')
    ax2.set_title('Zero-One: Train/Test Losses & KL Analysis')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    # Use linear scale for x-axis
    # ax2.set_xscale('linear')
    # log-log plot
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.tight_layout()
    
    csv_filename = csv_filename.replace('.csv', '_plot.png')
    # Save the figure
    plt.savefig(csv_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{csv_filename}'")

    plt.show()

