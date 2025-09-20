
import csv
import math
import os
import torch
from losses import BoundedCrossEntropyLoss, ZeroOneLoss
from training import transform_bce_to_unit_interval
import numpy as np


def ln(x):
    """Natural logarithm function."""
    return math.log(x)


def kl(p, q, eps=1e-10):
    """
    Compute KL divergence between two Bernoulli distributions.
    
    Args:
        p: First probability
        q: Second probability
        eps: Small epsilon to prevent numerical issues with zero values
    
    Returns:
        KL divergence KL(p||q)
    """
    # Add epsilon to prevent log(0) and division by 0
    q = max(min(q, 1.0 - eps), eps)  # Clamp q to [eps, 1-eps]

    if p == 0:
        return ln(1/(1-q))
    if p == 1:
        return ln(1/q)

    y = p * ln(p / q) + (1 - p) * ln((1 - p) / (1 - q))
    return y


def invert_kl(p, kl_val, eps=1e-10):
    """
    Invert KL divergence to find q such that KL(p||q) = kl_val.
    
    Args:
        p: First probability (fixed)
        kl_val: Target KL divergence value
        eps: Small epsilon to prevent numerical issues with zero values
    
    Returns:
        q such that KL(p||q) = kl_val
    """
    
    l, u, r = p, 1, 1
    while ((u - l) > 1 / 100000):
        if kl(p, r, eps) < kl_val:
            l = r
            r = (r + u) / 2
        else:
            u = r
            r = (r + l) / 2
    return r


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

    print(f"   Loaded {len(beta_values)} rows")
    return beta_values, list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses, list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses, n_samples

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # TODO: THESE NAMES SHOULD BE CHANGED WITH THE TRUE NAMES OF THE FILES
    csv_filename = "SCL2W1000SGLD2kLR0005BBCE.csv"

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
    plt.tight_layout()
    
    # Create CSV directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save training data
    if csv_filename.endswith('.csv'):
        csv_filename = csv_filename[:-4]
    csv_filename = f"plots/{csv_filename}_plot.png"

    # Save the figure
    plt.savefig(csv_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{csv_filename}'")

    plt.show()

# # Publication-ready plotting function with additional customization
# def create_publication_plot(plot_type='bce', showkls=0, save_format='both', 
#                           colorblind_friendly=True, style='default'):
#     """
#     Create publication-ready plots with enhanced styling options.
    
#     Args:
#         plot_type: 'bce' or '01' for BCE loss or 0-1 error plots
#         showkls: 0 or 1 to show/hide KL divergence plots
#         save_format: 'png', 'pdf', 'svg', or 'both' (png + pdf)
#         colorblind_friendly: Use colorblind-friendly palette
#         style: matplotlib style ('seaborn-v0_8', 'classic', 'bmh', etc.)
#     """
#     # Set style
#     try:
#         plt.style.use(style)
#     except:
#         plt.style.use('default')
    
#     # Publication settings
#     plt.rcParams.update({
#         'font.family': 'serif',
#         'font.serif': ['Times New Roman', 'Times', 'serif'],
#         'font.size': 16,
#         'axes.labelsize': 18,
#         'axes.titlesize': 20,
#         'xtick.labelsize': 16,
#         'ytick.labelsize': 16,
#         'legend.fontsize': 16,
#         'lines.linewidth': 3,
#         'lines.markersize': 10,
#         'figure.figsize': (12, 8),
#         'axes.grid': True,
#         'grid.alpha': 0.3,
#         'axes.axisbelow': True,
#         'savefig.dpi': 300,
#         'savefig.bbox': 'tight',
#         'text.usetex': False  # Set to True if you have LaTeX installed
#     })
    
#     # Color palette
#     if colorblind_friendly:
#         colors = {
#             'train': '#1f77b4',      # Blue
#             'test_bound': '#ff7f0e', # Orange  
#             'test': '#2ca02c',       # Green
#             'kl_train_test': '#d62728',  # Red
#             'kl_bound': '#9467bd'    # Purple
#         }
#     else:
#         colors = {
#             'train': '#2E86AB',
#             'test_bound': '#A23B72', 
#             'test': '#F18F01',
#             'kl_train_test': '#C73E1D',
#             'kl_bound': '#7209B7'
#         }
    
#     fig, ax = plt.subplots()
#     ax.semilogx()
    
#     # Plot data based on type
#     if plot_type == 'bce':
#         ax.plot(betas[1:], av_bcetrain[1:], 'o-', color=colors['train'], 
#                 linewidth=3, markersize=10, label='Training', 
#                 markerfacecolor='white', markeredgewidth=2.5)
#         ax.plot(betas[1:], predbce[1:], 's-', color=colors['test_bound'], 
#                 linewidth=3, markersize=8, label='Test Bound', 
#                 markerfacecolor='white', markeredgewidth=2.5)
#         ax.plot(betas[1:], av_bcetest[1:], '^-', color=colors['test'], 
#                 linewidth=3, markersize=10, label='Test', 
#                 markerfacecolor='white', markeredgewidth=2.5)
#         ylabel = 'BCE Loss'
#         plot_suffix = '_bce_publication'
#     else:  # plot_type == '01'
#         ax.plot(betas[1:], av_train01[1:], 'o-', color=colors['train'], 
#                 linewidth=3, markersize=10, label='Training', 
#                 markerfacecolor='white', markeredgewidth=2.5)
#         ax.plot(betas[1:], pred01, 's-', color=colors['test_bound'], 
#                 linewidth=3, markersize=8, label='Test Bound', 
#                 markerfacecolor='white', markeredgewidth=2.5)
#         ax.plot(betas[1:], av_test01[1:], '^-', color=colors['test'], 
#                 linewidth=3, markersize=10, label='Test', 
#                 markerfacecolor='white', markeredgewidth=2.5)
#         ylabel = '0-1 Error'
#         plot_suffix = '_01_publication'
    
#     if showkls == 1:
#         if plot_type == 'bce':
#             ax.plot(betas[1:], testkl[1:], 'v-', color=colors['kl_train_test'], 
#                     linewidth=2.5, markersize=7, label='KL(Train,Test)', alpha=0.8)
#         else:
#             ax.plot(betas[1:], testkl01[1:], 'v-', color=colors['kl_train_test'], 
#                     linewidth=2.5, markersize=7, label='KL(Train,Test)', alpha=0.8)
#         ax.plot(betas[1:], bounds[1:], 'D-', color=colors['kl_bound'], 
#                 linewidth=2.5, markersize=7, label='KL-Bound', alpha=0.8)
    
#     # Enhanced formatting
#     ax.set_xlabel(r'$\beta$', fontsize=20, fontweight='bold')
#     ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
#     ax.set_ylim(0, 0.6)
    
#     # Enhanced legend
#     ax.legend(frameon=True, fancybox=True, shadow=True, loc='best', 
#               framealpha=0.95, edgecolor='black', facecolor='white')
    
#     # Enhanced ticks
#     ax.minorticks_on()
#     ax.tick_params(which='minor', length=4, color='gray')
#     ax.tick_params(which='major', length=8, width=1.5)
    
#     # Add subtle border
#     for spine in ax.spines.values():
#         spine.set_linewidth(1.5)
    
#     plt.tight_layout()
    
#     # Generate base filename
#     os.makedirs('newplots', exist_ok=True)
#     if trueLabels == 1:
#         base_filename = truefilename[:-4] + plot_suffix
#     else:
#         base_filename = randomfilename[:-4] + plot_suffix
    
#     # Save in requested formats
#     if save_format in ['png', 'both']:
#         plt.savefig(f'newplots/{base_filename}.png', dpi=300, 
#                    bbox_inches='tight', facecolor='white', edgecolor='none')
#     if save_format in ['pdf', 'both']:
#         plt.savefig(f'newplots/{base_filename}.pdf', dpi=300, 
#                    bbox_inches='tight', facecolor='white', edgecolor='none')
#     if save_format == 'svg':
#         plt.savefig(f'newplots/{base_filename}.svg', dpi=300, 
#                    bbox_inches='tight', facecolor='white', edgecolor='none')
    
#     plt.show()
#     return fig, ax

# # Example usage for publication plots:
# # create_publication_plot('bce', showkls=1, save_format='both', colorblind_friendly=True)
# # create_publication_plot('01', showkls=1, save_format='pdf', colorblind_friendly=False)
