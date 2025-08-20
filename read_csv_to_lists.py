
import csv
import math
import os
import torch
from losses import BoundedCrossEntropyLoss, ZeroOneLoss
from bounds import invert_kl, ln
from training import transform_bce_to_unit_interval
import numpy as np
def read_csv_to_lists(csv_file_path, data_type="unknown"):
    """
    Read CSV file and return data organized by beta.
    
    Args:
        csv_file_path: Path to the CSV file
        data_type: "train" or "test" for labeling
        
    Returns:
        Dictionary with structure: {beta: [output*label values]}
    """
    
    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        return {}
    
    # Initialize data structure
    data = {}
    
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        
        # Skip header metadata (first 6 lines)
        for _ in range(6):
            next(reader)
        
        # Read column headers
        headers = next(reader)
        
        # Find product columns (those starting with 'Output_Label_Product_')
        product_columns = [i for i, header in enumerate(headers) if header.startswith('Output_Label_Product_')]
        
        # Read data rows
        row_count = 0
        for row in reader:
            if len(row) >= 3:  # Ensure we have Beta, Repetition, Sample_Count
                beta = float(row[0])
                repetition = int(row[1])
                sample_count = int(row[2])
                # Extract output*label products
                products = []
                for col_idx in product_columns:
                    if col_idx < len(row) and row[col_idx] != '':
                        products.append(float(row[col_idx]))
                data[beta] = {}
                # Store directly as list for each beta (assuming single repetition)
                data[beta][repetition] = products
                row_count += 1
    
    # Change the order of the keys for data dictionary; first repetitions and then beta
    new_data = {}
    for beta, rep_values_dict in data.items():
        for repetition, products in rep_values_dict.items():
            if repetition not in new_data:
                new_data[repetition] = {}
            new_data[repetition][beta] = products

    print(f"   Loaded {row_count} rows")
    return new_data

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # TODO: THESE NAMES SHOULD BE CHANGED WITH THE TRUE NAMES OF THE FILES
    # # True label
    test_csv_filename = "experiment_mnist_beta250-4000_rep1_20250820_125049_test_output_label_products_20250820_125049.csv"
    train_csv_filename = "experiment_mnist_beta250-4000_rep1_20250820_125049_train_output_label_products_20250820_125049.csv"
    
    # Random label
    # test_csv_filename = "experiment_mnist_random_beta250-4000_rep1_20250820_131024_test_output_label_products_20250820_131024.csv"
    # train_csv_filename = "experiment_mnist_random_beta250-4000_rep1_20250820_131024_train_output_label_products_20250820_131024.csv"

    # TODO: HERE I AM ASSUMED THAT CSV FILES ARE SAVED IN A SUBDIRECTORY NAMED 'csv_outputs' BUT IF IT IS NOT THE CASE THAT EXTRA ARGUMENT COULD EASILY BE OMITTED
    train_csv_path = os.path.join(script_dir, "csv_outputs", train_csv_filename)
    test_csv_path = os.path.join(script_dir, "csv_outputs", test_csv_filename)

    # Read the CSV files
    train_data = read_csv_to_lists(train_csv_path, "train")
    test_data = read_csv_to_lists(test_csv_path, "test")
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

    BCE_criterion = BoundedCrossEntropyLoss()
    ZO_criterion = ZeroOneLoss()

    train_BCE = {}
    # train_BCE2 = {}
    gamma_term = {}
    ramp_losses = {}
    test_bounds = {}
    zo_test_errors = {}
    for repetition, rep_values_dict in train_data.items():
        train_BCE[repetition] = {}
        ramp_losses[repetition] = {}
        zo_test_errors[repetition] = {}
        for beta, output in rep_values_dict.items():
            train_BCE[repetition][beta] = BCE_criterion(torch.tensor(output).unsqueeze(-1), torch.ones_like(torch.tensor(output)))
            train_BCE[repetition][beta] = transform_bce_to_unit_interval(train_BCE[repetition][beta], l_max=4)
            # train_BCE2[beta] = sum(bce(val) for val in output) / len(output)
            ramp_losses[repetition][beta] = sum(f(val, 0) for val in output) / len(output)
            zo_test_errors[repetition][beta] = ZO_criterion(torch.tensor(test_data[repetition][beta]).unsqueeze(-1), torch.ones_like(torch.tensor(test_data[repetition][beta]).unsqueeze(-1)))

    # Take average over repetitions of train_BCE
    betas = list(train_BCE[1].keys())
    average_train_BCE = {beta: np.mean([train_BCE[rep][beta] for rep in train_BCE]).item() for beta in betas}

    for repetition, rep_values_dict in train_data.items():
        gamma_term[repetition] = {}
        test_bounds[repetition] = {}
        for beta, output in rep_values_dict.items():
            # Filter integration betas to only include those up to current_beta
            relevant_betas = [b for b in rep_values_dict.keys() if b <= beta]

            # Always start from 0 and integrate up to current_beta
            prev_beta = 0.0
            integral_bound = 0.0
            for beta_k in relevant_betas:
                # Calculate beta difference from previous point
                beta_diff = beta_k - prev_beta
                
                # Add to integral approximation
                integral_bound += beta_diff * average_train_BCE[prev_beta]

                prev_beta = beta_k

            integral_upper_bound = integral_bound #TODO: Change to use: + integral_confidence

            # Compute the main generalization bound
            # Inner term: integral - β * L̂(h,x) + ln(2√n/δ)
            delta = 0.05  # Confidence level
            n = len(output)
            gammaterm = (integral_upper_bound - beta * train_BCE[repetition][beta] + math.log(2 * math.sqrt(n) / delta)) / n

            gamma_term[repetition][beta] = gammaterm
            test_bounds[repetition][beta] = invert_kl(ramp_losses[repetition][beta], gamma_term[repetition][beta],eps=1e-10)
    return train_data, test_data, train_BCE, ramp_losses, gamma_term, test_bounds, zo_test_errors

if __name__ == "__main__":
    train_data, test_data, train_BCE, ramp_losses, gamma_term, test_bounds, zo_test_errors = main()
    for repetition in train_data.keys():
        print(f"Repetition {repetition}:")
        for beta in train_data[repetition].keys():
            print(f"  Beta {beta}:")
            print(f"    Train BCE: {train_BCE[repetition][beta]}")
            print(f"    Gamma Term: {gamma_term[repetition][beta]}")
            print(f"    Ramp Loss: {ramp_losses[repetition][beta]}")
            print(f"    Test Bound: {test_bounds[repetition][beta]}")
            print(f"    ZO Test Error: {zo_test_errors[repetition][beta]}")

    # Plot average and std test bound and ZO test error over the repetitions against beta
    import matplotlib.pyplot as plt

    betas = list(test_bounds[1].keys())
    average_test_bound = {beta: np.mean([test_bounds[rep][beta] for rep in test_bounds]) for beta in betas}
    std_test_bound = {beta: np.std([test_bounds[rep][beta] for rep in test_bounds]) for beta in betas}
    average_zo_test_error = {beta: np.mean([zo_test_errors[rep][beta] for rep in zo_test_errors]) for beta in betas}
    std_zo_test_error = {beta: np.std([zo_test_errors[rep][beta] for rep in zo_test_errors]) for beta in betas}


    plt.figure(figsize=(12, 6))
    plt.errorbar(betas[1:], list(average_test_bound.values())[1:], yerr=list(std_test_bound.values())[1:], label="Test Bound", marker="o")
    plt.errorbar(betas[1:], list(average_zo_test_error.values())[1:], yerr=list(std_zo_test_error.values())[1:], label="Zero-One Test Error", marker="x")
    plt.xlabel("Beta")
    plt.ylabel("Value")
    plt.title("Test Bound and Zero-One Test Error")
    plt.legend()
    plt.grid()
    plt.show()
