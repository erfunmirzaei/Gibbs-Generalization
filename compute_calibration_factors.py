"""
Compute calibration factors for multiple CSV files.
"""
from plot import main, calibrate


def compute_calibration_factors(csv_filenames):
    """
    Compute calibration factors for a list of CSV files.
    
    Args:
        csv_filenames: List of CSV file names (10 files)
        
    Returns:
        list: List of calibration factors (one per CSV file)
    """
    calibration_factors = []
    
    for csv_filename in csv_filenames:
        print(f"Processing: {csv_filename}")
        
        # Read the CSV file using the main function from plot.py
        betas, list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, \
            list_test_01_losses, list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, \
            list_EMA_train_01_losses, list_EMA_test_01_losses, n_samples = main(csv_filename)
        
        # Extract sample size
        samplesize = n_samples[0]
        
        # Compute calibration factor for this file
        factor = calibrate(betas, list_EMA_train_BCE_losses, list_EMA_train_01_losses, samplesize, thresh=0.5)
        
        calibration_factors.append(factor)
        print(f"   Calibration factor: {factor:.4f}")
    
    return calibration_factors


if __name__ == "__main__":
    # Example usage with MRL1W500ULA2kLR001SAVAGE MNIST CSV files
    # csv_files = [
        # "MRL1W500ULA2kLR001SAVAGE_S1123_20260327-033154.csv",
        # "MRL1W500ULA2kLR001SAVAGE_S3232_20260327-124035.csv",
        # "MRL1W500ULA2kLR001SAVAGE_S42_20260326-220938.csv",
        # "MRL1W500ULA2kLR001SAVAGE_S52_20260327-075748.csv",
        # "MRL1W500ULA2kLR001SAVAGE_S74_20260327-134338.csv",
        # "MRL1W500ULA2kLR001SAVAGE_S787_20260327-034652.csv",

    # ]
    # CIFAR-10 values (commented out)
    # csv_files = [
    #     "CRL2W1500SGLD8kLR0005BBCE_S1_20260328-213924.csv",
    #     "CRL2W1500SGLD8kLR0005BBCE_S2_20260328-231021.csv",
    #     "CRL2W1500SGLD8kLR0005BBCE_S3_20260329-004506.csv",
    #     "CRL2W1500SGLD8kLR0005BBCE_S4_20260328-215710.csv",
    #     "CRL2W1500SGLD8kLR0005BBCE_S5_20260328-233704.csv",
    #     "CRL2W1500SGLD8kLR0005BBCE_S6_20260329-035728.csv",
    #     "CRL2W1500SGLD8kLR0005BBCE_S7_20260329-033912.csv",
    #     "CRL2W1500SGLD8kLR0005BBCE_S8_20260329-051502.csv",
    #     "CRL2W1500SGLD8kLR0005BBCE_S9_20260329-054245.csv",
    #     "CRL2W1500SGLD8kLR0005BBCE_S10_20260329-073227.csv",
    # ]
    # MNIST values (commented out)
    csv_files = [
        "MRL2W1000SGLD8kLR001BBCE_S1_20260329-054600.csv",
        "MRL2W1000SGLD8kLR001BBCE_S2_20260329-062018.csv",
        "MRL2W1000SGLD8kLR001BBCE_S3_20260329-065455.csv",
        "MRL2W1000SGLD8kLR001BBCE_S4_20260329-073118.csv",
        "MRL2W1000SGLD8kLR001BBCE_S5_20260329-080726.csv",
        # "MRL2W1000SGLD8kLR001BBCE_S6_20260329-081154.csv",
        "MRL2W1000SGLD8kLR001BBCE_S7_20260329-085010.csv",
        "MRL2W1000SGLD8kLR001BBCE_S8_20260329-092410.csv",
        "MRL2W1000SGLD8kLR001BBCE_S9_20260329-100427.csv",
        "MRL2W1000SGLD8kLR001BBCE_S10_20260329-103826.csv",
    ]
    
    # Compute calibration factors
    factors = compute_calibration_factors(csv_files)
    
    # Print results
    print("\n" + "="*50)
    print("Calibration Factors Summary:")
    print("="*50)
    for csv_file, factor in zip(csv_files, factors):
        print(f"{csv_file}: {factor:.4f}")
    print(f"\nMean calibration factor: {sum(factors)/len(factors):.4f}")
    print(f"Std calibration factor: {(sum((x - sum(factors)/len(factors))**2 for x in factors)/len(factors))**0.5:.4f}")
