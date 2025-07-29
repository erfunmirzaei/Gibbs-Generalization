"""
Enhanced results saving and loading system with hyperparameter tracking.

This module provides functionality to:
1. Save results with complete hyperparameter information
2. Load existing results and check for hyperparameter compatibility
3. Merge new results with existing ones when hyperparameters match
4. Generate comprehensive plots from merged results
"""

import json
import os
import hashlib
import time
from datetime import datetime
import numpy as np
import torch


def generate_hyperparameter_hash(hyperparams):
    """
    Generate a unique hash from hyperparameters for consistent file naming.
    
    Args:
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        str: SHA256 hash of hyperparameters
    """
    # Create a sorted string representation of hyperparameters
    def serialize_param(param):
        if isinstance(param, dict):
            return sorted(param.items())
        elif isinstance(param, list):
            return sorted(param) if all(isinstance(x, (int, float, str)) for x in param) else param
        elif callable(param):
            return str(param.__name__) if hasattr(param, '__name__') else str(param)
        else:
            return param
    
    serialized = {}
    for key, value in sorted(hyperparams.items()):
        serialized[key] = serialize_param(value)
    
    # Convert to JSON string for hashing
    param_string = json.dumps(serialized, sort_keys=True, default=str)
    return hashlib.sha256(param_string.encode()).hexdigest()[:12]  # Use first 12 chars


def create_hyperparameter_dict(**kwargs):
    """
    Create a standardized hyperparameter dictionary.
    
    Returns:
        dict: Standardized hyperparameter dictionary
    """
    return {
        'beta_values': kwargs.get('beta_values', []),
        'num_repetitions': kwargs.get('num_repetitions', 1),
        'num_epochs': kwargs.get('num_epochs', 1000),
        'a0': kwargs.get('a0', 0.1),
        'b': kwargs.get('b', 0.5),
        'sigma_gauss_prior': kwargs.get('sigma_gauss_prior', 1000),
        'device': kwargs.get('device', 'cpu'),
        'dataset_type': kwargs.get('dataset_type', 'synth'),
        'use_random_labels': kwargs.get('use_random_labels', False),
        'l_max': kwargs.get('l_max', 4.0),
        'mnist_classes': kwargs.get('mnist_classes', None),
        'train_dataset_size': kwargs.get('train_dataset_size', None),
        'test_dataset_size': kwargs.get('test_dataset_size', None),
        'batch_size': kwargs.get('batch_size', 128),
        'random_seed': kwargs.get('random_seed', 42),
        'normalize': kwargs.get('normalize', True)
    }


def save_results_with_hyperparams(results, hyperparams, base_filename=None):
    """
    Save results with hyperparameters in a structured format.
    
    Args:
        results: Dictionary containing experimental results
        hyperparams: Dictionary of hyperparameters
        base_filename: Optional base filename (will add hash and extension)
        
    Returns:
        str: Full path to saved file
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate filename based on hyperparameters
    param_hash = generate_hyperparameter_hash(hyperparams)
    
    if base_filename is None:
        dataset_str = hyperparams.get('dataset_type', 'unknown')
        if dataset_str == 'mnist' and hyperparams.get('mnist_classes'):
            classes = hyperparams['mnist_classes']
            if isinstance(classes, list) and len(classes) == 2:
                if isinstance(classes[0], list):
                    # Grouped classes - create a short description
                    g0_desc = f"{len(classes[0])}cls" if len(classes[0]) > 2 else f"{'_'.join(map(str, classes[0]))}"
                    g1_desc = f"{len(classes[1])}cls" if len(classes[1]) > 2 else f"{'_'.join(map(str, classes[1]))}"
                    class_str = f"{g0_desc}v{g1_desc}"
                else:
                    class_str = f"{classes[0]}v{classes[1]}"
                dataset_str += f"_{class_str}"
        
        base_filename = f"sgld_{dataset_str}_h{param_hash}"
    
    filename = f"results/{base_filename}.json"
    
    # Create the data structure to save
    data_to_save = {
        'metadata': {
            'created_timestamp': datetime.now().isoformat(),
            'hyperparameter_hash': param_hash,
            'total_experiments': sum(len(results[beta]['raw_train_bce']) for beta in results.keys()),
            'version': '1.0'
        },
        'hyperparameters': hyperparams,
        'results': {}
    }
    
    # Convert results to JSON-serializable format
    for beta, beta_results in results.items():
        data_to_save['results'][str(beta)] = {
            'train_bce_mean': float(beta_results['train_bce_mean']),
            'train_bce_var': float(beta_results['train_bce_var']),
            'train_bce_std': float(beta_results['train_bce_std']),
            'test_bce_mean': float(beta_results['test_bce_mean']),
            'test_bce_var': float(beta_results['test_bce_var']),
            'test_bce_std': float(beta_results['test_bce_std']),
            'train_01_mean': float(beta_results['train_01_mean']),
            'train_01_var': float(beta_results['train_01_var']),
            'train_01_std': float(beta_results['train_01_std']),
            'test_01_mean': float(beta_results['test_01_mean']),
            'test_01_var': float(beta_results['test_01_var']),
            'test_01_std': float(beta_results['test_01_std']),
            'raw_train_bce': beta_results['raw_train_bce'],
            'raw_test_bce': beta_results['raw_test_bce'],
            'raw_train_01': beta_results['raw_train_01'],
            'raw_test_01': beta_results['raw_test_01'],
            'num_repetitions': len(beta_results['raw_train_bce'])
        }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Results saved to: {filename}")
    print(f"Hyperparameter hash: {param_hash}")
    return filename


def load_existing_results(filename):
    """
    Load existing results from file.
    
    Args:
        filename: Path to the results file
        
    Returns:
        tuple: (hyperparams, results) or (None, None) if file doesn't exist
    """
    if not os.path.exists(filename):
        return None, None
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        hyperparams = data['hyperparameters']
        
        # Convert results back to the expected format
        results = {}
        for beta_str, beta_data in data['results'].items():
            beta = float(beta_str)
            results[beta] = {
                'train_bce_mean': beta_data['train_bce_mean'],
                'train_bce_var': beta_data['train_bce_var'],
                'train_bce_std': beta_data['train_bce_std'],
                'test_bce_mean': beta_data['test_bce_mean'],
                'test_bce_var': beta_data['test_bce_var'],
                'test_bce_std': beta_data['test_bce_std'],
                'train_01_mean': beta_data['train_01_mean'],
                'train_01_var': beta_data['train_01_var'],
                'train_01_std': beta_data['train_01_std'],
                'test_01_mean': beta_data['test_01_mean'],
                'test_01_var': beta_data['test_01_var'],
                'test_01_std': beta_data['test_01_std'],
                'raw_train_bce': beta_data['raw_train_bce'],
                'raw_test_bce': beta_data['raw_test_bce'],
                'raw_train_01': beta_data['raw_train_01'],
                'raw_test_01': beta_data['raw_test_01']
            }
        
        return hyperparams, results
    
    except Exception as e:
        print(f"Error loading results from {filename}: {e}")
        return None, None


def hyperparameters_match(hyperparams1, hyperparams2, ignore_keys=None):
    """
    Check if two hyperparameter dictionaries match.
    
    Args:
        hyperparams1: First hyperparameter dictionary
        hyperparams2: Second hyperparameter dictionary
        ignore_keys: List of keys to ignore in comparison
        
    Returns:
        bool: True if hyperparameters match
    """
    if ignore_keys is None:
        ignore_keys = ['train_dataset_size', 'test_dataset_size']  # These might vary slightly
    
    # Create copies without ignored keys
    h1 = {k: v for k, v in hyperparams1.items() if k not in ignore_keys}
    h2 = {k: v for k, v in hyperparams2.items() if k not in ignore_keys}
    
    # Generate hashes for comparison
    hash1 = generate_hyperparameter_hash(h1)
    hash2 = generate_hyperparameter_hash(h2)
    
    return hash1 == hash2


def merge_results(existing_results, new_results):
    """
    Merge new results with existing results.
    
    Args:
        existing_results: Dictionary of existing results
        new_results: Dictionary of new results to merge
        
    Returns:
        dict: Merged results dictionary
    """
    merged = {}
    
    # Get all beta values from both result sets
    all_betas = set(existing_results.keys()) | set(new_results.keys())
    
    for beta in all_betas:
        if beta in existing_results and beta in new_results:
            # Merge results for this beta
            existing = existing_results[beta]
            new = new_results[beta]
            
            # Combine raw data
            combined_train_bce = existing['raw_train_bce'] + new['raw_train_bce']
            combined_test_bce = existing['raw_test_bce'] + new['raw_test_bce']
            combined_train_01 = existing['raw_train_01'] + new['raw_train_01']
            combined_test_01 = existing['raw_test_01'] + new['raw_test_01']
            
            # Recompute statistics
            merged[beta] = {
                'train_bce_mean': np.mean(combined_train_bce),
                'train_bce_var': np.var(combined_train_bce),
                'train_bce_std': np.std(combined_train_bce),
                'test_bce_mean': np.mean(combined_test_bce),
                'test_bce_var': np.var(combined_test_bce),
                'test_bce_std': np.std(combined_test_bce),
                'train_01_mean': np.mean(combined_train_01),
                'train_01_var': np.var(combined_train_01),
                'train_01_std': np.std(combined_train_01),
                'test_01_mean': np.mean(combined_test_01),
                'test_01_var': np.var(combined_test_01),
                'test_01_std': np.std(combined_test_01),
                'raw_train_bce': combined_train_bce,
                'raw_test_bce': combined_test_bce,
                'raw_train_01': combined_train_01,
                'raw_test_01': combined_test_01
            }
            
        elif beta in existing_results:
            # Only in existing results
            merged[beta] = existing_results[beta].copy()
        else:
            # Only in new results
            merged[beta] = new_results[beta].copy()
    
    return merged


def save_or_merge_results(results, hyperparams, base_filename=None):
    """
    Save results, merging with existing results if hyperparameters match.
    
    Args:
        results: Dictionary containing experimental results
        hyperparams: Dictionary of hyperparameters
        base_filename: Optional base filename
        
    Returns:
        tuple: (filename, was_merged) - filename of saved results and boolean indicating if results were merged
    """
    # Generate the filename that would be used
    param_hash = generate_hyperparameter_hash(hyperparams)
    
    if base_filename is None:
        dataset_str = hyperparams.get('dataset_type', 'unknown')
        if dataset_str == 'mnist' and hyperparams.get('mnist_classes'):
            classes = hyperparams['mnist_classes']
            if isinstance(classes, list) and len(classes) == 2:
                if isinstance(classes[0], list):
                    # Grouped classes
                    g0_desc = f"{len(classes[0])}cls" if len(classes[0]) > 2 else f"{'_'.join(map(str, classes[0]))}"
                    g1_desc = f"{len(classes[1])}cls" if len(classes[1]) > 2 else f"{'_'.join(map(str, classes[1]))}"
                    class_str = f"{g0_desc}v{g1_desc}"
                else:
                    class_str = f"{classes[0]}v{classes[1]}"
                dataset_str += f"_{class_str}"
        
        base_filename = f"sgld_{dataset_str}_h{param_hash}"
    
    filename = f"results/{base_filename}.json"
    
    # Check if file exists and hyperparameters match
    existing_hyperparams, existing_results = load_existing_results(filename)
    
    if existing_results is not None and hyperparameters_match(existing_hyperparams, hyperparams):
        print(f"Found existing results with matching hyperparameters!")
        print(f"Existing experiments: {sum(len(existing_results[beta]['raw_train_bce']) for beta in existing_results.keys())}")
        print(f"New experiments: {sum(len(results[beta]['raw_train_bce']) for beta in results.keys())}")
        
        # Merge results
        merged_results = merge_results(existing_results, results)
        total_experiments = sum(len(merged_results[beta]['raw_train_bce']) for beta in merged_results.keys())
        print(f"Total experiments after merge: {total_experiments}")
        
        # Save merged results
        save_results_with_hyperparams(merged_results, hyperparams, base_filename)
        return filename, True
        
    else:
        if existing_results is not None:
            print(f"Found existing results but hyperparameters don't match.")
            print(f"Creating new file: {filename}")
        else:
            print(f"No existing results found. Creating new file: {filename}")
        
        # Save new results
        save_results_with_hyperparams(results, hyperparams, base_filename)
        return filename, False


def print_hyperparameter_comparison(hyperparams1, hyperparams2):
    """
    Print a comparison of two hyperparameter dictionaries.
    
    Args:
        hyperparams1: First hyperparameter dictionary
        hyperparams2: Second hyperparameter dictionary
    """
    print("\nHyperparameter Comparison:")
    print("=" * 50)
    
    all_keys = set(hyperparams1.keys()) | set(hyperparams2.keys())
    
    for key in sorted(all_keys):
        val1 = hyperparams1.get(key, "NOT_FOUND")
        val2 = hyperparams2.get(key, "NOT_FOUND")
        
        if val1 == val2:
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} {key:<20}: {val1} | {val2}")
