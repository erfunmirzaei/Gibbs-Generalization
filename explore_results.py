#!/usr/bin/env python3
"""
Utility script to explore and manage saved experimental results.
"""

import os
import json
from datetime import datetime
from results_manager import load_existing_results, hyperparameters_match, generate_hyperparameter_hash

def list_saved_results():
    """List all saved results files with their metadata."""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("No results directory found.")
        return
    
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not json_files:
        print("No JSON results files found.")
        return
    
    print("Saved Experimental Results")
    print("=" * 70)
    print(f"{'Filename':<30} {'Beta Values':<20} {'Reps':<6} {'Dataset':<10} {'Hash':<12}")
    print("-" * 70)
    
    for filename in sorted(json_files):
        filepath = os.path.join(results_dir, filename)
        hyperparams, results = load_existing_results(filepath)
        
        if hyperparams and results:
            beta_values = hyperparams.get('beta_values', [])
            beta_str = f"{len(beta_values)} values" if len(beta_values) > 3 else str(beta_values)
            total_reps = sum(len(results[beta]['raw_train_bce']) for beta in results.keys())
            dataset = hyperparams.get('dataset_type', 'unknown')
            hash_val = generate_hyperparameter_hash(hyperparams)[:8]
            
            print(f"{filename:<30} {beta_str:<20} {total_reps:<6} {dataset:<10} {hash_val:<12}")
        else:
            print(f"{filename:<30} {'ERROR':<20} {'?':<6} {'?':<10} {'?':<12}")


def show_result_details(filename):
    """Show detailed information about a specific results file."""
    filepath = f"results/{filename}" if not filename.startswith('results/') else filename
    
    hyperparams, results = load_existing_results(filepath)
    
    if not hyperparams or not results:
        print(f"❌ Could not load results from {filepath}")
        return
    
    print(f"Detailed Results for: {filename}")
    print("=" * 70)
    
    # Show metadata
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        metadata = data.get('metadata', {})
        print(f"Created: {metadata.get('created_timestamp', 'Unknown')}")
        print(f"Total experiments: {metadata.get('total_experiments', 'Unknown')}")
        print(f"Version: {metadata.get('version', 'Unknown')}")
    except:
        pass
    
    print(f"Hyperparameter hash: {generate_hyperparameter_hash(hyperparams)}")
    print()
    
    # Show hyperparameters
    print("Hyperparameters:")
    print("-" * 30)
    for key, value in sorted(hyperparams.items()):
        if key == 'mnist_classes' and isinstance(value, list) and len(value) == 2:
            print(f"  {key:<20}: {value}")
        elif isinstance(value, dict) and len(str(value)) > 50:
            print(f"  {key:<20}: <dict with {len(value)} items>")
        else:
            print(f"  {key:<20}: {value}")
    print()
    
    # Show results summary
    print("Results Summary:")
    print("-" * 50)
    print(f"{'Beta':<8} {'Reps':<6} {'Train BCE':<12} {'Test BCE':<12} {'Gen Error':<12}")
    print("-" * 50)
    
    for beta in sorted(results.keys()):
        beta_results = results[beta]
        num_reps = len(beta_results['raw_train_bce'])
        train_bce = beta_results['train_bce_mean']
        test_bce = beta_results['test_bce_mean']
        gen_error = test_bce - train_bce
        
        print(f"{beta:<8.0f} {num_reps:<6} {train_bce:<12.4f} {test_bce:<12.4f} {gen_error:<12.4f}")


def compare_hyperparameters(file1, file2):
    """Compare hyperparameters between two results files."""
    filepath1 = f"results/{file1}" if not file1.startswith('results/') else file1
    filepath2 = f"results/{file2}" if not file2.startswith('results/') else file2
    
    hyperparams1, _ = load_existing_results(filepath1)
    hyperparams2, _ = load_existing_results(filepath2)
    
    if not hyperparams1:
        print(f"❌ Could not load hyperparameters from {filepath1}")
        return
    if not hyperparams2:
        print(f"❌ Could not load hyperparameters from {filepath2}")
        return
    
    print(f"Hyperparameter Comparison")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print("=" * 70)
    
    match = hyperparameters_match(hyperparams1, hyperparams2)
    print(f"Hyperparameters match: {'✅ YES' if match else '❌ NO'}")
    print()
    
    all_keys = set(hyperparams1.keys()) | set(hyperparams2.keys())
    
    print(f"{'Parameter':<25} {'File 1':<25} {'File 2':<25} {'Match':<5}")
    print("-" * 80)
    
    for key in sorted(all_keys):
        val1 = hyperparams1.get(key, "NOT_FOUND")
        val2 = hyperparams2.get(key, "NOT_FOUND")
        
        # Truncate long values
        str1 = str(val1)[:24] if len(str(val1)) <= 24 else str(val1)[:21] + "..."
        str2 = str(val2)[:24] if len(str(val2)) <= 24 else str(val2)[:21] + "..."
        
        match_symbol = "✅" if val1 == val2 else "❌"
        
        print(f"{key:<25} {str1:<25} {str2:<25} {match_symbol:<5}")


def find_compatible_results(target_file):
    """Find all results files with compatible hyperparameters."""
    target_filepath = f"results/{target_file}" if not target_file.startswith('results/') else target_file
    target_hyperparams, _ = load_existing_results(target_filepath)
    
    if not target_hyperparams:
        print(f"❌ Could not load hyperparameters from {target_filepath}")
        return
    
    results_dir = 'results'
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    compatible_files = []
    
    for filename in json_files:
        if filename == os.path.basename(target_file):
            continue
            
        filepath = os.path.join(results_dir, filename)
        hyperparams, _ = load_existing_results(filepath)
        
        if hyperparams and hyperparameters_match(target_hyperparams, hyperparams):
            compatible_files.append(filename)
    
    print(f"Files compatible with {target_file}:")
    print("=" * 50)
    
    if compatible_files:
        for filename in compatible_files:
            print(f"  ✅ {filename}")
        print(f"\nThese {len(compatible_files)} files could be merged together.")
    else:
        print("  No compatible files found.")


def main():
    """Main function with command-line interface."""
    import sys
    
    if len(sys.argv) == 1:
        print("Results Manager - Explore and manage experimental results")
        print("=" * 60)
        print()
        print("Usage:")
        print("  python explore_results.py list                    # List all results")
        print("  python explore_results.py show <filename>         # Show details")
        print("  python explore_results.py compare <file1> <file2> # Compare hyperparams")
        print("  python explore_results.py compatible <filename>   # Find compatible files")
        print()
        list_saved_results()
        
    elif sys.argv[1] == 'list':
        list_saved_results()
        
    elif sys.argv[1] == 'show' and len(sys.argv) > 2:
        show_result_details(sys.argv[2])
        
    elif sys.argv[1] == 'compare' and len(sys.argv) > 3:
        compare_hyperparameters(sys.argv[2], sys.argv[3])
        
    elif sys.argv[1] == 'compatible' and len(sys.argv) > 2:
        find_compatible_results(sys.argv[2])
        
    else:
        print("Invalid command. Use without arguments to see usage.")


if __name__ == "__main__":
    main()
