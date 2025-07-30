#!/usr/bin/env python3
"""Simple script to test JSON loading and plotting."""

import json
import sys
import os

print("🔍 Testing JSON file loading...")

json_file = '/home/emirzaei/Gibbs-Generalization/results/sgld_mnist_5clsv5cls_hce1b10d8d09f.json'

try:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("✅ JSON loaded successfully!")
    print(f"Metadata: {data['metadata']}")
    print(f"Beta values: {data['hyperparameters']['beta_values']}")
    
    results = data['results']
    print(f"Results keys: {list(results.keys())}")
    
    # Test accessing a specific result
    if '250' in results:
        print(f"Beta 250 train BCE mean: {results['250']['train_bce_mean']}")
        print(f"Beta 250 test BCE mean: {results['250']['test_bce_mean']}")
    
    print("✅ Basic data access successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
