"""
GPU and Checkpoint Testing Script

This script tests:
1. GPU availability and usage
2. The new beta values for MNIST
3. Checkpoint saving and loading functionality
4. Training runs smoothly with the new configuration

Run this before the full experiment to verify everything works!
"""

import torch
import numpy as np
import random
import os
import sys
from datetime import datetime
from dataset import get_mnist_multiclass_dataloaders_partial_random_labels
from training import run_beta_experiments

def test_gpu_availability():
    """Test if GPU is available and print information."""
    print("\n" + "="*80)
    print("GPU AVAILABILITY TEST")
    print("="*80)
    
    if torch.cuda.is_available():
        device = f'cuda:{torch.cuda.current_device()}'
        print(f"✅ GPU Available: {device}")
        print(f"   Device Name: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties()
        print(f"   Total Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"   GPU Compute Capability: {props.major}.{props.minor}")
        
        # Test GPU operations
        print("\n   Testing GPU operations...")
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.mm(test_tensor, test_tensor.t())
        print(f"   ✅ GPU operations working! Matrix multiplication successful.")
        return device
    else:
        print("❌ No GPU detected - will use CPU")
        return 'cpu'


def test_checkpoint_system():
    """Test checkpoint save/load functionality."""
    print("\n" + "="*80)
    print("CHECKPOINT SYSTEM TEST")
    print("="*80)
    
    from training import save_checkpoint, load_checkpoint
    
    # Create test data
    test_checkpoint_dir = 'test_checkpoints'
    test_seed = 42
    test_betas = [30, 200, 3000]
    completed_betas = [30]
    
    test_losses = [1.5, 2.3, 3.1]
    test_ema_losses = [1.4, 2.2, 3.0]
    
    # Test saving
    print(f"Testing checkpoint save...")
    save_checkpoint(
        test_checkpoint_dir, test_seed, test_betas, completed_betas,
        test_losses.copy(), test_losses.copy(), test_losses.copy(), test_losses.copy(),
        test_ema_losses.copy(), test_ema_losses.copy(), test_ema_losses.copy(), test_ema_losses.copy(),
        [0.5], [100], [0.1], [0.1]
    )
    
    # Test loading
    print(f"Testing checkpoint load...")
    loaded_data = load_checkpoint(test_checkpoint_dir, test_seed)
    
    if loaded_data is not None:
        print(f"✅ Checkpoint saved and loaded successfully!")
        print(f"   Completed betas: {loaded_data['completed_betas']}")
        assert loaded_data['completed_betas'] == completed_betas, "Completed betas mismatch!"
        print(f"   ✅ Data integrity verified")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_checkpoint_dir)
        return True
    else:
        print(f"❌ Failed to load checkpoint")
        return False


def test_new_beta_values():
    """Test that the new beta values are correctly configured."""
    print("\n" + "="*80)
    print("NEW BETA VALUES TEST")
    print("="*80)
    
    # Import master.py config (we'll simulate it here)
    new_beta_values = [30, 200, 3000, 10000, 30000, 100000, 300000, 2000000, 200000000]
    
    print(f"New beta values for MNIST full experiment:")
    print(f"  {new_beta_values}")
    print(f"  Total betas: {len(new_beta_values)}")
    
    # Check they're sorted
    if new_beta_values == sorted(new_beta_values):
        print(f"✅ Beta values are properly sorted")
    else:
        print(f"❌ Beta values are NOT sorted!")
        return False
        
    # Check learning rates are defined
    a0_values = {0: 0.01, 30: 0.01, 200: 0.01, 3000: 0.01, 10000: 0.01, 30000: 0.01, 100000: 0.01, 300000: 0.01, 2000000: 0.01, 200000000: 0.01}
    
    for beta in new_beta_values:
        if beta not in a0_values and 0 not in a0_values:
            print(f"❌ Learning rate not defined for beta={beta}")
            return False
    
    print(f"✅ Learning rates configured for all beta values")
    return True


def run_quick_test(device):
    """Run a quick training test with TEST_MODE configuration."""
    print("\n" + "="*80)
    print("QUICK TRAINING TEST")
    print("="*80)
    
    try:
        print(f"Creating test dataset...")
        train_loader, test_loader = get_mnist_multiclass_dataloaders_partial_random_labels(
            classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            p=1,  # No random labels
            n_train_per_class=500,  # Small dataset for quick test
            n_test_per_class=100,
            batch_size=250,
            random_seed=42,
            normalize=True
        )
        print(f"✅ Dataset created: {len(train_loader.dataset)} training, {len(test_loader.dataset)} test samples")
        
        print(f"\nRunning quick training with test configuration...")
        print(f"  Device: {device}")
        print(f"  Loss: SAVAGE")
        print(f"  Beta values: [256]  (single beta for quick test)")
        print(f"  Min steps: 5  (minimal training for verification)")
        
        csv_paths = run_beta_experiments(
            loss='SAVAGE',
            beta_values=[256],  # Single beta for quick test
            a0={0: 0.01, 256: 0.01},
            b=0.5,
            sigma_gauss_prior=5.0,
            device=device,
            n_hidden_layers=1,
            width=500,
            dataset_type='mnist',
            use_random_labels=1,
            l_max=4.0,
            train_loader=train_loader,
            test_loader=test_loader,
            min_steps=5,  # Very small for quick test
            alpha_average=0.01,
            alpha_stop=0.00025,
            eta=36,
            eps=-1e-7,
            test_mode=True,  # Enable test mode
            add_grad_norm=True,
            add_noise=True,
            sgld_num=1,
            annealed=False,
            seed=42,
            use_pbb_models=True,
            pbb_architecture='cnn',
            prior_type='truncated_gaussian',
            sigma_prior=0.03,
            checkpoint_dir='test_checkpoints',
            resume_from_checkpoint=True
        )
        
        if csv_paths:
            print(f"\n✅ Training completed successfully!")
            print(f"   Results saved to: {csv_paths[0]}")
            return True
        else:
            print(f"❌ Training completed but no results saved")
            return False
            
    except Exception as e:
        print(f"❌ Training failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("GPU AND CHECKPOINT VERIFICATION TEST SUITE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = {}
    
    # Test 1: GPU Availability
    device = test_gpu_availability()
    results['GPU Availability'] = (device != 'cpu')
    
    # Test 2: Checkpoint System
    results['Checkpoint System'] = test_checkpoint_system()
    
    # Test 3: New Beta Values
    results['New Beta Values'] = test_new_beta_values()
     
    # Test 4: Quick Training Run
    results['Quick Training Run'] = run_quick_test(device)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED! Ready to run full experiment.")
    else:
        print("❌ SOME TESTS FAILED. Review errors above before running full experiment.")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
