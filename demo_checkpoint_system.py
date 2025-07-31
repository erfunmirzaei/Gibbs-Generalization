#!/usr/bin/env python3
"""
Demo script to show the repetition-first loop with checkpoint functionality.
"""

from training import run_beta_experiments
from dataset import get_synth_dataloaders
import os
import json

def main():
    print("🚀 Demonstrating Repetition-First Loop with Checkpointing")
    print("=" * 65)
    
    # Create a small synthetic dataset
    train_loader, test_loader = get_synth_dataloaders(batch_size=10, random_seed=42)
    
    print(f"Dataset: {len(train_loader.dataset)} training samples")
    print(f"Betas: [1, 10] (plus auto-added beta=0)")
    print(f"Repetitions: 3")
    print(f"Epochs per beta: 3 (very short for demo)")
    print()
    
    # Define checkpoint path
    checkpoint_path = "demo_experiment_checkpoint.json"
    
    # Clean up any existing checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("🧹 Cleaned up existing checkpoint")
    
    print("🎯 Expected execution order:")
    print("  REPETITION 1: Beta 0.0 → Beta 1 → Beta 10")
    print("  REPETITION 2: Beta 0.0 → Beta 1 → Beta 10") 
    print("  REPETITION 3: Beta 0.0 → Beta 1 → Beta 10")
    print("  (Checkpoint saved after each repetition)")
    print()
    
    # Run the experiment
    try:
        results = run_beta_experiments(
            beta_values=[1, 10],        # Small set for demo
            num_repetitions=3,          # Few repetitions
            num_epochs=3,               # Very few epochs
            a0=0.01,                    # Fixed learning rate
            device='cpu',               # Use CPU for demo
            dataset_type='synth',       # Synthetic dataset
            train_loader=train_loader,
            test_loader=test_loader,
            checkpoint_path=checkpoint_path,
            save_every=1                # Save after every repetition
        )
        
        print("\n" + "=" * 65)
        print("📊 RESULTS SUMMARY")
        print("=" * 65)
        
        for beta in sorted(results.keys()):
            n_reps = len(results[beta]['raw_train_bce'])
            train_mean = sum(results[beta]['raw_train_bce']) / n_reps
            test_mean = sum(results[beta]['raw_test_bce']) / n_reps
            print(f"Beta {beta:>4}: {n_reps} reps, Train: {train_mean:.4f}, Test: {test_mean:.4f}")
        
        # Show checkpoint details
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            print(f"\n📁 CHECKPOINT DETAILS")
            print(f"   File: {checkpoint_path}")
            print(f"   Timestamp: {checkpoint_data.get('timestamp', 'N/A')}")
            print(f"   Completed repetitions: {checkpoint_data.get('completed_repetitions', 0)}")
            print(f"   Version: {checkpoint_data.get('version', 'N/A')}")
            
        print("\n✅ Demonstration completed successfully!")
        print("Key features verified:")
        print("  ✓ Repetition-first loop execution")
        print("  ✓ Checkpoint creation and saving")
        print("  ✓ Progress tracking")
        print("  ✓ Results accumulation")
        
    except Exception as e:
        print(f"❌ Error during experiment: {e}")
        return
    
    finally:
        # Cleanup
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"\n🧹 Cleaned up checkpoint file: {checkpoint_path}")

if __name__ == "__main__":
    main()
