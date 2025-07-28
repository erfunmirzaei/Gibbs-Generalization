#!/usr/bin/env python3
"""
Demo script showing the new descriptive filename generation system.

This script demonstrates how experimental parameters are automatically 
encoded into descriptive filenames for easy identification and organization.
"""

from bounds import generate_filename


def demo_filename_generation():
    """Demonstrate the filename generation with various parameter combinations."""
    
    print("="*70)
    print("DESCRIPTIVE FILENAME GENERATION DEMO")
    print("="*70)
    
    # Example 1: Full experiment
    print("\n1. Full Experiment:")
    print("   Parameters: beta=[1,10,30,50,70,100,200], reps=30, epochs=10k, lr=0.1, sigma=10")
    filename1 = generate_filename(
        beta_values=[1, 10, 30, 50, 70, 100, 200],
        num_repetitions=30,
        num_epochs=10000,
        a0=0.1,
        sigma_gauss_prior=10,
        dataset_type='synth',
        file_type='results',
        extension='txt'
    )
    print(f"   Generated: {filename1}")
    
    # Example 2: Test mode
    print("\n2. Test Mode:")
    print("   Parameters: beta=[1,10,50,200], reps=5, epochs=100, lr=0.1, sigma=10")
    filename2 = generate_filename(
        beta_values=[1, 10, 50, 200],
        num_repetitions=5,
        num_epochs=100,
        a0=0.1,
        sigma_gauss_prior=10,
        dataset_type='synth',
        file_type='results',
        extension='txt'
    )
    print(f"   Generated: {filename2}")
    
    # Example 3: Different learning rate
    print("\n3. Different Learning Rate:")
    print("   Parameters: beta=[1,50,200], reps=20, epochs=5k, lr=0.01, sigma=5")
    filename3 = generate_filename(
        beta_values=[1, 50, 200],
        num_repetitions=20,
        num_epochs=5000,
        a0=0.01,
        sigma_gauss_prior=5,
        dataset_type='synth',
        file_type='results',
        extension='txt'
    )
    print(f"   Generated: {filename3}")
    
    # Example 4: Plot file
    print("\n4. Corresponding Plot File:")
    print("   Same parameters as #3, but for plot")
    filename4 = generate_filename(
        beta_values=[1, 50, 200],
        num_repetitions=20,
        num_epochs=5000,
        a0=0.01,
        sigma_gauss_prior=5,
        dataset_type='synth',
        file_type='plot',
        extension='png'
    )
    print(f"   Generated: {filename4}")
    
    # Example 5: Scientific notation learning rate
    print("\n5. Very Small Learning Rate:")
    print("   Parameters: beta=[1,10], reps=10, epochs=1k, lr=1e-4, sigma=1")
    filename5 = generate_filename(
        beta_values=[1, 10],
        num_repetitions=10,
        num_epochs=1000,
        a0=1e-4,
        sigma_gauss_prior=1,
        dataset_type='synth',
        file_type='results',
        extension='txt'
    )
    print(f"   Generated: {filename5}")
    
    print("\n" + "="*70)
    print("FILENAME STRUCTURE BREAKDOWN")
    print("="*70)
    print("Format: sgld_{type}_{dataset}_beta{min}-{max}_lr{lr}_sigma{prior}_ep{epochs}_rep{reps}.{ext}")
    print()
    print("Components:")
    print("  - sgld: Project identifier")
    print("  - {type}: 'results', 'plot', 'analysis', etc.")
    print("  - {dataset}: 'synth', 'mnist', etc.")
    print("  - beta{min}-{max}: Range of beta values tested")
    print("  - lr{lr}: Learning rate (periods replaced with 'p')")
    print("  - sigma{prior}: Gaussian prior sigma parameter")
    print("  - ep{epochs}: Number of epochs ('k' for thousands)")
    print("  - rep{reps}: Number of repetitions per beta")
    print("  - {ext}: File extension")
    
    print("\n" + "="*70)
    print("BENEFITS")
    print("="*70)
    print("✓ No filename conflicts between different experiments")
    print("✓ Parameters visible at a glance")
    print("✓ Easy to organize and find specific results")
    print("✓ Reproducibility information embedded in filename")
    print("✓ Automatic generation - no manual naming required")


if __name__ == "__main__":
    demo_filename_generation()
