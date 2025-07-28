#!/usr/bin/env python3
"""
Test script to verify that individual repetition data is being saved correctly.
"""

import sys
import os
sys.path.append('.')

# Set test flags
import main
main.TEST_MODE = True
main.USE_RANDOM_LABELS = False

# Run a quick test
if __name__ == "__main__":
    print("Testing individual repetition data saving...")
    print("Running quick experiment with TEST_MODE=True")
    print("This will generate a results file with individual repetition data.")
    
    # Import and run main
    from main import main as run_main
    run_main()
    
    print("\n" + "="*70)
    print("✅ Test completed!")
    print("Check the generated .txt file in the results/ folder")
    print("Look for the 'INDIVIDUAL REPETITION DATA' section")
    print("="*70)
