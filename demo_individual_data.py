#!/usr/bin/env python3
"""
Demonstration script showing the individual repetition data saving functionality.
"""

print("="*70)
print("INDIVIDUAL REPETITION DATA SAVING - DEMONSTRATION")
print("="*70)

print("\n✅ IMPLEMENTATION COMPLETED!")
print("\nThe save_results_to_file() function in bounds.py has been enhanced to include:")
print("\n1. INDIVIDUAL REPETITION DATA section")
print("2. For each beta value:")
print("   - Complete table showing each repetition's final errors")
print("   - Train BCE, Test BCE, Train 0-1, Test 0-1 for each run")
print("   - Computed generalization errors (Test - Train) for each run")
print("   - Summary statistics (Mean, Std, Min, Max) for all metrics")

print("\n" + "="*50)
print("WHAT'S SAVED:")
print("="*50)
print("✓ Final train BCE loss for all repetitions")
print("✓ Final test BCE loss for all repetitions") 
print("✓ Final train zero-one loss for all repetitions")
print("✓ Final test zero-one loss for all repetitions")
print("✓ Computed generalization errors for each repetition")
print("✓ Summary statistics for each beta value")

print("\n" + "="*50)
print("EXAMPLE OUTPUT:")
print("="*50)
print("""
Beta = 10:
----------------------------------------
Number of repetitions: 5

Rep  Train BCE    Test BCE     Train 0-1    Test 0-1     BCE Gen.Err  0-1 Gen.Err 
---- ------------ ------------ ------------ ------------ ------------ ------------
1    0.098325     0.173999     0.120000     0.180000     0.075674     0.060000    
2    0.060075     0.092794     0.060000     0.110000     0.032719     0.050000    
3    0.160001     0.167412     0.160000     0.180000     0.007411     0.020000    
4    0.140042     0.195789     0.140000     0.200000     0.055747     0.060000    
5    0.147790     0.204883     0.160000     0.220000     0.057094     0.060000    
---- ------------ ------------ ------------ ------------ ------------ ------------
Mean 0.121245     0.166972     0.128000     0.178000     0.045727     0.050000    
Std  0.041314     0.044213     0.041473     0.041473     0.026279     0.017321    
Min  0.060075     0.092794     0.060000     0.110000     0.007417     0.020000    
Max  0.160001     0.204883     0.160000     0.220000     0.075659     0.060000    
""")

print("\n" + "="*50)
print("USAGE:")
print("="*50)
print("Just run your experiments as usual!")
print("The individual repetition data will automatically be")
print("included in the generated .txt results file.")
print("\nLook for the section:")
print("'INDIVIDUAL REPETITION DATA (Final Train/Test Errors)'")

print("\n" + "="*70)
print("✅ READY TO USE!")
print("="*70)
