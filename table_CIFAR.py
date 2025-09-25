import csv
import os
import numpy as np

def read_csv_2_lists(csv_file_path):
    """
    Read CSV file and return data organized by beta.
    
    Args:
        csv_file_path: Path to the CSV file
    Returns:
        list 
    """     
    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        return []
    
    # Initialize data structure
    n_samples = []
    beta_values = []
    list_train_BCE_losses = []
    list_test_BCE_losses = []
    list_train_01_losses = []
    list_test_01_losses = []
    list_EMA_train_BCE_losses = []
    list_EMA_test_BCE_losses = []
    list_EMA_train_01_losses = []
    list_EMA_test_01_losses = []
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        n_samples_idx = headers.index("Sample_size")
        beta_idx = headers.index("Beta")
        train_BCE_idx = headers.index("BCE_Train")
        test_BCE_idx = headers.index("BCE_Test")
        train_01_idx = headers.index("0-1_Train")
        test_01_idx = headers.index("0-1_Test")
        EMA_train_BCE_idx = headers.index("EMA_BCE_Train")
        EMA_test_BCE_idx = headers.index("EMA_BCE_Test")
        EMA_train_01_idx = headers.index("EMA_0-1_Train")
        EMA_test_01_idx = headers.index("EMA_0-1_Test")
        for row in reader:
            if len(row) > 1 and row[0] != 'Summary:':  # Skip empty lines
                n_samples.append(int(row[n_samples_idx])) 
                beta_values.append(float(row[beta_idx]))
                list_train_BCE_losses.append(float(row[train_BCE_idx]))
                list_test_BCE_losses.append(float(row[test_BCE_idx]))
                list_train_01_losses.append(float(row[train_01_idx]))
                list_test_01_losses.append(float(row[test_01_idx]))
                list_EMA_train_BCE_losses.append(float(row[EMA_train_BCE_idx]))
                list_EMA_test_BCE_losses.append(float(row[EMA_test_BCE_idx]))
                list_EMA_train_01_losses.append(float(row[EMA_train_01_idx]))
                list_EMA_test_01_losses.append(float(row[EMA_test_01_idx]))

            elif len(row) > 1 and row[0] == 'Summary:':
                summary_string = ', '.join(row[1:])
                print(f"Summary from CSV: {summary_string}")
    print(f"   Loaded {len(beta_values)} rows")
    return beta_values, list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses, list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses, n_samples



def main(csv_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))

   
    csv_path = os.path.join(script_dir, "csv_EMA", csv_filename)
    # Read the CSV files
    beta_values, list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,\
    list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses, n_samples = read_csv_2_lists(csv_path)

    return beta_values, list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses, n_samples


#if __name__ == "__main__":
#    beta_values, list_train_BCE_losses, list_test_BCE_losses, list_train_01_losses, list_test_01_losses,\
#    list_EMA_train_BCE_losses, list_EMA_test_BCE_losses, list_EMA_train_01_losses, list_EMA_test_01_losses, n_samples = main("csv_filename.csv")


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX



import math
import matplotlib.pyplot as plt
import numpy as np

#Standard functions
def ln (x):
    y = math.log(x)
    return y

# relative entropy of Bernoulli variables
def kl (p,q):
    if p == 0:
        y = ln(1/(1-q))
    else:
        y = p*ln(p/q)+(1-p)*ln((1-p)/(1-q))
    return y


def invert_kl ( p, kl_val ):
              l, u, r = p, 1, 0.5
              while ((u-l) > 1/100000):
                            if kl (p,r) < kl_val:
                                          l = r
                                          r = (r+u)/2
                            else:
                                          u = r
                                          r = (r+l)/2
              return r

def invert_kls ( ps, kl_vals ):
	rs = []
	index = 0
	while len ( rs ) < len ( ps ):
		rs.append ( invert_kl ( ps[index],kl_vals[index] ) )
		index = index + 1
	return rs

# compute kl between lists
def compute_kls (ps, qs):
    rs = []
    index = 0
    while len ( rs ) < len ( ps ):
        if (qs[index]<1):
            rs.append ( kl ( ps[index],qs[index] ) )
        else:
            rs.append ( 0 )
        index = index + 1
    return rs

def maxlist (l):
    m = l[0]
    for x in l:
        m = max (m,x)
    return m

def minlist (l):
    m = l[0]
    for x in l:
        m = min (m,x)
    return m

# compute integral
def integral ( betas, train, factor ):
	u, s, index = [0], 0, 0
	while ( len (u) < len ( betas ) ):
		s = s + factor*(betas[index+1]-betas[index])*train[index]
		u.append(s)
		index = index + 1
	return u
 

def gammas ( betas, train, factor ):
	int = integral ( betas, train, factor ) 
	index = 0
	b = []
	while ( len(b) < len(train) ):
		b.append (int[index] - factor*betas[index]*train[index]) 
		index = index + 1
	return b

def klbounds ( betas, train, samplesize, delta, factor ):
	g = gammas (betas, train, factor )
	index = 0
	b = []
	while ( len(b) < len(train) ):
		b.append ((g[index] + ln(2*math.sqrt(samplesize)/delta))/samplesize)
		index = index + 1
	return b

def predictbce (betas, av_bcetrain, samplesize, factor):
    bounds = klbounds ( betas, av_bcetrain, samplesize, 0.01, factor)
    index = 0
    ps = [] 
    while index < len ( bounds ): 
        ps.append ( invert_kl (av_bcetrain[index],bounds[index]) ) 
        index = index + 1 
    return ps

def predict01 (betas, av_bcetrain, av_train01, samplesize, factor):
    bounds = klbounds ( betas, av_bcetrain, samplesize, 0.01, factor )
    index = 0
    ps = [] 
    while index < len ( bounds ): 
        ps.append ( invert_kl (av_train01[index],bounds[index]) ) 
        index = index + 1
        ps[0]=0.5
    return ps

def predict01hoeffding (betas, av_bcetrain, av_train01, samplesize, factor):
    g = gammas (betas, av_bcetrain, factor )
    index = 0
    ps = [] 
    while index < len ( g ): 
        ps.append ( av_train01[index] + math.sqrt((g[index]*( 1 + 1/samplesize) 
                                     + ln(max(g[index],1)*(samplesize+1)/0.01))/(2*samplesize) )) 
        index = index + 1 
    return ps

def calibrate (betas, av_bcetrain, av_train01, samplesize, thresh=0.5):
    l,r = 0.1,100
    factor = (l + r) / 2
    pred = predict01 (betas, av_bcetrain, av_train01, samplesize, factor)
    while (r-l > 0.01) or (minlist(pred) < 0.5):
        if minlist (pred[1:])  < thresh:
            l = factor
        if minlist (pred[1:]) >= thresh:
            r = factor
        factor = (l + r) / 2
        pred = predict01 (betas, av_bcetrain, av_train01, samplesize, factor)
    return factor


# CONTROL ------------------------------------------------

display = 1       # 0 = BBCE, 1 = 01
trueLabels = 1     # 0 = random, 1 = true labels
boundtype = 0      # 0 = kl 1 = Hoeffding 2 = Bernstein
showkls = 0        # 0 = don't show, 1 = show

# GET DATA
# naming conventions: 
# ( Dataset ) = M for MNIST, C for CIFAR
# ( C or R ) = correct or random labels
# ( L# ) = number of hidden layers.
# ( W# ) = width of hidden layers
# ( LMC method )  ULA, SGLD
# ( #k  ) = Samplesize in 1000's
# ( LR# ) learning rate where 001 = 0.01 etc
# ( loss fctn ) BBCE, Savage

truefilename_1, randomfilename_1 = "CCL2W1500SGLD8kLR0005BBCE.csv", "CRL2W1500SGLD8kLR0005BBCE.csv"

# for calibration load random data first
betas_1, bcetrain_1, bcetest_1, train01_1, test01_1, av_bcetrain_1, av_bcetest_1,\
        av_train01_1, av_test01_1, samplesize_1 = main( randomfilename_1 )
samplesize_1 = samplesize_1[0]
title = 'random labels '
print  (samplesize_1)      

if boundtype == 0: 
    bt = ' kl'
if boundtype == 1:
    bt = ' Hoeffding'

print (betas_1)
factor = calibrate (betas_1, av_bcetrain_1, av_train01_1, samplesize_1)


if trueLabels == 1:   # then reload 
    betas_1, bcetrain_1, bcetest_1, train01_1, test01_1, av_bcetrain_1, av_bcetest_1,\
        av_train01_1, av_test01_1, samplesize_1 = main( truefilename_1 ) 
    title = 'true labels '
    samplesize_1 = samplesize_1[0]
    #av_bcetrain[0]=(0.5+av_bcetrain[1])/2

print ('calibration factor =',factor)
pred01_1 = predict01 (betas_1, av_bcetrain_1, av_train01_1, samplesize_1, factor)

truefilename_2, randomfilename_2 = "CCL3W1000SGLD8kLR0005BBCE.csv", "CRL3W1000SGLD8kLR0005BBCE.csv"

# for calibration load random data first
betas_2, bcetrain_2, bcetest_2, train01_2, test01_2, av_bcetrain_2, av_bcetest_2,\
        av_train01_2, av_test01_2, samplesize_2 = main( randomfilename_2 )
samplesize_2 = samplesize_2[0]
title = 'random labels '
print  (samplesize_2)      

if boundtype == 0: 
    bt = ' kl'
if boundtype == 1:
    bt = ' Hoeffding'

print (betas_2)

factor = calibrate (betas_2, av_bcetrain_2, av_train01_2, samplesize_2)


if trueLabels == 1:   # then reload 
    betas_2, bcetrain_2, bcetest_2, train01_2, test01_2, av_bcetrain_2, av_bcetest_2,\
        av_train01_2, av_test01_2, samplesize_2 = main( truefilename_2 ) 
    title = 'true labels '
    samplesize_2 = samplesize_2[0]
    #av_bcetrain[0]=(0.5+av_bcetrain[1])/2

print ('calibration factor =',factor)
pred01_2 = predict01 (betas_2, av_bcetrain_2, av_train01_2, samplesize_2, factor)

truefilename_3, randomfilename_3 = "CCLVW500SGLD8kLR0005BBCE.csv", "CRLVW500SGLD8kLR0005BBCE.csv"

# for calibration load random data first
betas_3, bcetrain_3, bcetest_3, train01_3, test01_3, av_bcetrain_3, av_bcetest_3,\
        av_train01_3, av_test01_3, samplesize_3 = main( randomfilename_3 )
samplesize_3 = samplesize_3[0]
title = 'random labels '
print  (samplesize_3)      

if boundtype == 0: 
    bt = ' kl'
if boundtype == 1:
    bt = ' Hoeffding'

print (betas_3)

factor = calibrate (betas_3, av_bcetrain_3, av_train01_3, samplesize_3)


if trueLabels == 1:   # then reload 
    betas_3, bcetrain_3, bcetest_3, train01_3, test01_3, av_bcetrain_3, av_bcetest_3,\
        av_train01_3, av_test01_3, samplesize_3 = main( truefilename_3 ) 
    title = 'true labels '
    samplesize_3 = samplesize_3[0]
    #av_bcetrain[0]=(0.5+av_bcetrain[1])/2

print ('calibration factor =',factor)
pred01_3 = predict01 (betas_3, av_bcetrain_3, av_train01_3, samplesize_3, factor)

test_matrix = np.zeros((3,len(av_test01_1)))
bound_matrix = np.zeros((3,len(av_test01_1)))
for i in range(3):
    for j in range(len(av_test01_1)):
        if i == 0:
            test_matrix[i][j] = av_test01_1[j]
            bound_matrix[i][j] = pred01_1[j]
        if i == 1:
            test_matrix[i][j] = av_test01_2[j]
            bound_matrix[i][j] = pred01_2[j]
        if i == 2:
            test_matrix[i][j] = av_test01_3[j]
            bound_matrix[i][j] = pred01_3[j]

print('Test and Bound matrices:')
print(test_matrix)
print('-----')
print(bound_matrix)

# Create formatted table
print('\n' + '='*80)
print('TABLE: Test Error and Generalization Bounds by Architecture and Beta')
print('='*80)

# First table: Test Errors
print('\nTEST ERRORS (0-1 Loss):')
print('-'*60)
header_test = f"{'Beta':>8} | {'Architecture 1':>15} | {'Architecture 2':>15} | {'Architecture 3':>15}"
print(header_test)
print('-'*len(header_test))

for j in range(len(betas_1)):
    beta_val = betas_1[j]
    test1, test2, test3 = test_matrix[0][j], test_matrix[1][j], test_matrix[2][j]
    row = f"{beta_val:>8.0f} | {test1:>15.4f} | {test2:>15.4f} | {test3:>15.4f}"
    print(row)

# Second table: Generalization Bounds
print('\n\nGENERALIZATION BOUNDS:')
print('-'*60)
header_bound = f"{'Beta':>8} | {'Architecture 1':>15} | {'Architecture 2':>15} | {'Architecture 3':>15}"
print(header_bound)
print('-'*len(header_bound))

for j in range(len(betas_1)):
    beta_val = betas_1[j]
    bound1, bound2, bound3 = bound_matrix[0][j], bound_matrix[1][j], bound_matrix[2][j]
    row = f"{beta_val:>8.0f} | {bound1:>15.4f} | {bound2:>15.4f} | {bound3:>15.4f}"
    print(row)

# Combined table: Test / Bound pairs
print('\n\nCOMBINED TABLE (Test Error / Bound):')
print('-'*85)
header_combined = f"{'Beta':>8} | {'Architecture 1':>25} | {'Architecture 2':>25} | {'Architecture 3':>25}"
print(header_combined)
print('-'*len(header_combined))

for j in range(len(betas_1)):
    beta_val = betas_1[j]
    test1, bound1 = test_matrix[0][j], bound_matrix[0][j]
    test2, bound2 = test_matrix[1][j], bound_matrix[1][j]
    test3, bound3 = test_matrix[2][j], bound_matrix[2][j]
    
    row = f"{beta_val:>8.0f} | {test1:>8.4f} / {bound1:>8.4f} | {test2:>8.4f} / {bound2:>8.4f} | {test3:>8.4f} / {bound3:>8.4f}"
    print(row)

print('='*80)
print('\nLegend:')
print('Architecture 1: CCL2W1500SGLD8kLR0005BBCE (2 hidden layers, 1500 width)')
print('Architecture 2: CCL3W1000SGLD8kLR0005BBCE (3 hidden layers, 1000 width)')
print('Architecture 3: CCLVW500SGLD8kLR0005BBCE (VGG-16)')
print('Test: Actual test error (0-1 loss)')
print('Bound: Generalization bound prediction')
print('='*80)

# Generate LaTeX table
print('\n\nLaTeX Table:')
print('='*60)

# Find indices for beta = 1000 and beta = 64000
beta_1000_idx = None
beta_64000_idx = None
for i, beta in enumerate(betas_1):
    if beta == 1000.0:
        beta_1000_idx = i
    elif beta == 64000.0:
        beta_64000_idx = i

if beta_1000_idx is not None and beta_64000_idx is not None:
    # Extract the required values
    bound_1000_arch1 = bound_matrix[0][beta_1000_idx]
    bound_1000_arch2 = bound_matrix[1][beta_1000_idx]
    bound_1000_arch3 = bound_matrix[2][beta_1000_idx]
    
    test_64000_arch1 = test_matrix[0][beta_64000_idx]
    test_64000_arch2 = test_matrix[1][beta_64000_idx]
    test_64000_arch3 = test_matrix[2][beta_64000_idx]
    
    bound_64000_arch1 = bound_matrix[0][beta_64000_idx]
    bound_64000_arch2 = bound_matrix[1][beta_64000_idx]
    bound_64000_arch3 = bound_matrix[2][beta_64000_idx]
    
    # Generate LaTeX table
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|c|c|}}
\\hline
& Architecture 1 & Architecture 2 & Architecture 3 \\\\
\\hline
Bound at $\\beta = 1000$ & {bound_1000_arch1:.4f} & {bound_1000_arch2:.4f} & {bound_1000_arch3:.4f} \\\\
\\hline
Test Error at $\\beta = 64000$ & {test_64000_arch1:.4f} & {test_64000_arch2:.4f} & {test_64000_arch3:.4f} \\\\
\\hline
Bound at $\\beta = 64000$ & {bound_64000_arch1:.4f} & {bound_64000_arch2:.4f} & {bound_64000_arch3:.4f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Generalization bounds and test errors for different neural network architectures on CIFAR-10 dataset.}}
\\label{{tab:cifar_results}}
\\end{{table}}
"""
    
    print(latex_table)
    
    # Save LaTeX table to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    latex_file_path = os.path.join(script_dir, "cifar_results_table.tex")
    
    with open(latex_file_path, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to: {latex_file_path}")
    
    # Also print values for verification
    print(f"\nExtracted values:")
    print(f"Bound at beta=1000: Arch1={bound_1000_arch1:.4f}, Arch2={bound_1000_arch2:.4f}, Arch3={bound_1000_arch3:.4f}")
    print(f"Test at beta=64000: Arch1={test_64000_arch1:.4f}, Arch2={test_64000_arch2:.4f}, Arch3={test_64000_arch3:.4f}")
    print(f"Bound at beta=64000: Arch1={bound_64000_arch1:.4f}, Arch2={bound_64000_arch2:.4f}, Arch3={bound_64000_arch3:.4f}")
    
else:
    print("Error: Could not find beta values 1000 or 64000 in the data")

print('='*60)