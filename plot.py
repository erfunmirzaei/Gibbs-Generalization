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
calibration = 1    # 0 = no calibration 1 = do it
singledraw = 1     # 0 = posterior average, 1 = single draw
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

truefilename, randomfilename = "CCL2W1500SGLD8kLR0005BBCE.csv", "CRL2W1500SGLD8kLR0005BBCE.csv"

# for calibration load random data first
betas, bcetrain, bcetest, train01, test01, av_bcetrain, av_bcetest,\
        av_train01, av_test01, samplesize = main( randomfilename )
samplesize = samplesize[0]
title = 'random labels '
print  (samplesize)      

if boundtype == 0: 
    bt = ' kl'
if boundtype == 1:
    bt = ' Hoeffding'

print (betas)

if calibration == 1:
    factor = calibrate (betas, av_bcetrain, av_train01, samplesize, thresh=0.50)
else:
    factor = 1

if trueLabels == 1:   # then reload 
    betas, bcetrain, bcetest, train01, test01, av_bcetrain, av_bcetest,\
        av_train01, av_test01, samplesize = main( truefilename ) 
    title = 'true labels '
    samplesize = samplesize[0]

print ('calibration factor =',factor)

bounds = klbounds ( betas, av_bcetrain, samplesize, 0.01, factor)
testkl = compute_kls (av_bcetrain,av_bcetest)
testkl01 = compute_kls (av_train01,av_test01)
predbce = predictbce (betas, av_bcetrain, samplesize, factor)
if boundtype == 1: 
    pred01 = predict01hoeffding (betas, av_bcetrain, av_train01, samplesize, factor)
if boundtype == 0:
    if singledraw == 1:
        pred01 = predict01 (betas, av_bcetrain, train01, samplesize, factor)
    else:
        pred01 = predict01 (betas, av_bcetrain, av_train01, samplesize, factor)
        
def showbce (showkls):
    plt.rcParams.update({
        'font.size': 14,           # Base font size
        'axes.labelsize': 16,      # x and y labels
        'axes.titlesize': 18,      # Title size
        'xtick.labelsize': 14,     # x tick labels
        'ytick.labelsize': 14,     # y tick labels
        'legend.fontsize': 14,     # Legend font size
        'lines.linewidth': 2.5,    # Line width
        'lines.markersize': 8,     # Marker size
        'figure.figsize': (10, 7), # Figure size for better aspect ratio
        'axes.grid': False,         # Enable grid
        'grid.alpha': 0.3,         # Grid transparency
        'axes.axisbelow': True     # Put grid behind data
    })
    
    fig, ax = plt.subplots()
    ax.semilogx()
    
    # Enhanced plotting with original colors and enhanced markers
    if singledraw == 1:
        ax.plot(betas[1:], bcetrain[1:], 'o-k', linewidth=2.5, 
                markersize=8, label='Train Error', markerfacecolor='white', markeredgewidth=2)
        ax.plot(betas[1:], predbce[1:], 's-r', linewidth=2.5, 
                markersize=7, label='Test Bound', markerfacecolor='white', markeredgewidth=2)
        ax.plot(betas[1:], bcetest[1:], '^-b', linewidth=2.5, 
                markersize=8, label='Test Error', markerfacecolor='white', markeredgewidth=2)
    else:
        ax.plot(betas[1:], av_bcetrain[1:], 'o-k', linewidth=2.5, 
                markersize=8, label='Train Error', markerfacecolor='white', markeredgewidth=2)
        ax.plot(betas[1:], predbce[1:], 's-r', linewidth=2.5, 
                markersize=7, label='Test Bound', markerfacecolor='white', markeredgewidth=2)
        ax.plot(betas[1:], av_bcetest[1:], '^-b', linewidth=2.5, 
                markersize=8, label='Test Error', markerfacecolor='white', markeredgewidth=2)

    if showkls == 1:
        ax.plot(betas[1:], testkl[1:], 'v-g', linewidth=2, 
                markersize=6, label='KL(Train,Test)', alpha=0.8)
        ax.plot(betas[1:], bounds[1:], 'D-y', linewidth=2, 
                markersize=6, label='KL-Bound', alpha=0.8)
    
    # Enhanced formatting
    ax.set_xlabel('Beta', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    ax.set_ylim(0, 0.6)
    
    # Better legend
    ax.legend(frameon=True, fancybox=False, shadow=False, loc='best', 
              framealpha=0.9, edgecolor='black')
    
    # Add minor ticks for better readability
    ax.minorticks_on()
    ax.tick_params(which='minor', length=3, color='gray')
    ax.tick_params(which='major', length=6, width=1.2)
    
    # Ensure directory exists
    os.makedirs('newplots', exist_ok=True)
    
    # Generate filename
    if trueLabels == 1:
        csv_filename = truefilename[:-4]
        if display == 1:
            csv_filename = csv_filename + '_01'
        else:
            csv_filename = csv_filename + '_loss'
        if singledraw == 1:
            csv_filename = csv_filename + '_singledraw'
        csv_filename = 'newplots/' + csv_filename + '.png'
    else:
        csv_filename = randomfilename[:-4]
        if display == 1:
            csv_filename = csv_filename + '_01'
        else:
            csv_filename = csv_filename + '_loss'
        if singledraw == 1:
            csv_filename = csv_filename + '_singledraw'
        csv_filename = 'newplots/' + csv_filename + '.png'

    # Save with high quality
    plt.savefig(csv_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plt.show()
    
def show01 (showkls):
    plt.rcParams.update({
        'font.size': 14,           # Base font size
        'axes.labelsize': 16,      # x and y labels
        'axes.titlesize': 18,      # Title size
        'xtick.labelsize': 14,     # x tick labels
        'ytick.labelsize': 14,     # y tick labels
        'legend.fontsize': 14,     # Legend font size
        'lines.linewidth': 2.5,    # Line width
        'lines.markersize': 8,     # Marker size
        'figure.figsize': (10, 7), # Figure size for better aspect ratio
        'axes.grid': False,         # Enable grid
        'grid.alpha': 0.3,         # Grid transparency
        'axes.axisbelow': True     # Put grid behind data
    })
    
    fig, ax = plt.subplots()
    ax.semilogx()
    
    # Enhanced plotting with original colors and enhanced markers
    if singledraw == 1:
        ax.plot(betas[1:], train01[1:], 'o-k', linewidth=2.5, 
                markersize=8, label='Train Error', markerfacecolor='white', markeredgewidth=2)
        ax.plot(betas[1:], pred01[1:], 's-r', linewidth=2.5, 
                markersize=7, label='Test Bound', markerfacecolor='white', markeredgewidth=2)
        ax.plot(betas[1:], test01[1:], '^-b', linewidth=2.5, 
                markersize=8, label='Test Error', markerfacecolor='white', markeredgewidth=2)
    else:
        ax.plot(betas[1:], av_train01[1:], 'o-k', linewidth=2.5, 
                markersize=8, label='Train Error', markerfacecolor='white', markeredgewidth=2)
        ax.plot(betas[1:], pred01[1:], 's-r', linewidth=2.5, 
                markersize=7, label='Test Bound', markerfacecolor='white', markeredgewidth=2)
        ax.plot(betas[1:], av_test01[1:], '^-b', linewidth=2.5, 
                markersize=8, label='Test Error', markerfacecolor='white', markeredgewidth=2)

    if showkls == 1:
        ax.plot(betas[1:], testkl01[1:], 'v-g', linewidth=2, 
                markersize=6, label='KL(Train,Test)', alpha=0.8)
        ax.plot(betas[1:], bounds[1:], 'D-y', linewidth=2, 
                markersize=6, label='KL-Bound', alpha=0.8)
        ax.plot(betas[1:], bounds[1:], 'x-y', linewidth=2, 
                markersize=6, label='kl-bound', alpha=0.8)

    
    # Enhanced formatting
    ax.set_xlabel('Beta', fontsize=18)
    ax.set_ylabel('0-1 Error', fontsize=18)
    ax.set_ylim([0, 0.63])
    
    # Better legend
    ax.legend(frameon=True, fancybox=False, shadow=False, loc='best', 
              framealpha=0.9, edgecolor='black')
    
    # Add minor ticks for better readability
    ax.minorticks_on()
    ax.tick_params(which='minor', length=3, color='gray')
    ax.tick_params(which='major', length=6, width=1.2)
    
    # Ensure directory exists
    os.makedirs('newplots', exist_ok=True)
    
    # Generate filename
    if trueLabels == 1:
        csv_filename = truefilename[:-4]
        if display == 1:
            csv_filename = csv_filename + '_01'
        else:
            csv_filename = csv_filename + '_loss'
        if singledraw == 1:
            csv_filename = csv_filename + '_singledraw'
        csv_filename = 'newplots/' + csv_filename + '.png'
    else:
        csv_filename = randomfilename[:-4]
        if display == 1:
            csv_filename = csv_filename + '_01'
        else:
            csv_filename = csv_filename + '_loss'
        if singledraw == 1:
            csv_filename = csv_filename + '_singledraw'
        csv_filename = 'newplots/' + csv_filename + '.png'

    # Save with high quality
    plt.savefig(csv_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plt.show()

if display == 1:
    show01 (showkls) 
else:
    showbce (showkls)