# -*- coding: utf-8 -*-

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
		s = s + factor*(betas[index+1]-betas[index])*av_bcetrain[index]
		u.append(s)
		index = index + 1
	return u
 

def gammas ( betas, train, factor ):
	int = integral ( betas, train, factor ) 
	index = 0
	b = []
	while ( len(b) < len(train) ):
		b.append (int[index] - factor*betas[index]*av_bcetrain[index]) 
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

def calibrate (etas, av_bcetrain, av_train01, samplesize):
    l,r = 0.1,100
    factor = (l + r) / 2
    pred = predict01 (betas, av_bcetrain, av_train01, samplesize, factor)
    pred[0] = pred[1]
    while (r-l > 0.01) or (minlist(pred) < 0.5):
        if minlist (pred)  < 0.5:
            l = factor
        if minlist (pred) >= 0.5:
            r = factor
        factor = (l + r) / 2
        pred = predict01 (betas, av_bcetrain, av_train01, samplesize, factor)
        pred[0] = pred[1]
    return factor


# CONTROL ------------------------------------------------

display = 0        # 0 = BBCE, 1 = 01
trueLabels = 0     # 0 = random, 1 = true labels
boundtype = 0      # 0 = kl 1 = Hoeffding 2 = Bernstein
showkls = 0        # 0 = don't show, 1 = show

# GET DATA
# naming conventions: 
# ( c or r ) = correct or random labels
# ( Dataset ) = M for MNIST, C for CIFAR
# ( L# ) = number of hidden layers. if 1 then W=500, if 2 then W=1000
# ( LMC method )  U = ULA, S = SGLD
# ( #k  ) = Samplesize in 1000's
# ( # ) learning rate where 001 = 0.01 etc
# ( loss fctn )  B = BBCE, S = Savage

filename = 'ML1S2k001B'
truefilename, randomfilename = 'c'+filename+'.csv', 'r'+filename+'.csv'

truefilename, randomfilename = "MCL1W500ULA2kLR001SAVAGE.csv", "MRL1W500ULA2kLR001SAVAGE.csv"

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

# av_bcetrain[0]=0.5
#av_bcetrain[0]=av_bcetrain[1]
av_bcetest[0] = av_bcetest[1]
av_train01[0] = 0.5
av_test01[0] = 0.5

factor = calibrate (betas, av_bcetrain, av_train01, samplesize)


if trueLabels == 1:   # then reload 
    betas, bcetrain, bcetest, train01, test01, av_bcetrain, av_bcetest,\
        av_train01, av_test01, samplesize = main( truefilename ) 
    title = 'true labels '
    samplesize = samplesize[0]
    #av_bcetrain[0]=(0.5+av_bcetrain[1])/2

print ('calibration factor =',factor)

bounds = klbounds ( betas, av_bcetrain, samplesize, 0.01, factor)
testkl = compute_kls (av_bcetrain,av_bcetest)
testkl01 = compute_kls (av_train01,av_test01)
predbce = predictbce (betas, av_bcetrain, samplesize, factor)
if boundtype == 1: 
    pred01 = predict01hoeffding (betas, av_bcetrain, av_train01, samplesize, factor)
if boundtype == 0:
    pred01 = predict01 (betas, av_bcetrain, av_train01, samplesize, factor)
av_bcetest[0] = av_bcetest[1]
#print (pred01)

def showbce (showkls):
    fig, ax = plt.subplots()             # Create a figure containing a single Axes.
    ax.semilogx()
    ax.plot(betas, av_bcetrain, 'o-k', label = 'train')                 # Plot some data on the Axes.
    ax.plot(betas, predbce   , '+-r', label = 'test bound')                 # Plot some data on the Axes.
    ax.plot(betas, av_bcetest, '*-g', label = 'test')                 # Plot some data on the Axes.
    if showkls == 1:
      ax.plot(betas, testkl , '*-g', label = 'kl(train,test)')                 # Plot some data on the Axes.
      ax.plot(betas, bounds , 'x-y', label = 'kl-bound')                 # Plot some data on the Axes.
    # ax.set_title (filename + ' '+title +'BCE-error')
    ax.legend()
    
    os.makedirs('newplots', exist_ok=True)
    if trueLabels == 1:
        csv_filename = truefilename[:-4]
        if display == 1:
            csv_filename = csv_filename + '_01'
        else:
            csv_filename = csv_filename + '_loss'
        csv_filename = 'newplots/' + csv_filename + '.png'
    else:
        csv_filename = randomfilename[:-4]
        if display == 1:
            csv_filename = csv_filename + '_01'
        else:
            csv_filename = csv_filename + '_loss'
        csv_filename = 'newplots/' + csv_filename + '.png'

    # Save the figure
    plt.savefig(csv_filename, dpi=300, bbox_inches='tight')
    
    plt.show()                           # Show the figure.
    
def show01 (showkls):
    fig, ax = plt.subplots()             # Create a figure containing a single Axes.
    ax.semilogx()
    ax.plot(betas, av_train01, 'o-k', label = 'train')                 # Plot some data on the Axes.
    ax.plot(betas, pred01, '+-r', label = 'test bound')                 # Plot some data on the Axes.
    ax.plot(betas, av_test01, '*-b', label = 'test')                 # Plot some data on the Axes.
    if showkls == 1:
      ax.plot(betas, testkl01 , '.-g', label = 'kl(train,test)')                 # Plot some data on the Axes.
      ax.plot(betas, bounds , 'x-y', label = 'kl-bound')                 # Plot some data on the Axes.
    # ax.set_title (filename + ' '+title + '01-error'+bt)
    ax.legend()
    os.makedirs('newplots', exist_ok=True)
    if trueLabels == 1:
        csv_filename = truefilename[:-4]
        if display == 1:
            csv_filename = csv_filename + '_01'
        else:
            csv_filename = csv_filename + '_loss'
        csv_filename = 'newplots/' + csv_filename + '.png'
    else:
        csv_filename = randomfilename[:-4]
        if display == 1:
            csv_filename = csv_filename + '_01'
        else:
            csv_filename = csv_filename + '_loss'
        csv_filename = 'newplots/' + csv_filename + '.png'

    # Save the figure
    plt.savefig(csv_filename, dpi=300, bbox_inches='tight')
    plt.show()                           # Show the figure.

if display == 1:
    show01 (showkls) 
else:
    showbce (showkls)
