# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 08:24:31 2025

@author: am
"""

import os
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# generic infinite impulse response filter
# returns current and last values

def iir (current, c_data, c_recurs, mem_data, mem_recurs):
# 0. remember last output
    last = mem_recurs[0]
# 1. compute filter output
    v = 0
    i = 0
    mem_data[0] = current
    while i < len(c_data):
        v = v + c_data[i]*mem_data[i]
        i = i+1
    i = 0
    while i < len(c_recurs):
        v = v + c_recurs[i]*mem_recurs[i]
        i = i+1
# 2. reshuffle memory data
    i = len(mem_recurs) - 1
    while i > 0:
        mem_recurs[i] = mem_recurs[i-1] 
        i = i-1 
    mem_recurs[0] = v 
    i = len(mem_data) - 1
    while i > 0:
        mem_data[i] = mem_data[i-1] 
        i = i-1 
    return v, last

# translate cutoff frequence (in Nyquist units) to digital filter
def psi (cutoff):
    return ( math.cos(cutoff*3.14/2)/math.sin(cutoff*3.14/2))

# simple exponential moving average
def create_ema (cutoff, initialvalue):
    p = psi ( cutoff )
    mem_data, mem_recurs = [initialvalue],[initialvalue]
    c_data, c_recurs = [1/p],[1-1/p]
    return mem_data, mem_recurs, c_data, c_recurs

# EMA with nonrecursive 2nd order                
def create_ema2 (cutoff, initialvalue):
    p = psi ( cutoff )
    mem_data, mem_recurs = [initialvalue,initialvalue],[initialvalue]
    c_data, c_recurs = [1/(2*p),1/(2*p)],[1-1/p]
    return mem_data, mem_recurs, c_data, c_recurs
        
# 2md degree butterworth
def create_btw (cutoff, init):
    p = psi ( cutoff )
    mem_data, mem_recurs = [init,init,init],[init,init]
    denom = p**2 + math.sqrt(2)*p + 1
    c_data = [1/denom, 2/denom, 1/denom]
    c_recurs = [-2*(1-p**2)/denom,-(p**2 - math.sqrt(2)*p + 1)/denom]
    return mem_data, mem_recurs, c_data, c_recurs

start = 100 # start value for simulation

global_cutoff = 0.001 # cutoff frequency in simulation

# create filters
mem_dema, mem_rema, c_dema, c_rema = create_ema (global_cutoff,start)
mem_dem2, mem_rem2, c_dem2, c_rem2 = create_ema2(global_cutoff,start)
mem_dbtw, mem_rbtw, c_dbtw, c_rbtw = create_btw (global_cutoff,start)

def ema(current):
    return iir (v, c_dema, c_rema, mem_dema, mem_rema)
def em2(current):
    return iir (v, c_dem2, c_rem2, mem_dem2, mem_rem2)
def btw(current):
    return iir (v, c_dbtw, c_rbtw, mem_dbtw, mem_rbtw)

def noise(noisetype,noiselevel):
    if noisetype == 0:
        return random.gauss(0,noiselevel)
    if noisetype == 1:
        return random.choice([-noiselevel,-noiselevel,2*noiselevel])

noisetype = 0
noiselevel = 2

rate = 0.001
threshold = 0.0005

stopema = 0
stopem2 = 0
stopbtw = 0

t    = 0
vs   = []
emas = []
em2s = []
btws = []
ts   = []
while t < 20000:
    v = (start-10)*math.exp(-rate*t) + noise(noisetype,noiselevel) + 10 
    wbtw,lastbtw = btw (v )
    wema,lastema = ema (v )
    wem2,lastem2 = em2 (v )
    if t > 0.1/rate:
#        print (v,', ',w)
        if wbtw - lastbtw > threshold:
            stopbtw = 1
        if wema - lastema > threshold:
            stopema = 1
        if wem2 - lastem2 > threshold:
            stopem2 = 1
        if t % 100 == 0:
            vs.append(v)
            if stopbtw == 0:
                btws.append(wbtw)
            else:
                btws.append(0)
            if stopema == 0:
                emas.append(wema)
            else:
                emas.append(0)
            if stopem2 == 0:
                em2s.append(wem2)
            else:
                em2s.append(0)
            ts.append(t)
    t = t+1
print('t = ',t)

fig, ax = plt.subplots()             # Create a 
ax.plot(ts, vs, color = '#909090', linewidth=1.0, label = 'signal')                 # Plot some data on the Axes.
ax.plot(ts, btws, color = 'g', linewidth=2.0, label = 'btw')                 # Plot some data on the Axes.
ax.plot(ts, em2s, color = 'y', linewidth=2.0, label = 'em2')                 # Plot some data on the Axes.
ax.plot(ts, emas, color = 'r', linewidth=2.0, label = 'ema')                 # Plot some data on the Axes.
ax.legend()
plt.show()                           # Show the figure.

    
print (exponent)

