#!/usr/bin/python
# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import subprocess
import math
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, Column 
from scipy.stats import linregress
from scipy import interpolate
from scipy import polyval, polyfit
from scipy import odr
import pylab as py
from matplotlib import gridspec

from redTools import *
from Kcorrect import *
################################################################# 
def add_axis(ax, xlim, ylim):
    
    x1, x2 = xlim[0], xlim[1]
    y1, y2 = ylim[0], ylim[1]
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)

    ax.minorticks_on()
    ax.tick_params(which='major', length=7, width=1.5)
    ax.tick_params(which='minor', length=4, color='#000033', width=1.0)     
    
    # additional Y-axis (on the right)
    y_ax = ax.twinx()
    y_ax.set_ylim(y1, y2)
    y_ax.set_yticklabels([])
    y_ax.minorticks_on()
    y_ax.tick_params(which='major', length=7, width=1.5, direction='in')
    y_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')

    # additional X-axis (on the top)
    x_ax = ax.twiny()
    x_ax.set_xlim(x1, x2)
    x_ax.set_xticklabels([])
    x_ax.minorticks_on()
    x_ax.tick_params(which='major', length=7, width=1.5, direction='in')
    x_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    
    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(12)   
    

########################################################### Begin

def plot_Band(ax, band1='r', band2='w1'):

    inFile  = 'ESN_HI_catal.csv'
    table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

    table = extinctionCorrect(table)
    table = Kcorrection(table)

    text1 = '\overline{'+band1+'}-\overline{W}1'            # example: cr-W1
       
    if band2=='w1':
        text1 = r'\overline{'+band1+r'}-\overline{W}1'            # example: cr-W1
        text2 = 'C_{12W1}'              # example: c21w
    else: 
        text1 = '\overline{'+band1+'}-\overline{W}2'
        if band1=='w1': text1 = r'\overline{W}1-\overline{W}2'
        text2 = 'C_{12W2}'
        
    delta = np.abs(table[band2]-table[band2+'_'])
    index, = np.where(delta<=0.15)
    table = trim(table, index)

    delta = np.abs(table[band1]-table[band1+'_'])
    index, = np.where(delta<=0.15)
    table = trim(table, index)

    table['c21w'] = table['m21'] - table[band2]
    table['r_w1'] = table[band1] - table[band2]

    table['Ec21w'] = np.sqrt(table['m21_e']**2+0.05**2)
    
    if band1!='u':
        table['Er_w1'] = 0.*table['r_w1']+0.05*np.sqrt(2.)
    else: 
        table['Er_w1'] = 0.*table['r_w1']+np.sqrt(0.1**2+0.05**2)

    index, = np.where(table['logWimx']>1)
    table = trim(table, index)

    index, = np.where(table['r_w1']<4)
    table = trim(table, index)


    ########################################################Face-ON

    ## Get the initial estimations using Face-on galaxies
    ## AB:    a0*logWimx+b0
    ## Delta: alfa*X**2+beta*X+gama
    AB, Delta, table, cov, Delta_e = faceON(table)
    ########################################################### END


    pgc = table['pgc']
    logWimx = table['logWimx']
    logWimx_e = table['logWimx_e']
    inc = table['inc']
    r_w1 = table['r_w1']
    c21w = table['c21w'] 
    Er_w1 = table['Er_w1']
    Ec21w = table['Ec21w']

    a0, b0 = AB[0],AB[1]
    a = 1./a0
    b = -1.*b0/a0

    #########################################################################  

    delta = r_w1-(a0*logWimx+b0)
    
    for i in range(-1,6):
        
        x = []
        y = []
        for ii in range(len(c21w)):
            xi = c21w[ii]
            if xi>=i and xi<i+1:
                x.append(xi)
                y.append(delta[ii])
        if len(x)>1:
            ax.errorbar(np.median(x), np.median(y), yerr=np.std(y), xerr=np.std(x), fmt='o', alpha=0.7, color='red', markersize=4)    
    
    
    ax.plot(c21w, delta, '.', color='black', markersize=3, alpha=0.5)
    #ax.errorbar(c21w, delta, xerr=Ec21w, yerr=delta*0.+0.1, color='k', fmt='.', alpha=0.2)

    add_axis(ax,[-1.5,7.5],[-1.3,1.3])


    
    ax.set_ylabel(r'$'+'('+text1+')-('+text1+')_{fit}$', fontsize=16, labelpad=7)
    ax.set_xlabel('$'+text2+'$', fontsize=16, labelpad=7) 


    ax.plot([-1,8], [0,0], 'k:')

    X_err_median = np.median(Ec21w)
    Y_err_median = np.median(delta*0.+0.1)
    Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
    x0 = 0.9*Xlm[0]+0.1*Xlm[1]
    y0 = 0.2*Ylm[0]+0.8*Ylm[1]
    ax.errorbar([x0], [y0], yerr=[Y_err_median], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)

    xfit = np.linspace(-2,10, 100)
    yfit = Fdelta(Delta, xfit)
    ax.plot(xfit, yfit, '--', color='blue')   
    
    x0 = 0.60*Xlm[0]+0.40*Xlm[1]
    y0 = 0.80*Ylm[0]+0.20*Ylm[1]
    ax.text(x0,y0, r'$\delta_2=$'+'%.2f'%Delta[0]+'$\pm$'+'%.2f'%Delta_e[0], fontsize=14, color='k')

    y0 = 0.88*Ylm[0]+0.12*Ylm[1]
    ax.text(x0,y0, r'$\delta_1=$'+'%.2f'%Delta[1]+'$\pm$'+'%.2f'%Delta_e[1], fontsize=14, color='k')

    y0 = 0.96*Ylm[0]+0.04*Ylm[1]
    ax.text(x0,y0, r'$\delta_0=$'+'%.2f'%Delta[2]+'$\pm$'+'%.2f'%Delta_e[2], fontsize=14, color='k')
    #########################################################################  
################################################################# 
################################################################# 
################################################################# 
################################################################# 



fig = py.figure(figsize=(13, 8), dpi=100)   
fig.subplots_adjust(wspace=0.4, hspace = 0.25, top=0.97, bottom=0.09, left=0.08, right=0.98)
gs = gridspec.GridSpec(2, 3) 
p = 0

ax = plt.subplot(gs[p]) ; p+=1 
plot_Band(ax, band1='u', band2='w2')

ax = plt.subplot(gs[p]) ; p+=1  
plot_Band(ax, band1='g', band2='w2')

ax = plt.subplot(gs[p]) ; p+=1  
plot_Band(ax, band1='r', band2='w2')

ax = plt.subplot(gs[p]) ; p+=1 
plot_Band(ax, band1='i', band2='w2')

ax = plt.subplot(gs[p]) ; p+=1  
plot_Band(ax, band1='z', band2='w2')

ax = plt.subplot(gs[p]) ; p+=1  
plot_Band(ax, band1='w1', band2='w2')

plt.show()
