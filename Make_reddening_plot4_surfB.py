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
import sklearn.datasets as ds
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
def plot_array(inFile, scatter=False, binned=True):
    
    
    R, Input_u2, T_u2  = getBand(inFile, band1 = 'u', band2 = 'w2')
    R, Input_g2, T_g2  = getBand(inFile, band1 = 'g', band2 = 'w2')
    R, Input_r2, T_r2  = getBand(inFile, band1 = 'r', band2 = 'w2')
    R, Input_i2, T_i2  = getBand(inFile, band1 = 'i', band2 = 'w2')
    R, Input_z2, T_z2  = getBand(inFile, band1 = 'z', band2 = 'w2')
    R, Input_w1, T_w1  = getBand(inFile, band1 = 'w1', band2 = 'w2')
    Input2 = {} ; T2 = {}
    T2["u"]  = T_u2
    T2["g"]  = T_g2
    T2["r"]  = T_r2
    T2["i"]  = T_i2
    T2["z"]  = T_z2
    T2["w1"]  = T_w1
    Input2["u"] = Input_u2
    Input2["g"] = Input_g2
    Input2["r"] = Input_r2
    Input2["i"] = Input_i2
    Input2["z"] = Input_z2    
    Input2["w1"] = Input_w1    
    
    dye = {"u":"blue","g":"green","r":"red","i":"orange","z":"maroon","w1":"purple" }
    
    fig = py.figure(figsize=(16, 3.5), dpi=100)    
    fig.subplots_adjust(wspace=0, top=0.95, bottom=0.15, left=0.06, right=0.99)
    
    gs = gridspec.GridSpec(1, 6) 

    p = 0
    ####################################################
    
    band_lst = ['u', 'g','r','i','z', 'w1']
    
    if True:

        for band in band_lst:
            
            xlabel = False; ylabel=False
            if band=='u': ylabel=True
            if band=='r': xlabel=True
            
            ax = plt.subplot(gs[p]) ; p+=1
            plot_Rinc(ax, T2[band], Input2[band], color=dye[band], scatter=scatter, binned=binned, xlabel=xlabel, ylabel=ylabel, band=band)
            yticks = ax.yaxis.get_major_ticks()
            if band!='u': yticks[-1].label1.set_visible(False)
            if band!='u': plt.setp(ax.get_yticklabels(), visible=False)  
                 
    #####################################################
    
    plt.subplots_adjust(hspace=.0, wspace=0)

    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')    
    ax.annotate(r'$\langle \mu_2 \rangle^{(i)}_e$', (0.50,0.02), xycoords='figure fraction', size=18, color='black')
    
    fig.savefig("A_w12_inc_surfB.eps")
    fig.savefig("A_w12_inc_surfB.png")
    plt.show()
    
################################################################## 
def plot_Rinc(ax, T2, Input2, color='red', scatter=False, binned=False, xlabel=True, ylabel=True, X_twin=True, Y_twin=True, band='r'):
    
    myDic = {}

    pc0     = Input2[2]
    inc     = Input2[3]
    table = T2[5]
    mu50  = table['mu50']
    r_w1  = table['r_w1']
    a,b,c,d, alpha, beta, gamma, Ealpha, Ebeta = getReddening_params(band1=band, band2='w2')
    q2 = 10**(-1.*gamma)
    F = log_a_b(inc, q2)


    R = r_w1 - (alpha*pc0+beta)
    #if band!='w1': 
        #R[np.where(R<0)]=0
    A2 = R


    
    a0,b0,c0,d0= -5.44067348e-03, 3.58846252e-01, -7.87817385e+00, 5.83625168e+01
    reddening = {"u":1.45,"g":1.0,"r":0.77,"i":0.63,"z":0.52,"w1":0.06}
    A1 = F*(a0*mu50**3+b0*mu50**2+c0*mu50+d0)*reddening[band]
    
    #A2 = F*(a*pc0**3+b*pc0**2+c*pc0+d)

    

    INC = mu50
    dA = A2-A1
    
    if scatter:
        ax.plot(INC, dA, 'o', color='black', markersize=1, alpha=0.15)
        rms = np.sqrt(np.mean(dA**2))
        ax.text(25,-0.4, r'$RMS=$'+'%.2f'%rms+' mag', fontsize=12, color='k')
    
    
    ax.text(25.3,0.4, band, fontsize=16, color=color) 
    
    if binned:
        xl = []
        yl= []
        yel=[]
        
        low = 20; high=26
        for i in np.arange(low,high,0.5):
            
            x = []
            y = []
            for ii in range(len(dA)):
                xi = INC[ii]
                if xi>i and xi<=i+0.5:
                    x.append(xi)
                    y.append(dA[ii])
            if len(x)>0:
                
                x = np.asarray(x)
                y = np.asarray(y)
                
                average   = np.median(y)
                stdev = np.std(y)
                
                index = np.where(y<average+3.*stdev)
                x = x[index]
                y = y[index]
                
                index = np.where(y>average-3.*stdev)
                x = x[index]
                y = y[index]        

                if len(x)>=7 and np.median(x)>20.2 and np.median(x)<25.5:
                    if np.median(x)>21:
                        ax.errorbar(np.median(x), np.median(y), yerr=np.std(y), fmt='o', color=color, markersize=5)
                    else:
                        ax.errorbar(np.median(x), np.median(y), yerr=np.std(y), fmt='o', color=color, markersize=5, markerfacecolor='white')
                
                xl.append(np.median(x))
                yl.append(np.median(y))
                yel.append(np.std(y))
            
            
    ax.tick_params(which='major', length=6, width=1.5, direction='in')
    ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    ax.minorticks_on()

    ax.plot([26,20], [0,0], 'k:')
    

    if ylabel: ax.set_ylabel(r'$A^{(i)}_{W2}-A^{(i)}_{\mu_2}$', fontsize=16) 

    ylim = [-0.5,0.5]

    
    ax.set_ylim(ylim)     
    ax.set_xlim([25.8,20.2]) 
    
    if Y_twin:
        y_ax = ax.twinx()
        y_ax.set_ylim(ylim[0],ylim[1])
        y_ax.set_yticklabels([])
        y_ax.minorticks_on()
        y_ax.tick_params(which='major', length=6, width=1.5, direction='in')
        y_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')        
    
    if X_twin:
        x_ax = ax.twiny()
        x_ax.set_xlim(25.8,20.2)
        x_ax.set_xticklabels([])
        x_ax.minorticks_on()
        x_ax.tick_params(which='major', length=6, width=1.5, direction='in')
        x_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')     

    
    #if len(dR)>0:
        
        #x0=-2.5; y0=-0.3
        #plt.errorbar([x0], [y0], xerr=[np.median(pc0)], yerr=[np.median(dR)], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)
        

    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14) 

###########################################################



plot_array('ESN_HI_catal.csv', scatter=True, binned=True)     


