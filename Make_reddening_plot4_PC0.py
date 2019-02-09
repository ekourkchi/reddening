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
    
    R, Input_u1, T_u1  = getBand(inFile, band1 = 'u', band2 = 'w1')
    R, Input_g1, T_g1  = getBand(inFile, band1 = 'g', band2 = 'w1')
    R, Input_r1, T_r1  = getBand(inFile, band1 = 'r', band2 = 'w1')
    R, Input_i1, T_i1  = getBand(inFile, band1 = 'i', band2 = 'w1')
    R, Input_z1, T_z1  = getBand(inFile, band1 = 'z', band2 = 'w1')
    Input1 = {} ; T1 = {}
    T1["u"]  = T_u1
    T1["g"]  = T_g1
    T1["r"]  = T_r1
    T1["i"]  = T_i1
    T1["z"]  = T_z1
    Input1["u"] = Input_u1
    Input1["g"] = Input_g1
    Input1["r"] = Input_r1
    Input1["i"] = Input_i1
    Input1["z"] = Input_z1
    
    R, Input_u2, T_u2  = getBand(inFile, band1 = 'u', band2 = 'w2')
    R, Input_g2, T_g2  = getBand(inFile, band1 = 'g', band2 = 'w2')
    R, Input_r2, T_r2  = getBand(inFile, band1 = 'r', band2 = 'w2')
    R, Input_i2, T_i2  = getBand(inFile, band1 = 'i', band2 = 'w2')
    R, Input_z2, T_z2  = getBand(inFile, band1 = 'z', band2 = 'w2')
    Input2 = {} ; T2 = {}
    T2["u"]  = T_u2
    T2["g"]  = T_g2
    T2["r"]  = T_r2
    T2["i"]  = T_i2
    T2["z"]  = T_z2
    Input2["u"] = Input_u2
    Input2["g"] = Input_g2
    Input2["r"] = Input_r2
    Input2["i"] = Input_i2
    Input2["z"] = Input_z2    
    
    
    dye = {"u":"blue","g":"green","r":"red","i":"orange","z":"maroon","w1":"purple" }
    
    fig = py.figure(figsize=(15, 6), dpi=100)    
    fig.subplots_adjust(wspace=0, top=0.95, bottom=0.10, left=0.08, right=0.98)
    
    gs = gridspec.GridSpec(2, 5, height_ratios=[1,0.7]) 

    p = 0
    ####################################################
    
    band_lst = ['u', 'g','r','i','z']
    
    for jj in range(2):
        
        
        for band in band_lst:
            
            xlabel = False; ylabel=False
            if band=='u': ylabel=True
            if jj==1 and band=='r': xlabel=True
            
            ax = plt.subplot(gs[p]) ; p+=1
            plot_Rinc(ax, T1[band], Input1[band], T2[band], Input2[band], color=dye[band], scatter=scatter, binned=binned, xlabel=xlabel, ylabel=ylabel, band=band, mode=jj)
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
    #ax.annotate(r'$A_{W2}-A_{W1} \/\/ [mag]$', (0.008,0.56), xycoords='figure fraction', size=16, color='black', rotation=90)
    
    #ax.annotate(r'$inclination \/ [deg]$', (0.52,0.02), xycoords='figure fraction', size=16, color='black')
    
    fig.savefig("A_w12_inc.eps")
    fig.savefig("A_w12_inc.png")
    plt.show()
    
################################################################## 
def plot_Rinc(ax, T1, Input1, T2, Input2, color='red', scatter=False, binned=False, xlabel=True, ylabel=True, X_twin=True, Y_twin=True, band='r', mode=1):
    
    myDic = {}
    band2 = 'w1'
    pgc     = Input1[0]
    pc0     = Input1[2]
    inc     = Input1[3]
    table = T1[5]
    Epc0  = table['Epc0']
    Einc  = table['inc_e']
    if mode==1: 
        pc0 = 1.021*pc0-0.095
        Epc0*=1.021
        band2 = 'w2'
    a,b,c,d, alpha, beta, gamma, Ealpha, Ebeta = getReddening_params(band1=band, band2=band2)
    q2 = 10**(-1.*gamma)
    F = log_a_b(inc, q2)
    dF2 = Elogab2(inc, q2, Einc)
    A1 = F*(a*pc0**3+b*pc0**2+c*pc0+d)
    dA1 = np.sqrt(dF2*(a*pc0**3+b*pc0**2+c*pc0+d)**2+(F*(3*a*pc0**2+2*b*pc0+c)*Epc0)**2)
    for i in range(len(pgc)):
        myDic[pgc[i]]=[inc, A1[i], dA1[i]]
    

    pgc     = Input2[0]
    pc0     = Input2[2]
    inc     = Input2[3]
    table = T2[5]
    Epc0  = table['Epc0']
    Einc  = table['inc_e']
    a,b,c,d, alpha, beta, gamma, Ealpha, Ebeta = getReddening_params(band1=band, band2='w2')
    q2 = 10**(-1.*gamma)
    F = log_a_b(inc, q2)
    dF2 = Elogab2(inc, q2, Einc)
    A2 = F*(a*pc0**3+b*pc0**2+c*pc0+d)
    dA2 = np.sqrt(dF2*(a*pc0**3+b*pc0**2+c*pc0+d)**2+(F*(3*a*pc0**2+2*b*pc0+c)*Epc0)**2)
    INC = []
    dA = []
    for i in range(len(pgc)):
        if pgc[i] in myDic:
            INC.append(inc[i])
            dA.append(A2[i]-myDic[pgc[i]][1])
    
    dA = np.asarray(dA)
    if scatter:
        ax.plot(INC, dA, 'o', color='black', markersize=1, alpha=0.15)
        if mode==1:
            
            rms = np.sqrt(np.mean(dA**2))
            ax.text(46,-0.012, r'$RMS=$'+'%.2f'%rms+' mag', fontsize=12, color='k')

    if binned:
        xl = []
        yl= []
        yel=[]
        
        low = 45; high=90
        for i in np.arange(low,high,5):
            
            x = []
            y = []
            for ii in range(len(dA)):
                xi = INC[ii]
                if xi>i and xi<=i+5:
                    x.append(xi)
                    y.append(dA[ii])
            if len(x)>0:
                
                x = np.asarray(x)
                y = np.asarray(y)
                
                average   = np.median(y)
                stdev = np.std(y)
                
                index = np.where(y<average+2.*stdev)
                x = x[index]
                y = y[index]
                
                index = np.where(y>average-2.*stdev)
                x = x[index]
                y = y[index]        

                ax.errorbar(np.median(x), np.median(y), yerr=np.std(y), fmt='o', color=color, markersize=5)
                
                xl.append(np.median(x))
                yl.append(np.median(y))
                yel.append(np.std(y))
            
            
    ax.tick_params(which='major', length=6, width=1.5, direction='in')
    ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    ax.minorticks_on()

    ax.plot([40,100], [0,0], 'k:')
    
    if xlabel: ax.set_xlabel(r'$inclination \/ [deg]$', fontsize=16, labelpad=7)
    if mode==1:
        if ylabel: ax.set_ylabel(r'$A^{(i)}_{W2}-[A^{(i)}_{W2}]$', fontsize=16) 
        ax.text(47,0.02, band, fontsize=14, color=color)
        ylim = [-0.019,0.029]
    else: 
        if ylabel: ax.set_ylabel(r'$A^{(i)}_{W2}-A^{(i)}_{W1} \/\/\/ [mag]$', fontsize=16) 
        ax.text(47,0.10, band, fontsize=14, color=color)
        ylim = [-0.05,0.12]
    
    ax.set_ylim(ylim)     
    ax.set_xlim([41,99]) 
    
    if Y_twin:
        y_ax = ax.twinx()
        y_ax.set_ylim(ylim[0],ylim[1])
        y_ax.set_yticklabels([])
        y_ax.minorticks_on()
        y_ax.tick_params(which='major', length=6, width=1.5, direction='in')
        y_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')        
    
    if X_twin:
        x_ax = ax.twiny()
        x_ax.set_xlim(41,99)
        x_ax.set_xticklabels([])
        x_ax.minorticks_on()
        x_ax.tick_params(which='major', length=6, width=1.0, direction='in')
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


