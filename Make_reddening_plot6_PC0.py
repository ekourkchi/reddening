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
from linear_mcmc import *
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
    
    fig = py.figure(figsize=(5,6), dpi=100)    
    fig.subplots_adjust(wspace=0, top=0.97, bottom=0.10, left=0.2, right=0.98)
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.9,0.5]) 

    ####################################################
    
    band_lst = ['r']
    
    for jj in range(1):
        
        
        for band in band_lst:
            
            ylabel=True
            xlabel=True
            
            p=0
            ax = plt.subplot(gs[p]) ; p+=1
            
            plot_Rinc(gs, T1[band], Input1[band], T2[band], Input2[band], color=dye[band], scatter=scatter, binned=binned, xlabel=xlabel, ylabel=ylabel, band=band)
            yticks = ax.yaxis.get_major_ticks()
            #if band!='u': yticks[-1].label1.set_visible(False)
            #if band!='u': plt.setp(ax.get_yticklabels(), visible=False)  
                 
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
    
    fig.savefig("P0_w12.eps")
    fig.savefig("P0_w12.png")
    plt.show()
    
################################################################## 
def plot_Rinc(gs, T1, Input1, T2, Input2, color='red', scatter=False, binned=False, xlabel=True, ylabel=True, X_twin=True, Y_twin=True, band='r'):
    
    p=0
    ax = plt.subplot(gs[p]) ; p+=1
    
    myDic = {}
    
    pgc     = Input1[0]
    pc0     = Input1[2]
    inc     = Input1[3]
    table = T1[5]
    Epc0  = table['Epc0']
    Einc  = table['inc_e']
    a,b,c,d, alpha, beta, gamma, Ealpha, Ebeta = getReddening_params(band1=band, band2='w1')
    q2 = 10**(-1.*gamma)
    F = log_a_b(inc, q2)
    dF2 = Elogab2(inc, q2, Einc)
    A1 = F*(a*pc0**3+b*pc0**2+c*pc0+d)
    dA1 = np.sqrt(dF2*(a*pc0**3+b*pc0**2+c*pc0+d)**2+(F*(3*a*pc0**2+2*b*pc0+c)*Epc0)**2)
    for i in range(len(pgc)):
        myDic[pgc[i]]=[pc0[i],Epc0[i]]
    
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
    PC_w1 = []
    PC_w2 = []
    EPC_w1 = []
    EPC_w2 = []
    for i in range(len(pgc)):
        if pgc[i] in myDic:
            PC_w2.append(pc0[i])
            EPC_w2.append(Epc0[i])
            PC_w1.append(myDic[pgc[i]][0])
            EPC_w1.append(myDic[pgc[i]][1])
            
    
    PC_w1 = np.asarray(PC_w1)
    PC_w2 = np.asarray(PC_w2)
    EPC_w1 = np.asarray(EPC_w1)
    EPC_w2 = np.asarray(EPC_w2)
    if scatter:
        ax.plot(PC_w1, PC_w2, 'o', color='black', markersize=2, alpha=0.2)
    
    ### Fitting a curve
    #AB, cov  = np.polyfit(PC_w1,PC_w2, 1, cov=True, full = False)
    #m, b = AB[0], AB[1]
    x_ = np.linspace(-4,4,50)
    #y_ = m*x_+b
    #ax.plot(x_, y_, 'r--') 
    
    M,B,samples=linMC(PC_w1, PC_w2, EPC_w1, EPC_w2)
    m = M[0] ; me=0.5*(M[1]+M[2])
    b = B[0] ; be=0.5*(B[1]+B[2])
    y_, yu, yl = linSimul(samples, x_, size=500)
    ax.fill_between(x_, y_+2*yu, y_-2*yl, color='r', alpha=0.5, edgecolor="none")
    ax.plot(x_, m*x_+b, 'r--') 
    
    
    delta = np.abs(PC_w2-(m*PC_w1+b))
    rms = np.sqrt(np.median(np.square(delta)))
    ax.text(0,-2, "m= "+"%.3f" % m+'$\pm$'+"%.3f" % me, fontsize=14)
    ax.text(0,-2.5, "b= "+"%.3f" % b+'$\pm$'+"%.3f" % be, fontsize=14)
    ax.text(0,-3, r'$RMS=$'+'%.3f'%rms, fontsize=14, color='k')   
    plt.errorbar([-2.5], [2.5], xerr=[np.median(EPC_w1)], yerr=[np.median(EPC_w2)], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)
        
    if binned:
        xl = []
        yl= []
        yel=[]
        
        low = -4; high=3.5
        for i in np.arange(low,high,0.5):
            
            x = []
            y = []
            for ii in range(len(PC_w2)):
                xi = PC_w1[ii]
                if xi>i and xi<=i+0.5:
                    x.append(xi)
                    y.append(PC_w2[ii])
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

                ax.errorbar(np.median(x), np.median(y), yerr=np.std(y), fmt='o', color=color, markersize=5)
                
                xl.append(np.median(x))
                yl.append(np.median(y))
                yel.append(np.std(y))
            
            
    ax.tick_params(which='major', length=6, width=1.5, direction='in')
    ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    ax.minorticks_on()
    

    ax.set_ylim([-3.6,3.6])     
    ax.set_xlim([-3.6,3.6])    
    
    if xlabel: ax.set_xlabel(r'$P_{1,w1}$', fontsize=16)
    if ylabel: ax.set_ylabel(r'$P_{1,w2}$', fontsize=16) 
    
    if Y_twin:
        y_ax = ax.twinx()
        y_ax.set_ylim(-3.6,3.6)
        y_ax.set_yticklabels([])
        y_ax.minorticks_on()
        y_ax.tick_params(which='major', length=6, width=1.5, direction='in')
        y_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')        
    
    if X_twin:
        x_ax = ax.twiny()
        x_ax.set_xlim(-3.6,3.6)
        x_ax.set_xticklabels([])
        x_ax.minorticks_on()
        x_ax.tick_params(which='major', length=6, width=1.0, direction='in')
        x_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')     

       

    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14)
                
    
    ax = plt.subplot(gs[p])
    ax.plot(PC_w1, PC_w2-PC_w1, 'o', color='black', markersize=2, alpha=0.2)
    ax.plot(x_, (m*x_+b)-x_, 'r--') 
    ax.fill_between(x_, (y_+2*yu)-x_, (y_-2*yl)-x_, color='r', alpha=0.5, edgecolor="none")
    
    ax.set_xlabel(r'$P_{1,w1}$', fontsize=16)
    ax.set_ylabel(r'$P_{1,w2}-P_{1,w1}$', fontsize=16)
    ax.set_xlim([-3.6,3.6])
    ax.set_ylim([-0.5,0.5])

    ax.tick_params(which='major', length=6, width=1.5, direction='in')
    ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    ax.minorticks_on() 
    
    if Y_twin:
        y_ax = ax.twinx()
        y_ax.set_ylim(-0.5,0.5)
        y_ax.set_yticklabels([])
        y_ax.minorticks_on()
        y_ax.tick_params(which='major', length=6, width=1.5, direction='in')
        y_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')        
    
    if X_twin:
        x_ax = ax.twiny()
        x_ax.set_xlim(-3.6,3.6)
        x_ax.set_xticklabels([])
        x_ax.minorticks_on()
        x_ax.tick_params(which='major', length=6, width=1.0, direction='in')
        x_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')     

      

    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14)
###########################################################



plot_array('ESN_HI_catal.csv', scatter=True, binned=False)     


