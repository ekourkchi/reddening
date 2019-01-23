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
def R_model(inc, x_, band):
    
    y_ = 0
    
    if band=='u':
        a=-0.004
        b=-0.016
        c=0.057
        d=0.473
        alpha = 0.021
        beta = -0.158
        gamma = 2.795
        q2 = 10**(-1.*gamma)
        y_ = log_a_b(inc, q2)*(a*x_**3+b*x_**2+c*x_+d)+(alpha*x_+beta)
    if band=='g':
        a=-0.002
        b=-0.014
        c=0.030
        d=0.346
        alpha = 0.032
        beta = -0.101
        gamma = 2.986
        q2 = 10**(-1.*gamma)
        y_ = log_a_b(inc, q2)*(a*x_**3+b*x_**2+c*x_+d)+(alpha*x_+beta)
    if band=='r':
        a=-0.002
        b=-0.012
        c=0.020
        d=0.266
        alpha = 0.042
        beta = -0.072
        gamma = 3.120
        q2 = 10**(-1.*gamma)
        y_ = log_a_b(inc, q2)*(a*x_**3+b*x_**2+c*x_+d)+(alpha*x_+beta)
    if band=='i':
        a=-0.001
        b=-0.011
        c=0.012
        d=0.216
        alpha = 0.042
        beta = -0.044
        gamma = 3.186
        q2 = 10**(-1.*gamma)
        y_ = log_a_b(inc, q2)*(a*x_**3+b*x_**2+c*x_+d)+(alpha*x_+beta)
    if band=='z':
        a=-0.001
        b=-0.009
        c=0.007
        d=0.169
        alpha = 0.044
        beta = -0.019
        gamma = 3.286
        q2 = 10**(-1.*gamma)
        y_ = log_a_b(inc, q2)*(a*x_**3+b*x_**2+c*x_+d)+(alpha*x_+beta)
    return y_
    
    
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
def plot_array(inFile, scatter=False, binned=True, band2='w1'):
    
    R_u, Input_u, T = getBand(inFile, band1 = 'u', band2 = band2)
    R_g, Input_g, T = getBand(inFile, band1 = 'g', band2 = band2)
    R_r, Input_r, T = getBand(inFile, band1 = 'r', band2 = band2)
    R_i, Input_i, T = getBand(inFile, band1 = 'i', band2 = band2)
    R_z, Input_z, T = getBand(inFile, band1 = 'z', band2 = band2)
    R_w1,Input_w1, T = getBand(inFile, band1 = 'w1', band2 = 'w2')
    
    R = {}
    Input = {}
    
    R["u"] = R_u
    R["g"] = R_g
    R["r"] = R_r
    R["i"] = R_i
    R["z"] = R_z
    R["w1"] = R_w1

    Input["u"] = Input_u
    Input["g"] = Input_g
    Input["r"] = Input_r
    Input["i"] = Input_i
    Input["z"] = Input_z
    Input["w1"] = Input_w1
    
    dye = {"u":"blue","g":"green","r":"red","i":"orange","z":"maroon","w1":"purple" }
    
    fig = py.figure(figsize=(15, 12), dpi=100)    
    fig.subplots_adjust(wspace=0, top=0.97, bottom=0.07, left=0.06, right=0.98)
    gs = gridspec.GridSpec(4, 6, height_ratios=[1,1,1,1]) 

    p = 0
    ####################################################
    PC0_0 = 2
    PC0_1 = 4
    for jj in range(4):
        
        
        for band in ['u', 'g','r','i','z','w1']:
            
            xlabel = False; ylabel=False
            #if band=='u': ylabel=True
            if jj==3 and band=='i': xlabel=True
            
            ax = plt.subplot(gs[p]) ; p+=1
            plot_Rinc(ax, R[band], Input[band], pc0_lim=[PC0_0,PC0_1], color=dye[band], scatter=scatter, binned=binned, xlabel=xlabel, ylabel=ylabel, band=band)
            yticks = ax.yaxis.get_major_ticks()
            if band!='u': yticks[-1].label1.set_visible(False)
            if jj!=3: plt.setp(ax.get_xticklabels(), visible=False)
            if band!='u': plt.setp(ax.get_yticklabels(), visible=False)  
              
        
        PC0_0-=2
        PC0_1-=2     
        #####################################################
    
    plt.subplots_adjust(hspace=.0, wspace=0)

    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')    
    ax.annotate(r'$A_{w2}^{inc}$', (0.010,0.52), xycoords='figure fraction', size=16, color='black', rotation=90)

    plt.show()
    
################################################################## 
def plot_Rinc(ax, R, Input, pc0_lim=[-1,1], color='red', scatter=False, binned=False, xlabel=True, ylabel=True, X_twin=True, Y_twin=True, band='r'):
    
    pgc     = Input[0]
    r_w1    = Input[1]
    pc0     = Input[2]
    inc     = Input[3]
    
    index = np.where(pc0>=pc0_lim[0])
    r_w1 = r_w1[index]
    pc0 = pc0[index]
    pgc = pgc[index]
    inc = inc[index]
    R = R[index]

    index = np.where(pc0<pc0_lim[1])
    r_w1 = r_w1[index]
    pc0 = pc0[index]
    pgc = pgc[index]
    inc = inc[index]
    R = R[index]    

    ### Model
    if True:  
        inc__ = np.arange(45,90,0.1)
        
        N = len(inc__)
        r_min = np.zeros(N)
        r_max = np.zeros(N)
        for ii in range(N):
            
            r_lst = []
            for pc0_ in np.arange(pc0_lim[0], pc0_lim[1], 0.1):
                    r = R_model(inc__[ii], pc0_, band)
                    r_lst.append(r)
            r_min[ii] = np.min(r_lst)
            r_max[ii] = np.max(r_lst)
        ax.fill_between(inc__, r_min, r_max, alpha=0.35, facecolor=color)        
        
        
        
        
    if scatter:
        ax.plot(inc, R, 'o', color='black', markersize=1, alpha=0.2)
  
    if binned:
        xl = []
        yl= []
        yel=[]
        
        low = 45; high=90
        for i in np.arange(low,high,5):
            
            x = []
            y = []
            for ii in range(len(R)):
                xi = inc[ii]
                if xi>i and xi<=i+5:
                    x.append(xi)
                    y.append(R[ii])
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
    
    #ax.text(45,0.8, r''+"%.0f" % (c21w_[0])+'$< c21W_1 <'+"%.0f" % (c21w_[1])+'$', color=color, fontsize=11)
    ax.text(47,-0.7, r''+"%.1f" % (pc0_lim[0])+'$< PC0 <$'+"%.1f" % (pc0_lim[1]), fontsize=13)
    
    ax.text(47,1.1, band, fontsize=14, color=color)

    ax.set_ylim([-0.9,1.4])     
    ax.set_xlim([41,99])    
    ax.plot([0,100], [0,0], 'k:')
    
    if xlabel: ax.set_xlabel(r'$inclination \/ [deg]$', fontsize=16)
    if ylabel: ax.set_ylabel(r'$A_{w2}^{(inc)}$', fontsize=16) 
    
    if Y_twin:
        y_ax = ax.twinx()
        y_ax.set_ylim(-0.9,1.4)
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

    
    #if not SigmaR is None and len(SigmaR)>0:
        
        #Y_err_median = np.mean(SigmaR)
        #x0=94; y0=-0.5
        #plt.errorbar([x0], [y0], yerr=[Y_err_median], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)
        

    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
################################################################# 



plot_array('ESN_HI_catal.csv', scatter=True, binned=True, band2='w2')                
