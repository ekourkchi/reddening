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
import george
from george import kernels

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
def plot_array(inFile, scatter=False, binned=True, band2='w2'):
    
    R_u, Input_u, T_u  = getBand(inFile, band1 = 'u', band2 = band2)
    R_g, Input_g, T_g  = getBand(inFile, band1 = 'g', band2 = band2)
    R_r, Input_r, T_r  = getBand(inFile, band1 = 'r', band2 = band2)
    R_i, Input_i, T_i  = getBand(inFile, band1 = 'i', band2 = band2)
    R_z, Input_z, T_z  = getBand(inFile, band1 = 'z', band2 = band2)
    R_w1,Input_w1, T_w1 = getBand(inFile, band1 = 'w1', band2 = 'w2')
    
    R = {}
    Input = {}
    T = {}
    
    T["u"]  = T_u
    T["g"]  = T_g
    T["r"]  = T_r
    T["i"]  = T_i
    T["z"]  = T_z
    T["w1"] = T_w1
    
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
    if band2=='w2': 
        gs = gridspec.GridSpec(4, 6, height_ratios=[1,1,1,1]) 
    else: gs = gridspec.GridSpec(4, 5, height_ratios=[1,1,1,1])

    p = 0
    ####################################################
    PC0_0 = 2
    PC0_1 = 4
    
    if band2=='w2': 
        band1_lst = ['u', 'g','r','i','z','w1']
    else: band1_lst = ['u', 'g','r','i','z']
    
    for jj in range(4):

        for band1 in band1_lst:
            
            xlabel = False; ylabel=False
            #if band1=='u': ylabel=True
            if jj==3 and band1=='i': xlabel=True
            
            ax = plt.subplot(gs[p]) ; p+=1
            plot_Rinc(ax, T[band1], Input[band1], pc0_lim=[PC0_0,PC0_1], color=dye[band1], scatter=scatter, binned=binned, xlabel=xlabel, ylabel=ylabel, band1=band1, band2=band2)
            yticks = ax.yaxis.get_major_ticks()
            if band1!='u': yticks[-1].label1.set_visible(False)
            if jj!=3: plt.setp(ax.get_xticklabels(), visible=False)
            if band1!='u': plt.setp(ax.get_yticklabels(), visible=False)  
              
        
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
    ax.annotate(r'$A_{'+band2+'}^{(i)} \/\/ [mag]$', (0.010,0.56), xycoords='figure fraction', size=16, color='black', rotation=90)
    
    ax.annotate(r'$inclination \/ [deg]$', (0.47,0.02), xycoords='figure fraction', size=16, color='black')
    
    fig.savefig("A_"+band2+"_inc_GPgrg.png")
    plt.show()
    
################################################################## 
def plot_Rinc(ax, T, Input, pc0_lim=[-1,1], color='red', scatter=False, binned=False, xlabel=True, ylabel=True, X_twin=True, Y_twin=True, band1='r', band2='w2'):
    
    pgc     = Input[0]
    r_w1    = Input[1]
    pc0     = Input[2]
    inc     = Input[3]
    
    AB = T[2] ; table = T[5]
    a0, b0 = AB[0], AB[1]
    
    Er_w1 = table['Er_w1']
    Epc0  = table['Epc0']
    Einc  = table['inc_e']    

    a,b,c,d, alpha, beta, gamma, Ealpha, Ebeta = getReddening_params(band1=band1, band2=band2)
    q2 = 10**(-1.*gamma)
    F = log_a_b(inc, q2)
    dF2 = Elogab2(inc, q2, Einc)
    A = F*(a*pc0**3+b*pc0**2+c*pc0+d)
    dA = np.sqrt(dF2*(a*pc0**3+b*pc0**2+c*pc0+d)**2+(F*(3*a*pc0**2+2*b*pc0+c)*Epc0)**2)
    R = r_w1-(alpha*pc0+beta)
    dR = np.sqrt(Er_w1**2+(alpha*Epc0)**2+(Ealpha*pc0)**2+Ebeta**2)
    D = R - A

    N = len(pc0)
    dR2 = Er_w1**2+(alpha*Epc0)**2+(Ealpha*pc0)**2
    noise2 = (dR2+dA**2)*np.eye(N)

    X = np.ones(shape = (2,N))
    X[0] = pc0
    X[1] = inc
    X = X.T
    y = D

    if band1=='u': theta = [ 2.12107366,  4.62358258, -2.67502258,  0.12339909]
    if band1=='g': theta = [3.29876662, 7.30325097, 0.24330005, 0.07810393]
    if band1=='r': theta = [3.5360765 , 7.51644074, 0.32119272, 0.07308647]
    if band1=='i': theta = [3.6535672 , 8.04667536, 0.40698926, 0.08026879]
    if band1=='z': theta = [3.78077626, 8.14568691, 0.09446865, 0.08405013]
    if band1=='w1': theta = [2.20842506e+01,  1.32881714e+01, -2.13119882e+01,  1.01638499e-02]
    
    l1 = np.exp(theta[0])
    l2 = np.exp(theta[1])
    sigma = np.exp(theta[2])
    yerr = np.diagonal(np.sqrt(noise2))+theta[3]
    kernel = sigma * kernels.Matern52Kernel([l1,l2], ndim=2, axes=[0, 1])

    gp = george.GP(kernel)
    gp.compute(X, yerr)
    
    index = np.where(pc0>=pc0_lim[0])
    r_w1 = r_w1[index]
    pc0   = pc0[index]
    pgc   = pgc[index]
    inc   = inc[index]
    Er_w1 = Er_w1[index]
    Epc0  = Epc0[index]

    index = np.where(pc0<pc0_lim[1])
    r_w1 = r_w1[index]
    pc0   = pc0[index]
    pgc   = pgc[index]
    inc   = inc[index]
    Er_w1 = Er_w1[index]
    Epc0  = Epc0[index]
    
    
    R = r_w1 - (alpha*pc0+beta)
    dR = np.sqrt(Er_w1**2+(alpha*Epc0)**2+(Ealpha*pc0)**2+Ebeta**2)    

    ### Model
    if True:  
        _pc, _inc = np.linspace(pc0_lim[0], pc0_lim[1],20), np.linspace(45,90,20)
        _pc, _inc = np.meshgrid(_pc, _inc)
        X_ = np.c_[_pc.ravel(), _inc.ravel()]
        _A, var_A = gp.predict(y, X_, return_var=True)     
        _A = _A.reshape(_pc.shape)
        
        F__ = log_a_b(_inc, q2)
        _A += F__*(a*_pc**3+b*_pc**2+c*_pc+d)
        
        
        inc__ = np.linspace(45,90,20)
        
        N = len(inc__)
        r_min = np.zeros(N)
        r_max = np.zeros(N)
        r_med = np.zeros(N)
        for ii in range(N):
            
            indx = np.where(_inc==inc__[ii])
            r_min[ii] = np.min(_A[indx])
            r_max[ii] = np.max(_A[indx])
            r_med[ii] = np.median(_A[indx])
        #if band1!='w1': 
        ax.fill_between(inc__, r_min, r_max, alpha=0.35, facecolor=color)        
        
    if scatter:
        ax.plot(inc, R, 'o', color='black', markersize=1, alpha=0.4)
        
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

        
    ### Fitting a curve
    ax.plot(inc__, r_med, 'k--')             
            
    ax.tick_params(which='major', length=6, width=1.5, direction='in')
    ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    ax.minorticks_on()
    
    #ax.text(45,0.8, r''+"%.0f" % (c21w_[0])+'$< c21W_1 <'+"%.0f" % (c21w_[1])+'$', color=color, fontsize=11)
    ax.text(52,-0.7, r''+"%.1f" % (pc0_lim[0])+'$< P_{0,'+band2+'} <$'+"%.1f" % (pc0_lim[1]), fontsize=13)
    
    ax.text(47,1.4, band1, fontsize=14, color=color)

    ax.set_ylim([-0.9,1.7])     
    ax.set_xlim([41,99])    
    ax.plot([0,100], [0,0], 'k:')
    
    #if xlabel: ax.set_xlabel(r'$inclination \/ [deg]$', fontsize=16)
    #if ylabel: ax.set_ylabel(r'$A_{w2}^{(inc)}$', fontsize=16) 
    
    if Y_twin:
        y_ax = ax.twinx()
        y_ax.set_ylim(-0.9,1.7)
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

    
    if len(dR)>0:
        
        x0=47; y0=1.
        plt.errorbar([x0], [y0], yerr=[np.median(dR)], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)
        

    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
################################################################# 



plot_array('ESN_HI_catal.csv', scatter=True, binned=True, band2='w2')                
