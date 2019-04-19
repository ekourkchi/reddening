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
                tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14)    
                
########################################################### Begin
def plot_array(inFile, scatter=False, binned=True, band2='w2'):
    
    AB_u, _, _, table_u  = faceON_surfaceB(inFile, band1 = 'u', band2 = band2)
    AB_g, _, _, table_g  = faceON_surfaceB(inFile, band1 = 'g', band2 = band2)
    AB_r, _, _, table_r  = faceON_surfaceB(inFile, band1 = 'r', band2 = band2)
    AB_i, _, _, table_i  = faceON_surfaceB(inFile, band1 = 'i', band2 = band2)
    AB_z, _, _, table_z  = faceON_surfaceB(inFile, band1 = 'z', band2 = band2)
    AB_w1, _, _, table_w1 = faceON_surfaceB(inFile, band1 = 'w1', band2 = 'w2')
    
    R = {}
    Input = {}
    T = {}
    
    T["u"]  = AB_u
    T["g"]  = AB_g
    T["r"]  = AB_r
    T["i"]  = AB_i
    T["z"]  = AB_z
    T["w1"] = AB_w1
    
    R["u"]  = table_u['r_w1']-(AB_u[0]*table_u['mu50']+AB_u[1])
    R["g"]  = table_g['r_w1']-(AB_g[0]*table_g['mu50']+AB_g[1])
    R["r"]  = table_r['r_w1']-(AB_r[0]*table_r['mu50']+AB_r[1])
    R["i"]  = table_i['r_w1']-(AB_i[0]*table_i['mu50']+AB_i[1])
    R["z"]  = table_z['r_w1']-(AB_z[0]*table_z['mu50']+AB_z[1])
    R["w1"] = table_w1['r_w1']-(AB_w1[0]*table_w1['mu50']+AB_w1[1])

    Input["u"]  = table_u
    Input["g"]  = table_g
    Input["r"]  = table_r
    Input["i"]  = table_i
    Input["z"]  = table_z
    Input["w1"] = table_w1    
    
    dye = {"u":"blue","g":"green","r":"red","i":"orange","z":"maroon","w1":"purple" }
    
    fig = py.figure(figsize=(17, 12), dpi=100)    
    fig.subplots_adjust(wspace=0, top=0.97, bottom=0.07, left=0.06, right=0.98)
    
    if band2=='w2': 
        gs = gridspec.GridSpec(4, 6, height_ratios=[1,1,1,1]) 
    else: gs = gridspec.GridSpec(4, 5, height_ratios=[1,1,1,1]) 

    p = 0
    ####################################################
    inc_0 = 80
    inc_1 = 90
    
    if band2=='w2': 
        band_lst = ['u', 'g','r','i','z','w1']
    else: band_lst = ['u', 'g','r','i','z']
    
    for jj in range(4):
        
        
        for band in band_lst:
            
            xlabel = False; ylabel=False
            #if band=='u': ylabel=True
            if jj==4 and band=='i': xlabel=True
            
            ax = plt.subplot(gs[p]) ; p+=1
            plot_Rinc(ax, T[band], Input[band], inc_lim=[inc_0,inc_1], color=dye[band], scatter=scatter, binned=binned, xlabel=xlabel, ylabel=ylabel, band=band, band2=band2)
            yticks = ax.yaxis.get_major_ticks()
            if band!='u': yticks[-1].label1.set_visible(False)
            if jj!=3: plt.setp(ax.get_xticklabels(), visible=False)
            if band!='u': plt.setp(ax.get_yticklabels(), visible=False)  
              
        
        inc_0-=10
        inc_1-=10     
        #####################################################
    
    plt.subplots_adjust(hspace=.0, wspace=0)

    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')    
    ax.annotate(r'$A_{'+band2+'}^{(i)} \/\/ [mag]$', (0.008,0.56), xycoords='figure fraction', size=16, color='black', rotation=90)
    
    ax.annotate(r'$P_{1,'+band2+'}$', (0.52,0.02), xycoords='figure fraction', size=16, color='black')
    
    fig.savefig("A_"+band2+"_surfB.png")
    plt.show()
    
################################################################## 
def plot_Rinc(ax, T, Input, inc_lim=[85,90], color='red', scatter=False, binned=False, xlabel=True, ylabel=True, X_twin=True, Y_twin=True, band='r', band2='w2'):
    
    r_w1    = Input['r_w1']
    pc0     = Input['mu50']
    inc     = Input['inc']
    
    AB = T
    a0, b0 = AB[0], AB[1]
    
    Er_w1 = Input['Er_w1']
    Epc0  = Input['Emu50']
    
    index = np.where(inc>=inc_lim[0])
    r_w1 = r_w1[index]
    pc0   = pc0[index]
    inc   = inc[index]
    Er_w1 = Er_w1[index]
    Epc0  = Epc0[index]

    index = np.where(inc<inc_lim[1])
    r_w1 = r_w1[index]
    pc0   = pc0[index]
    inc   = inc[index]
    Er_w1 = Er_w1[index]
    Epc0  = Epc0[index]
    
    a,b,c,d, alpha, beta, gamma, Ealpha, Ebeta = getReddening_params_surfB(band1=band, band2=band2)
    
    R = r_w1 - (alpha*pc0+beta)
    dR = np.sqrt(Er_w1**2+(alpha*Epc0)**2+(Ealpha*pc0)**2+Ebeta**2)


    ### Model
    if True:  
        pc0__ = np.arange(20,26,0.5)
        
        N = len(pc0__)
        r_min = np.zeros(N)
        r_max = np.zeros(N)
        for ii in range(N):
            
            r_lst = []
            for inc_ in np.arange(inc_lim[0], inc_lim[1], 1):
                    q2 = 10**(-1.*gamma)
                    r = log_a_b(inc_, q2)*(a*pc0__[ii]**3+b*pc0__[ii]**2+c*pc0__[ii]+d)
                    r_lst.append(r)
            r_min[ii] = np.min(r_lst)
            r_max[ii] = np.max(r_lst)
        #if band!='w1': 
            #ax.fill_between(pc0__, r_min, r_max, alpha=0.30, facecolor=color)


    if scatter:
        ax.plot(pc0, R, 'o', color='black', markersize=1, alpha=0.42)
    
    ### Fitting a curve
    #a, b, c, d  = np.polyfit(pc0,R, 3)
    #y_ = a*x_**3+b*x_**2+c*x_+d
    x_ = np.linspace(20,26,500)
    q2 = 10**(-1.*gamma)
    y_ = log_a_b(np.median(inc), q2)*(a*x_**3+b*x_**2+c*x_+d)

    indx, = np.where(y_>0)
    x_ = x_[indx]
    y_ = y_[indx]
    ax.plot(x_, y_, 'k--')  
        
    if binned:
        xl = []
        yl= []
        yel=[]
        
        low = 20; high=26
        for i in np.arange(low,high,0.5):
            
            x = []
            y = []
            for ii in range(len(R)):
                xi = pc0[ii]
                if xi>i and xi<=i+0.5:
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
    ax.text(-1.2,-0.4, r''+"%.0f" % (inc_lim[0])+'$< inc. <$'+"%.0f" % (inc_lim[1]), fontsize=14)
    
    ax.text(-3,1.3, band, fontsize=14, color=color)

    ax.set_ylim([-0.7,1.7])     
    ax.set_xlim([26,20])    
    ax.plot([20,26], [0,0], 'k:')
    
    #if xlabel: ax.set_xlabel('$P_1$', fontsize=16)
    #if ylabel: ax.set_ylabel(r'$A_{w2}^{(inc)}$', fontsize=16) 
    
    if Y_twin:
        y_ax = ax.twinx()
        y_ax.set_ylim(-0.7,1.7)
        y_ax.set_yticklabels([])
        y_ax.minorticks_on()
        y_ax.tick_params(which='major', length=6, width=1.5, direction='in')
        y_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')        
    
    if X_twin:
        x_ax = ax.twiny()
        x_ax.set_xlim(26,20)
        x_ax.set_xticklabels([])
        x_ax.minorticks_on()
        x_ax.tick_params(which='major', length=6, width=1.0, direction='in')
        x_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')     

    
    if len(dR)>0:
        
        x0=-2.5; y0=-0.3
        plt.errorbar([x0], [y0], xerr=[np.median(pc0)], yerr=[np.median(dR)], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)
        

    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14) 

###########################################################



plot_array('ESN_HI_catal.csv', scatter=True, binned=True, band2='w2')     


