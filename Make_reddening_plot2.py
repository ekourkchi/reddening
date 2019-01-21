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
import copy 
from matplotlib.font_manager import FontProperties
from redTools import *
from Kcorrect import *
from matplotlib import *
################################################################# 
def plot_array(inFile, scatter=False, binned=True, band2='w1'):
    
    R_u, SigmaR_u, Input_u = getBand(inFile, band1 = 'u', band2 = band2)
    R_g, SigmaR_g, Input_g = getBand(inFile, band1 = 'g', band2 = band2)
    R_r, SigmaR_r, Input_r = getBand(inFile, band1 = 'r', band2 = band2)
    R_i, SigmaR_i, Input_i = getBand(inFile, band1 = 'i', band2 = band2)
    R_z, SigmaR_z, Input_z = getBand(inFile, band1 = 'z', band2 = band2)
    R_w1, SigmaR_w1, Input_w1 = getBand(inFile, band1 = 'w1', band2 = 'w2')
    
    R = {}
    SigmaR = {}
    Input = {}
    
    R["u"] = R_u
    R["g"] = R_g
    R["r"] = R_r
    R["i"] = R_i
    R["z"] = R_z
    R["w1"] = R_w1
    SigmaR["u"] = SigmaR_u
    SigmaR["g"] = SigmaR_g
    SigmaR["r"] = SigmaR_r
    SigmaR["i"] = SigmaR_i
    SigmaR["z"] = SigmaR_z
    SigmaR["w1"] = SigmaR_w1
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
    logWi0 = 2.6
    logWi1 = 3.0
    for jj in range(4):
        
        
        for band in ['u', 'g','r','i','z','w1']:
            
            xlabel = False; ylabel=False
            #if band=='u': ylabel=True
            if jj==3 and band=='i': xlabel=True
            
            ax = plt.subplot(gs[p]) ; p+=1
            plot_Rinc(ax, R[band], Input[band], c21w_=[0,8], logWimx_=[logWi0,logWi1], color=dye[band], SigmaR=SigmaR[band], scatter=scatter, binned=binned, xlabel=xlabel, ylabel=ylabel, band=band)
            yticks = ax.yaxis.get_major_ticks()
            if band!='u': yticks[-1].label1.set_visible(False)
            if jj!=3: plt.setp(ax.get_xticklabels(), visible=False)
            if band!='u': plt.setp(ax.get_yticklabels(), visible=False)  
              
        
        if jj==0: logWi1-=0.4 
        else: logWi1-=0.2

        if jj==2: logWi0-=0.4 
        else: logWi0-=0.2        
        #####################################################
    
    plt.subplots_adjust(hspace=.0, wspace=0)

    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')    
    ax.annotate(r'$A_{w2}^{(inc)}$', (0.010,0.52), xycoords='figure fraction', size=16, color='black', rotation=90)
    
    
    
    plt.show()
    

################################################################# 
def plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[2.4,2.6], color='red', scatter=False, binned=False, xlabel=True, ylabel=True, X_twin=True, Y_twin=True, SigmaR=None, band='r'):
    
    pgc     = Input[0]
    r_w1    = Input[1]
    logWimx = Input[2]
    c21w    = Input[3]
    inc     = Input[4]
    
    index = np.where(c21w<c21w_[1])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]
    if not SigmaR is None: SigmaR = SigmaR[index]

    index = np.where(c21w>=c21w_[0])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]
    if not SigmaR is None: SigmaR = SigmaR[index]

    index = np.where(logWimx>=logWimx_[0])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]
    if not SigmaR is None: SigmaR = SigmaR[index]

    index = np.where(logWimx<logWimx_[1])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]    
    if not SigmaR is None: SigmaR = SigmaR[index]

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
    ax.text(47,-0.7, r''+"%.1f" % (logWimx_[0])+'$< Log(W^i_{mx}) <$'+"%.1f" % (logWimx_[1]), fontsize=13)
    
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
def getBand(inFile, band1 = 'r', band2 = 'w1'):

    table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

    table = extinctionCorrect(table)
    table = Kcorrection(table)

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
    ## table0: table of face-on galaxies
    AB, Delta, table0, cov, Delta_e = faceON(table)

    ########################################################Inclined
    ### Inclined
    if True:
        index, = np.where(table['Sqlt']>3)
        table1 = trim(table, index)

        index, = np.where(table1['Wqlt']>3)
        table1 = trim(table1, index)

        index, = np.where(table1['flag']==0)
        table1 = trim(table1, index)

    pgc = table1['pgc']
    logWimx = table1['logWimx']
    logWimx_e = table1['logWimx_e']
    inc = table1['inc']
    r_w1 = table1['r_w1']
    c21w = table1['c21w'] 
    Er_w1 = table1['Er_w1']
    Ec21w = table1['Ec21w']


    Input = [pgc, r_w1, logWimx, c21w, inc]
    R = Reddening(r_w1, logWimx, c21w, AB, Delta)

    a0 = AB[0]  
    alfa = Delta[0]
    beta = Delta[1]

    SigmaR2 = np.ones_like(R)
    for i in range(len(R)):
        
        if c21w[i]>-beta/2./alfa:
            SigmaR2[i] = 0.1**2+((2.*alfa*c21w[i]+beta)**2)*Ec21w[i]**2+(a0*logWimx_e[i])**2
        else:
            SigmaR2[i] = 0.1**2+(a0*logWimx_e[i])**2

    SigmaR=np.sqrt(SigmaR2)
    
    return R, SigmaR, Input
###########################################################




plot_array('ESN_HI_catal.csv', scatter=True, binned=True, band2='w1')











########################################################### END

