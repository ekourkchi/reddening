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

################################################################# 
def add_axis(ax, xlim, ylim):
    
    x1, x2 = xlim[0], xlim[1]
    y1, y2 = ylim[0], ylim[1]
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)

    ax.minorticks_on()
    ax.tick_params(which='major', length=5, width=1.0)
    ax.tick_params(which='minor', length=2, color='#000033', width=1.0)     
    
    # additional Y-axis (on the right)
    y_ax = ax.twinx()
    y_ax.set_ylim(y1, y2)
    y_ax.set_yticklabels([])
    y_ax.minorticks_on()
    y_ax.tick_params(which='major', length=5, width=1.0, direction='in')
    y_ax.tick_params(which='minor', length=2, color='#000033', width=1.0, direction='in')

    # additional X-axis (on the top)
    x_ax = ax.twiny()
    x_ax.set_xlim(x1, x2)
    x_ax.set_xticklabels([])
    x_ax.minorticks_on()
    x_ax.tick_params(which='major', length=5, width=1.0, direction='in')
    x_ax.tick_params(which='minor', length=2, color='#000033', width=1.0, direction='in')
    

########################################################### Begin

def R_model(inc, logWimx, c21w):
    
    ## r - band
    a = -0.006802002435350885
    b = -0.07762235270941023
    c = 0.22179819621547234
    d = 0.43518156986858614
    e = -0.8992560487725044
    alfa = 2.749372013571478
    A = 0.009313224349010222
    B = 0.0032404933280595558
    C = -0.04583217678598152
    D = -0.10488093611844661
    E = 0.19416898288534062
    
    ## u - band
    #a = -0.007354279459235421
    #b = -0.17908490808638042
    #c = 0.47447700276590077
    #d = 0.8245040413230341
    #e = -1.6943768879751735
    #alfa = 2.5601495206921245
    #A = -0.01074817573276338
    #B = 0.2770377487410002
    #C = -0.6231893173944139
    #D = -0.7223116115360697
    #E = 1.5005894514267648
    
    # z - band
    #a = -0.00502155119836949
    #b = -0.05227424826770822
    #c = 0.1501079095875925
    #d = 0.30169351741662986
    #e = -0.6403697422030856
    #alfa = 2.9215307438528932
    #A = 0.010262932256879811
    #B = -0.03701207448926276
    #C = 0.05008130196370296
    #D = 0.04073946208622926
    #E = -0.10647549213832014
    
    q2 = 10**(-1.*alfa)   

    model = log_a_b(inc, q2)*(a*c21w**2+b*logWimx*c21w+c*c21w+d*logWimx+e)+(A*c21w**2+B*logWimx*c21w+C*c21w+D*logWimx+E)
    
    
    return model



################################################################# 
def plot_Rinc_array(R, Input, scatter=False, binned=True, SigmaR=None):
    
    fig = py.figure(figsize=(12, 15), dpi=100)    
    fig.subplots_adjust(wspace=0, top=0.97, bottom=0.07, left=0.08, right=0.98)
    gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1,1,1]) 
    p = 0
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1

    plot_Rinc(ax, R, Input, c21w_=[-2,1], logWimx_=[2.6,3.0], color='blue', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[2.6,3.0], color='green', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)    
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[2,3], logWimx_=[2.6,3.0], color='green', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[3,6], logWimx_=[2.6,3.0], color='red', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[-2,1], logWimx_=[2.4,2.6], color='blue', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ####################################################    
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[2.4,2.6], color='green', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[2,3], logWimx_=[2.4,2.6], color='green', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[3,6], logWimx_=[2.4,2.6], color='red', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################    
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[-2,1], logWimx_=[2.2,2.4], color='blue', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ####################################################    
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[2.2,2.4], color='green', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################    
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[2,3], logWimx_=[2.2,2.4], color='green', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[3,6], logWimx_=[2.2,2.4], color='red', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################     
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[-2,1], logWimx_=[1.8,2.2], color='blue', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    xticks = ax.xaxis.get_major_ticks()
    xticks[-1].label1.set_visible(False)
    #plt.setp(ax.get_xticklabels(), visible=False)
    ####################################################    
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[1.8,2.2], color='green', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    xticks = ax.xaxis.get_major_ticks()
    xticks[-1].label1.set_visible(False)
    #plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[2,3], logWimx_=[1.8,2.2], color='green', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    xticks = ax.xaxis.get_major_ticks()
    xticks[-1].label1.set_visible(False)
    #plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[3,6], logWimx_=[1.8,2.2], color='red', SigmaR=SigmaR, scatter=scatter, binned=binned, xlabel=False, ylabel=False)

    #xticks = ax.xaxis.get_major_ticks()
    #xticks[-1].label1.set_visible(False)
    #plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################     
    
    plt.subplots_adjust(hspace=.0, wspace=0)
    
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')    
    ax.annotate(r'$A_{w2}^{(inc)}$', (0.010,0.52), xycoords='figure fraction', size=18, color='black', rotation=90)
    ax.annotate(r'$inclination \/ [deg]$', (0.50,0.010), xycoords='figure fraction', size=18, color='black')

    plt.show()
    

################################################################# 
def plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[2.4,2.6], color='red', scatter=False, binned=False, xlabel=False, ylabel=False, X_twin=True, Y_twin=True, SigmaR=None):
    
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


    ### Model
    if True:
        
        inc__ = np.arange(45,90,0.1)
        logWimx__ = np.ones_like(inc__)*(logWimx_[0]+logWimx_[1])*0.5
        c21w__ = np.ones_like(inc__)*(c21w_[0]+c21w_[1])*0.5
        R__ = R_model(inc__, logWimx__, c21w__)
            
        #ax.plot(inc__, R__, '-', color=color)

        N = len(inc__)
        r_min = np.zeros(N)
        r_max = np.zeros(N)
        for ii in range(N):
            
            r_lst = []
            for iw in np.arange(logWimx_[0], logWimx_[1], 0.05):
                for ic in np.arange(c21w_[0], c21w_[1], 0.05):
                    r = R_model(inc__[ii], iw, ic)
                    r_lst.append(r)
            r_min[ii] = np.min(r_lst)
            r_max[ii] = np.max(r_lst)
        ax.fill_between(inc__, r_min, r_max, alpha=0.45, facecolor=color)


    if scatter:
        ax.plot(inc, R, 'o', color='black', markersize=2, alpha=0.5)    
    
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
    
    ax.text(45,0.8, r''+"%.0f" % (c21w_[0])+'$< C_{21W2} <'+"%.0f" % (c21w_[1])+'$', color=color, fontsize=14)
    ax.text(45,1.1, r''+"%.1f" % (logWimx_[0])+'$< Log(W^i_{mx}) <$'+"%.1f" % (logWimx_[1]), fontsize=14)

    ax.set_ylim([-0.9,1.4])     
    ax.set_xlim([41,99])    
    ax.plot([0,100], [0,0], 'k:')
    
    if xlabel: ax.set_xlabel(r'$inclination [deg]$', fontsize=16)
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

    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14)
                
    if not SigmaR is None and len(SigmaR)>0:
        
        Y_err_median = np.mean(SigmaR)
        x0=94; y0=-0.5
        plt.errorbar([x0], [y0], yerr=[Y_err_median], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)
        
        

################################################################# 

################################################################# 
def plot_RlogW(R, Input, inc_=[80,90], c21w_=[1,2], color='red', scatter=False):
    
        
    pgc     = Input[0]
    r_w1    = Input[1]
    logWimx = Input[2]
    c21w    = Input[3]
    inc     = Input[4]
      
    index = np.where(inc<=inc_[1])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(inc>=inc_[0])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(c21w>=c21w_[0])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(c21w<=c21w_[1])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]
    
    if scatter:
        plt.plot(logWimx, R, '.', color=color, markersize=2, alpha=0.4)

    xl = []
    yl= []
    yel=[]
    
    low = np.min(logWimx) ; high = np.max(logWimx)
    low = 1.7; high=3
    for i in np.arange(low,high,0.10):
        
        x = []
        y = []
        for ii in range(len(R)):
            xi = logWimx[ii]
            if xi>=i and xi<i+0.1:
                #x.append(xi)
                x.append(i+0.05)
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

            plt.errorbar(np.median(x), np.median(y), yerr=np.std(y), fmt='o', color=color)
            
            xl.append(np.median(x))
            yl.append(np.median(y))
            yel.append(np.std(y))


    plt.ylim([-1,1.5])     
    plt.xlim([1.7,3]) 
    
    
    


################################################################# 
inFile  = 'ESN_HI_catal.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

table = extinctionCorrect(table)
table = Kcorrection(table)

band1 = 'r'
band2 = 'w1'

delta = np.abs(table[band2]-table[band2+'_'])
index, = np.where(delta<=0.15)
table = trim(table, index)

delta = np.abs(table[band1]-table[band1+'_'])
index, = np.where(delta<=0.15)
table = trim(table, index)

table['c21w'] = table['m21'] - table[band2]
table['r_w1'] = table[band1] - table[band2]

table['Ec21w'] = np.sqrt(table['m21_e']**2+0.05**2)
table['Er_w1'] = 0.*table['r_w1']+0.1

index, = np.where(table['logWimx']>1.5)
table = trim(table, index)

index, = np.where(table['r_w1']<4)
table = trim(table, index)

index, = np.where(table['r_w1']>-2.5)
table = trim(table, index)

########################################################Face-ON

## Get the initial estimations using Face-on galaxies
## AB:    a0*logWimx+b0
## Delta: alfa*X**2+beta*X+gama
## table0: table of face-on galaxies
AB, Delta, table0, cov, Delta_e = faceON(table)
#table1 =table0
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
###########################################################

#plot_Rinc_array(R, Input, SigmaR=SigmaR, scatter=True, binned=False)

########################################################### END
N = len(logWimx)
C = np.zeros(N)
for i in range(N):
   if c21w[i]<1: C[i]=1
   if c21w[i]>=1 and c21w[i]< 3:C[i]=2
   if c21w[i]>=3:C[i]=3

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'$Log( W_{mx}^i)$':logWimx, '$r-W1$':r_w1,'$c21W1$':c21w, 'C':C}
d = pd.DataFrame.from_dict(data)
sns.set(style="ticks", color_codes=True)

#sns.set_context("paper")
#hue='C'
pp=sns.pairplot(d, diag_kind="kde", markers=".",plot_kws=dict(s=50, alpha=0.1, linewidth=0),diag_kws=dict(shade=True), vars=['$Log( W_{mx}^i)$', '$r-W1$', '$c21W1$'])
#pp= sns.pairplot(d)

for i in range(3):
 for j in range(3):
    ax = pp.axes[j,i]
    Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
    add_axis(ax, Xlm, Ylm)
    ax.grid(color='grey', linestyle='-', linewidth=0.2)


sns.set(font_scale=1.4)


plt.show()

