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
    
    ax.plot([-1.5,4], [-1.5*a+b,a*4+b], 'k--')

    for i in range(len(pgc)):
        if c21w[i]<1  :
            p1, = ax.plot(r_w1[i], logWimx[i], 'b.', markersize=7, alpha=1, label=r"$"+text2+" < 1$")
            #ax.errorbar(r_w1[i], logWimx[i], xerr=Er_w1[i], yerr=logWimx_e[i], color='b', alpha=0.2)
        if c21w[i]>=1 and c21w[i]< 2:
            p2, = ax.plot(r_w1[i], logWimx[i], 'g.', markersize=7, alpha=1, label=r"$1 < "+text2+" < 3$")  
            #ax.errorbar(r_w1[i], logWimx[i], xerr=Er_w1[i], yerr=logWimx_e[i], color='g', alpha=0.2)
        if c21w[i]>=2 and c21w[i]< 3:
            ax.plot(r_w1[i], logWimx[i], '.', color='green', markersize=7, alpha=1)   
            #ax.errorbar(r_w1[i], logWimx[i], xerr=Er_w1[i], yerr=logWimx_e[i], color='g', alpha=0.2)
        if c21w[i]>=3:
            p3, = ax.plot(r_w1[i], logWimx[i], 'r.', markersize=7, alpha=1, label=r"$3 < "+text2+"$")   
            #ax.errorbar(r_w1[i], logWimx[i], xerr=Er_w1[i], yerr=logWimx_e[i], color='r', alpha=0.2)
        
    ax.set_xlabel('$'+text1+'$', fontsize=16, labelpad=7)
    
    if band1 in ['u','i']:
       ax.set_ylabel(r'$Log( W_{mx}^i)$', fontsize=16, labelpad=7)
    
    
    rw_lim = [-2.2,1.8]
    if band1=='u': rw_lim = [-0.8,3.2]
    if band1=='g': rw_lim = [-1.8,2.2]
    if band1=='w1': rw_lim = [-1.9,2.1]

    add_axis(ax,rw_lim,[1.7,3])
    
    # Legend
    lns = [p1, p2, p3]
    if band1=='i':  ax.legend(handles=lns, loc=1, fontsize=13)
    #if band1=='w1':  
        #ax.legend(handles=lns, loc=4, fontsize=11)
        #band1 = '{w1}'

    X_err_median = np.median(Er_w1[i])
    Y_err_median = np.median(logWimx_e[i])
    Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
    x0 = 0.9*Xlm[0]+0.1*Xlm[1]
    y0 = 0.2*Ylm[0]+0.8*Ylm[1]
    ax.errorbar([x0], [y0], xerr=[Er_w1[i]], yerr=[Y_err_median], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)
    
    
    x0 = 0.45*Xlm[0]+0.55*Xlm[1]
    y0 = 0.65*Ylm[0]+0.35*Ylm[1]
    ax.text(x0,y0, r'$\alpha=$'+'%.2f'%a0+'$\pm$'+'%.2f'%np.sqrt(cov[0][0]), fontsize=14, color='k')
    

    y0 = 0.78*Ylm[0]+0.22*Ylm[1]
    b = b0 # a0*2.5+b0
    be = np.sqrt(cov[1][1]) #  np.sqrt(cov[0][0]*(2.5**2)+cov[1][1])
    ax.text(x0,y0, r'$\beta=$'+'%.2f'%b+'$\pm$'+'%.2f'%be, fontsize=14, color='k')    
    #########################################################################  

    #x0 = 0.97*Xlm[0]+0.03*Xlm[1]
    #y0 = 0.10*Ylm[0]+0.90*Ylm[1]
    #ax.text(x0,y0, r'$\alpha_{'+band1+'-W1}=$'+'%.2f'%a0+'$\pm$'+'%.2f'%np.sqrt(cov[0][0]), fontsize=11, color='maroon')
    
    #y0 = 0.20*Ylm[0]+0.80*Ylm[1]
    #b = b0 # a0*2.5+b0
    #be = np.sqrt(cov[1][1]) #  np.sqrt(cov[0][0]*(2.5**2)+cov[1][1])
    #ax.text(x0,y0, r'$\beta_{'+band1+'-W1}=$'+'%.2f'%b+'$\pm$'+'%.2f'%be, fontsize=11, color='maroon')  
################################################################# 
################################################################# 
################################################################# 
################################################################# 



fig = py.figure(figsize=(13, 8), dpi=100)   
fig.subplots_adjust(wspace=0, hspace = 0.3, top=0.97, bottom=0.10, left=0.07, right=0.98)
gs = gridspec.GridSpec(2, 3) 
p = 0

ax = plt.subplot(gs[p]) ; p+=1 
plot_Band(ax, band1='u', band2='w2')

ax = plt.subplot(gs[p]) ; p+=1  
plot_Band(ax, band1='g', band2='w2')
plt.setp(ax.get_yticklabels(), visible=False)

ax = plt.subplot(gs[p]) ; p+=1  
plot_Band(ax, band1='r', band2='w2')
plt.setp(ax.get_yticklabels(), visible=False)

ax = plt.subplot(gs[p]) ; p+=1 
plot_Band(ax, band1='i', band2='w2')

ax = plt.subplot(gs[p]) ; p+=1  
plot_Band(ax, band1='z', band2='w2')
plt.setp(ax.get_yticklabels(), visible=False)

ax = plt.subplot(gs[p]) ; p+=1  
plot_Band(ax, band1='w1', band2='w2')
plt.setp(ax.get_yticklabels(), visible=False)

plt.show()

