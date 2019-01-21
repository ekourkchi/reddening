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

def plot_Band(ax, band1='r', band2='w1'):
    
    inFile  = 'ESN_HI_catal.csv'
    scaler, pca = transform(inFile, band1=band1, band2=band2)
    
    table = getTable(inFile, band1=band1, band2=band2, faceOn=True)
    
    index, = np.where(table['Wba']>0.01)
    table = trim(table, index)

    text1 = '\overline{'+band1+'}-\overline{W}1'            # example: cr-W1
       
    if band2=='w1':
        text1 = r'\overline{'+band1+r'}-\overline{W}1'            # example: cr-W1
        text2 = 'C_{12W1}'              # example: c21w
    else: 
        text1 = '\overline{'+band1+'}-\overline{W}2'
        if band1=='w1': text1 = r'\overline{W}1-\overline{W}2'
        text2 = 'C_{12W2}'
       
    pgc = table['pgc']
    logWimx = table['logWimx']
    logWimx_e = table['logWimx_e']
    inc = table['inc']
    r_w1 = table['r_w1']
    c21w = table['c21w'] 
    Er_w1 = table['Er_w1']
    Ec21w = table['Ec21w']

    C82  = table['C82_w2']   # concentration 80%/20%
    mu50 = table['w2']+2.5*np.log10(2.*np.pi*(table['R50_w2']*60)**2)-2.5*np.log10(table['Wba'])
    
    data = {'$Log( W_{mx}^i)$':logWimx, '$c21W2$':c21w, '$\mu 50$':mu50}
    d = pd.DataFrame.from_dict(data)
    z_data = scaler.transform(d)
    pca_data = pca.transform(z_data)

    pc0 = pca_data[:,0]
    pc1 = pca_data[:,1]
    pc2 = pca_data[:,2]

    data = {'r-w1':r_w1, '$Log( W_{mx}^i)$':logWimx, '$c21W2$':c21w, '$\mu 50$':mu50, 'C82':C82, 'pc0':pc0}
    d = pd.DataFrame.from_dict(data)
    corr = d.corr()
    
    a0, b0  = np.polyfit(pc0, r_w1, 1)
    delta = np.abs(r_w1-(a0*pc0+b0))
    indx = np.where(delta<1)
    r_w1_ = r_w1[indx]
    pc0_ = pc0[indx]    
    
    AB, cov  = np.polyfit(pc0_, r_w1_, 1, cov=True, full = False)
    a0, b0 = AB[0], AB[1]
    y = np.linspace(-5,5,50)
    x = a0*y+b0
    ax.plot(x,y, 'k--')

    #########################################################################  

    for i in range(len(pgc)):
        if c21w[i]<1  :
            p1, = ax.plot(r_w1[i], pc0[i], 'b.', markersize=7, alpha=0.7, label=r"$"+text2+" < 1$")
        if c21w[i]>=1 and c21w[i]< 3:
            p2, = ax.plot(r_w1[i], pc0[i], 'g.', markersize=7, alpha=0.7, label=r"$1 < "+text2+" < 3$")  
        if c21w[i]>=3:
            p3, = ax.plot(r_w1[i], pc0[i], 'r.', markersize=7, alpha=0.7, label=r"$3 < "+text2+"$")   
        
    ax.set_xlabel('$'+text1+'$', fontsize=16, labelpad=7)
    
    if band1 in ['u','i']:
       ax.set_ylabel('$PC0$', fontsize=16, labelpad=7)
    
    
    rw_lim = [-2.2,1.8]
    if band1=='u': rw_lim = [-0.8,3.2]
    if band1=='g': rw_lim = [-1.8,2.2]
    if band1=='w1': rw_lim = [-1.9,2.1]

    add_axis(ax,rw_lim,[-4,4])
    
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
    #ax.errorbar([x0], [y0], xerr=[Er_w1[i]], yerr=[Y_err_median], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)
    
    
    x0 = 0.45*Xlm[0]+0.55*Xlm[1]
    y0 = 0.60*Ylm[0]+0.40*Ylm[1]
    ax.text(x0,y0, r'$\alpha=$'+'%.2f'%a0+'$\pm$'+'%.2f'%np.sqrt(cov[0][0]), fontsize=14, color='k')

    y0 = 0.70*Ylm[0]+0.30*Ylm[1]
    b = b0 # a0*2.5+b0
    be = np.sqrt(cov[1][1]) #  np.sqrt(cov[0][0]*(2.5**2)+cov[1][1])
    ax.text(x0,y0, r'$\beta=$'+'%.2f'%b+'$\pm$'+'%.2f'%be, fontsize=14, color='k')  
    

    delta = np.abs(r_w1_-(a0*pc0_+b0))
    rms = np.sqrt(np.mean(np.square(delta)))
    Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
    y0 = 0.80*Ylm[0]+0.20*Ylm[1]
    ax.text(x0,y0, r'$RMS=$'+'%.2f'%rms+' mag', fontsize=14, color='k')    
    
    
    y0 = 0.9*Ylm[0]+0.10*Ylm[1]
    ax.text(x0,y0, r'$Corr.=$'+'%.2f'%corr['r-w1']['pc0'], fontsize=14, color='k')    
    

  
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

