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
index, = np.where(table['logWimx']>1)
table = trim(table, index)
index, = np.where(table['r_w1']<4)
table = trim(table, index)
AB, Delta, table = faceON(table)
c21w1 = table['c21w']
########################################################### Begin
inFile  = 'ESN_HI_catal.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)
table = extinctionCorrect(table)
table = Kcorrection(table)
band1 = 'r'
band2 = 'w2'
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
index, = np.where(table['logWimx']>1)
table = trim(table, index)
index, = np.where(table['r_w1']<4)
table = trim(table, index)
AB, Delta, table = faceON(table)
c21w2 = table['c21w']
########################################################### Begin
fig = py.figure(figsize=(10, 4), dpi=100)   
fig.subplots_adjust(wspace=0, hspace = 0, top=0.97, bottom=0.15, left=0.10, right=0.98)
gs = gridspec.GridSpec(1,2) 
p = 0 
xlim=[-1.5,6.5]; ylim=[0,0.4]


ax = plt.subplot(gs[p]);p+=1
bins = np.arange(-1, 7, 1)
n, bins, patches = ax.hist(c21w1, bins, histtype='step',color=['purple'], label=['label'], fill=False, stacked=True, density=True)
ax.set_ylabel('Normalized density', fontsize=14)
ax.set_xlabel('$C21W_1$', fontsize=14)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
add_axis(ax,xlim,ylim)
S = np.sort(c21w1)
size = len(S)
med_S = np.median(S)
S_u = S[int(round(0.84*size))]
S_l = S[int(round(0.16*size))]
ax.plot([S_l,S_l], [0,0.4], color='k', linestyle=':') 
ax.plot([S_u,S_u], [0,0.4], color='k', linestyle=':')
ax.plot([med_S,med_S], [0,0.4], color='k', linestyle='-')

ax = plt.subplot(gs[p]);p+1
bins = np.arange(-1, 7, 1)
n, bins, patches = ax.hist(c21w2, bins, histtype='step',color=['black'], label=['label'], fill=False, stacked=True, density=True)
ax.set_xlabel('$C21W_2$', fontsize=14)
plt.setp(ax.get_yticklabels(), visible=False) 
ax.set_xlim(xlim)
ax.set_ylim(ylim)
add_axis(ax,xlim,ylim)
S = np.sort(c21w1)
size = len(S)
med_S = np.median(S)
S_u = S[int(round(0.84*size))]
S_l = S[int(round(0.16*size))]
ax.plot([S_l,S_l], [0,0.4], color='k', linestyle=':') 
ax.plot([S_u,S_u], [0,0.4], color='k', linestyle=':')
ax.plot([med_S,med_S], [0,0.4], color='k', linestyle='-')

plt.show()



