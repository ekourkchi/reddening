import os
import sys
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
from string import ascii_letters
import pandas as pd
import seaborn as sns
import matplotlib
import sklearn.datasets as ds
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

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
    
    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
########################################################### Begin
#### All but face-on
inFile  = 'ESN_HI_catal.csv'
band1 = 'r'
band2 = 'w2'


if band2=='w1':
        text1 = r'\overline{'+band1+r'}-\overline{W}1'            # example: cr-W1
        text2 = 'C_{21W1}'              # example: c21w
else: 
        text1 = '\overline{'+band1+'}-\overline{W}2'
        if band1=='w1': text1 = r'\overline{W}1-\overline{W}2'
        text2 = 'C_{21W2}'

## get transformations from non-Face-ON galaxies
scaler, pca = transform(inFile, band1=band1, band2=band2)
################################

#### Face ON
table = getTable(inFile, band1=band1, band2=band2, faceOn=True)

pgc = table['pgc']
logWimx = table['logWimx']
logWimx_e = table['logWimx_e']
inc = table['inc']
r_w1 = table['r_w1']
c21w = table['c21w'] 
Er_w1 = table['Er_w1']
Ec21w = table['Ec21w']

C82  = table['C82_w2']   # concentration 80%/20%
EC82 = table['EC82']
mu50 = table['mu50']
Emu50 = table['Emu50']

print len(logWimx)

data = {'$Log( W_{mx}^i)$':logWimx, '$c21W2$':c21w, '$\mu 50$':mu50}
order_of_keys = ['$Log( W_{mx}^i)$', '$c21W2$', '$\mu 50$']
list_of_tuples = [(key, data[key]) for key in order_of_keys]
data = OrderedDict(list_of_tuples)
n_comp = len(data)
d = pd.DataFrame.from_dict(data)
z_data = scaler.transform(d)
pca_data = pca.transform(z_data)
s = scaler.scale_
pca_inv_data = pca.inverse_transform(np.eye(n_comp)) # coefficients to make PCs from features
p0 = pca_inv_data[0,0]
p1 = pca_inv_data[0,1]
p2 = pca_inv_data[0,2]
Epc0 = np.sqrt((p0*logWimx_e/s[0])**2+(p1*Ec21w/s[1])**2+(p2*Emu50/s[2])**2)

fig = py.figure(figsize=(5, 4), dpi=100)
fig.subplots_adjust(top=0.97, bottom=0.15, left=0.18, right=0.98)
ax = fig.add_subplot(111)

a0, b0  = np.polyfit(logWimx, r_w1, 1)
delta = np.abs(r_w1-(a0*logWimx+b0))
indx = np.where(delta<1.)
r_w1_ = r_w1[indx]
logWimx_ = logWimx[indx]

      
a0, b0  = np.polyfit(logWimx_, r_w1_, 1)
y = np.linspace(1,3,50)
x = a0*y+b0


delta = r_w1-(a0*logWimx+b0)
ax.plot(c21w, delta, '.', color='black', markersize=3, alpha=0.5)
add_axis(ax,[-2,6],[-1.5,1.5])
for i in range(-1,6):
    
    x = []
    y = []
    for ii in range(len(c21w)):
        xi = c21w[ii]
        if xi>=i and xi<i+1:
            x.append(xi)
            y.append(delta[ii])
    if len(x)>0:
        ax.errorbar(np.median(x), np.median(y), yerr=np.std(y), xerr=np.std(x), fmt='o', color='red', alpha=0.8)
 
ax.set_ylabel(r'$'+'('+text1+')-('+text1+')_{fit}$', fontsize=14)
ax.set_xlabel(r'$'+text2+'$', fontsize=14) 


ax.plot([-2,6], [0,0], 'k:')


X_err_median = np.median(Ec21w)
Y_err_median = np.median(delta*0.+0.1)
Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
x0 = 0.9*Xlm[0]+0.1*Xlm[1]
y0 = 0.2*Ylm[0]+0.8*Ylm[1]
plt.errorbar([x0], [y0], yerr=[Y_err_median], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)


a, b, c  = np.polyfit(c21w,delta, 2)
x = c21w
y = delta
mydata = odr.Data(x, y, wd=Ec21w, we=delta*0.+0.1)
F = odr.Model(Fdelta)
myodr = odr.ODR(mydata, F, beta0=[a,b,c])
myoutput = myodr.run()

xfit = np.linspace(-2,10, 100)
yfit = Fdelta(myoutput.beta, xfit)
plt.plot(xfit, yfit, '--', color='blue')

print myoutput.beta

plt.savefig('deviation_c21w2.eps', dpi=600)
plt.show()
