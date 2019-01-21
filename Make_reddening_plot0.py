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

if band2=='w1':
    text1 = band1+'-W1'            # example: cr-W1
    text2 = '$c21W_1$'              # example: c21w
else: 
    text1 = band1+'-W2' 
    text2 = '$c21W_2$'

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


########################################################Face-ON

## Get the initial estimations using Face-on galaxies
## AB:    a0*logWimx+b0
## Delta: alfa*X**2+beta*X+gama
AB, Delta, table = faceON(table)
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

fig = py.figure(figsize=(9, 4), dpi=100)   
fig.subplots_adjust(wspace=0.25, top=0.95, bottom=0.13, left=0.09, right=0.98)
gs = gridspec.GridSpec(1, 2) 

ax = plt.subplot(gs[0]) 
ax.plot([-1,4], [-a+b,a*4+b], 'k--')

for i in range(len(pgc)):
   if c21w[i]<1  :
       p1, = ax.plot(r_w1[i], logWimx[i], 'b.', markersize=5, alpha=1, label=r"$c21W_1 < 1$")
       #plt.errorbar(r_w1[i], logWimx[i], xerr=Er_w1[i], yerr=logWimx_e[i], color='b', alpha=0.2)
   if c21w[i]>=1 and c21w[i]< 2:
       p2, = ax.plot(r_w1[i], logWimx[i], 'g.', markersize=5, alpha=1, label=r"$1 < c21W_1 < 3$")  
       #plt.errorbar(r_w1[i], logWimx[i], xerr=Er_w1[i], yerr=logWimx_e[i], color='g', alpha=0.2)
   if c21w[i]>=2 and c21w[i]< 3:
       ax.plot(r_w1[i], logWimx[i], '.', color='green', markersize=5, alpha=1)   
       #plt.errorbar(r_w1[i], logWimx[i], xerr=Er_w1[i], yerr=logWimx_e[i], color='g', alpha=0.2)
   if c21w[i]>=3:
       p3, = ax.plot(r_w1[i], logWimx[i], 'r.', markersize=5, alpha=1, label=r"$3 < c21W_1$")   
       #plt.errorbar(r_w1[i], logWimx[i], xerr=Er_w1[i], yerr=logWimx_e[i], color='r', alpha=0.2)
       
ax.set_xlabel('$'+text1+'$', fontsize=14)
ax.set_ylabel(r'$Log( W_{mx}^i)$', fontsize=14)

rw_lim = [-2.5,1.5]
if band1=='u': rw_lim = [-0.5,3.5]
if band1=='z': rw_lim = [-3.0,1.0]
if band1=='g': rw_lim = [-2.0,2.0]

add_axis(ax,rw_lim,[1.7,3])
  
# Legend
lns = [p1, p2, p3]
ax.legend(handles=lns, loc=0, fontsize=11)

X_err_median = np.median(Er_w1[i])
Y_err_median = np.median(logWimx_e[i])
Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
x0 = 0.1*Xlm[0]+0.9*Xlm[1]
y0 = 0.2*Ylm[0]+0.8*Ylm[1]
plt.errorbar([x0], [y0], xerr=[Er_w1[i]], yerr=[Y_err_median], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)
       
#########################################################################  

ax = plt.subplot(gs[1])

delta = r_w1-(a0*logWimx+b0)
ax.plot(c21w, delta, '.', color='black', markersize=3, alpha=0.5)
#ax.errorbar(c21w, delta, xerr=Ec21w, yerr=delta*0.+0.1, color='k', fmt='.', alpha=0.2)

add_axis(ax,[-2,8],[-1.5,1.5])

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
ax.set_xlabel(text2, fontsize=14) 


ax.plot([-2,8], [0,0], 'k:')

X_err_median = np.median(Ec21w)
Y_err_median = np.median(delta*0.+0.1)
Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
x0 = 0.9*Xlm[0]+0.1*Xlm[1]
y0 = 0.2*Ylm[0]+0.8*Ylm[1]
plt.errorbar([x0], [y0], yerr=[Y_err_median], color='k', fmt='o', alpha=0.7, capsize=3, markersize=5)

xfit = np.linspace(-2,10, 100)
yfit = Fdelta(Delta, xfit)
plt.plot(xfit, yfit, '--', color='blue')

#########################################################################  


#plt.plot(r_w1, c21w, 'b.')
    
#plt.xlim([-2,4])
#plt.ylim([-2,8])   
#plt.xlabel('cr-w1')
#plt.ylabel(r'$c21w$')     
#########################################################################  
#########################################################################  

#plt.plot(c21w, logWimx, 'g.', markersize=2, alpha=0.3)
    
#plt.xlim([-2,8])
#plt.ylim([1.7,3])  

#plt.ylabel(r'$Log( W_{mx}^i)$')
#plt.xlabel(r'$c21w$') 
######################################################################### 



plt.show()



#mean = myoutput.beta
#sd = myoutput.sd_beta
#cov =  myoutput.cov_beta
#print sd
#print cov
#N = 5000
#sample = np.random.multivariate_normal(mean, cov, N)
#alfa = np.zeros(N)
#beta = np.zeros(N)
#gama = np.zeros(N)
#for i in range(N):
    #alfa[i]=sample[i][0]
    #beta[i]=sample[i][1]
    #gama[i]=sample[i][2]
    
#print np.mean(alfa), np.mean(beta), np.mean(gama)

#y_std = np.ones_like(xfit)
#y_med = np.ones_like(xfit)
#Ys = np.zeros(N)
#for i in range(len(xfit)):
    
    #for j in range(N):
        #B = [alfa[j], beta[j], gama[j]]
        #f = Fdelta(B, [xfit[i]])
        #Ys[j] = f[0]
    
    #y_std[i] = np.std(Ys)
    #y_med[i] = np.mean(Ys)


#plt.plot(xfit, y_med, '--', color='red')

#plt.fill(np.concatenate([xfit, xfit[::-1]]),
         #np.concatenate([y_med - 1.000 * y_std,   # 1.9600
                        #(y_med + 1.000 * y_std)[::-1]]),
         #alpha=.3, fc='b', ec='None')
