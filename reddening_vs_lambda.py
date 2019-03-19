#!/usr/bin/python
# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import time
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
import corner
import emcee
import scipy.optimize as op
from scipy.linalg import cholesky, inv,det
from scipy.optimize import minimize
#from numpy.linalg import inv, det
import george
from george import kernels

from redTools import *
from Kcorrect import *

from matplotlib import rcParams
rcParams["font.size"] = 14
#rcParams["font.family"] = "sans-serif"
#rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

################################################################# 
################################################################# 
def plotMe(ax, pc0, band2 = 'w2'):

    
    band1 = ['u','g','r','i','z','w1']
    
    C0 = 2.99792458     #  speed of light = C0*1.E8
    lambda_u = 0.3551  # https://classic.sdss.org/dr7/instruments/imager/index.html
    lambda_g = 0.4686  # micron
    lambda_r = 0.6165
    lambda_i = 0.7481
    lambda_z = 0.8931
    lambda_w1 = 3.4
    lambda_w2 = 4.6    
    wavelengths = [lambda_u, lambda_g, lambda_r, lambda_i, lambda_z, lambda_w1]
    
    
    ####################
    A_lst = []
    inc = 45
    for band in band1:
        
        a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1=band, band2=band2)
        q2 = 10**(-1.*theta)
        A_mdl = log_a_b(inc, q2)*(a*pc0**3+b*pc0**2+c*pc0+d)
        A_lst.append(A_mdl)
    p1, = ax.plot(wavelengths, A_lst, 'o',  markersize=4, label=r'$i=45^o$')
    ####################
    ####################
    A_lst = []
    inc = 60
    for band in band1:
        
        a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1=band, band2=band2)
        q2 = 10**(-1.*theta)
        A_mdl = log_a_b(inc, q2)*(a*pc0**3+b*pc0**2+c*pc0+d)
        A_lst.append(A_mdl)
    p2, = ax.plot(wavelengths, A_lst, 'D',  markersize=4, label=r'$i=60^o$')
    ####################    
    ####################
    A_lst = []
    inc = 75
    for band in band1:
        
        a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1=band, band2=band2)
        q2 = 10**(-1.*theta)
        A_mdl = log_a_b(inc, q2)*(a*pc0**3+b*pc0**2+c*pc0+d)
        A_lst.append(A_mdl)
    p3, = ax.plot(wavelengths, A_lst, '^',  markersize=4, label=r'$i=75^o$')
    ####################      
    ####################
    A_lst = []
    inc = 90
    for band in band1:
        
        a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1=band, band2=band2)
        q2 = 10**(-1.*theta)
        A_mdl = log_a_b(inc, q2)*(a*pc0**3+b*pc0**2+c*pc0+d)
        A_lst.append(A_mdl)
    p4, = ax.plot(wavelengths, A_lst, 's',  markersize=4, label=r'$i=90^o$')
    ####################    
    
    
    
    ax.set_xlim(0.2,5)
    ax.set_xscale('log')
    ax.set_ylim(-0.1,1.9)    
    
    if pc0==-2:
        plt.xticks([0.5,1,2,3,4], ('0.5','1','2','3','4'))
  
    

    ax.plot([0.2,5],[0,0],'k:')

    ax.minorticks_on()
    ax.tick_params(which='major', length=7, width=1.5)
    ax.tick_params(which='minor', length=4, color='#000033', width=1.0)    

    ## additional Y-axis (on the right)
    y_ax = ax.twinx()
    y_ax.set_ylim(-0.1,1.9)
    y_ax.set_yticklabels([])
    y_ax.minorticks_on()
    y_ax.tick_params(which='major', length=7, width=1.5, direction='in')
    y_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    
    ###additional X-axis (on the top)
    x_ax = ax.twiny()
    x_ax.set_xlim(0.2,5)
    x_ax.set_xscale('log')
    x_ax.set_xticklabels([])
    x_ax.tick_params(which='major', length=7, width=1.5, direction='in')
    x_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    
    
    ##plt.xticks([1,2.5,4,5.5])
    ##plt.setp(ax.get_xticklabels(), visible=False)
    
    ax.text(0.9,1.5, r"$P_1 =$"+"%.0f" % (pc0), fontsize=13, color='black', weight='bold')
    
    if pc0==0:
        ax.set_ylabel(r'$A^{(i)}_{W2}$'+' [mag]', fontsize=14, labelpad=10)
    
    if pc0==2:
        lns = [p1, p2, p3, p4]
        ax.legend(handles=lns, fontsize=12, loc=0)
        
    if pc0==2:
        # Set scond x-axis
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xticks([])
        ax2.set_xticklabels([])
        ax2.minorticks_off()
        ax.minorticks_on()


        # Decide the ticklabel position in the new x-axis,
        # then convert them to the position in the old x-axis
        newlabel = ['u','g','r','i','z','W1']
        newpos   = wavelengths
        ax2.set_xticks(newpos)
        ax2.set_xticklabels(newlabel)
        
        ax2.xaxis.set_ticks_position('top') # set the position of the second x-axis to top
        ax2.xaxis.set_label_position('top') 
        ax2.spines['top'].set_position(('outward', 10))
        ax2.set_xlabel('Band')
        ax2.set_xlim(ax.get_xlim())
    

################################################################# 

fig = py.figure(figsize=(4.5, 12), dpi=100)   
fig.subplots_adjust(hspace=0, top=0.92, bottom=0.08, left=0.20, right=0.95)
gs = gridspec.GridSpec(5,1) 
p = 0


for i in [2,1.,0,-1.,-2]:
    ax = plt.subplot(gs[p]) ; p+=1 
    plotMe(ax, pc0=i)


#plt.xticks([1,2,3,4], ('1','2','3','4'))
#ax.tick_params(axis='x', which='minor', bottom=False)

ax.set_xlabel(r'$\lambda ~~ [\mu m]$', fontsize=14, labelpad=10)


fig.savefig("reddening_lambda.png")
fig.savefig("reddening_lambda.eps")
plt.show()

    



















