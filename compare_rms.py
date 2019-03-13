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
def getRMS(inFile, pc0_lim=[0,2], band1 = 'r', band2 = 'w2'):

    _, Input, T = getBand('ESN_HI_catal.csv', band1=band1 , band2=band2)
    r_w1    = Input[1]
    pc0     = Input[2]
    inc     = Input[3]
    table = T[5]
    Er_w1 = table['Er_w1']
    Epc0  = table['Epc0']
    Einc  = table['inc_e']


    a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1=band1, band2=band2)
    q2 = 10**(-1.*theta)

    R = r_w1-(alpha*pc0+beta)


    A_mdl = log_a_b(inc, q2)*(a*pc0**3+b*pc0**2+c*pc0+d)


    N = len(pc0)
    dR2 = Er_w1**2+(alpha*Epc0)**2+(Ealpha*pc0)**2
    noise2 = dR2*np.eye(N)
    X = np.ones(shape = (2,N))
    X[0] = pc0
    X[1] = inc
    X = X.T
    theta = george_params(band1=band1)
    l1 = np.exp(theta[0])
    l2 = np.exp(theta[1])
    sigma = np.exp(theta[2])
    yerr = np.diagonal(np.sqrt(noise2))+theta[3]
    ####kernel = sigma * kernels.Matern52Kernel([l1,l2], ndim=2, axes=[0, 1])
    kernel = sigma * kernels.ExpSquaredKernel([l1,l2], ndim=2, axes=[0, 1])
    
    gp = george.GP(kernel)
    gp.compute(X, yerr)
    A_grg, var_A_grg = gp.predict(R, X, return_var=True)


    indx = np.where(pc0_lim[0]<=pc0)
    R = R[indx]
    A_mdl = A_mdl[indx]
    A_grg = A_grg[indx]
    pc0 = pc0[indx]
    inc = inc[indx]

    indx = np.where(pc0<pc0_lim[1])
    R = R[indx]
    A_mdl = A_mdl[indx]
    A_grg = A_grg[indx]
    pc0 = pc0[indx]
    inc = inc[indx]


    rms_model = []
    rms_george = []
    for i in range(50,90,10):
        
        print i, i+10, ' .......'
        indx = np.where(i<inc)
        R_ = R[indx]
        A_mdl_ = A_mdl[indx]
        A_grg_ = A_grg[indx]
        pc0_ = pc0[indx]
        inc_ = inc[indx]
        indx = np.where(inc_<=i+10)
        R_ = R_[indx]
        A_mdl_ = A_mdl_[indx]
        A_grg_ = A_grg_[indx]
        pc0_ = pc0_[indx]
        inc_ = inc_[indx]  
        
        
        rms_model.append(np.sqrt(np.median((R_-A_mdl_)**2)))
        rms_george.append(np.sqrt(np.median((R_-A_grg_)**2)))

    return rms_model, rms_george
################################################################# 
################################################################# 
def plotRMS(ax, inFile, pc0_lim=[0,2]):
    
    X_axis = np.asarray([1,2.5,4,5.5])

    rms_model, rms_george = getRMS(inFile, band1 = 'u', band2 = 'w2', pc0_lim=pc0_lim)
    p1, = ax.plot(X_axis-0.5, rms_model, 'bo', label='u')
    ax.plot(X_axis-0.5, rms_george, 'bo', mfc='none', label='u')
        
    rms_model, rms_george = getRMS(inFile, band1 = 'g', band2 = 'w2', pc0_lim=pc0_lim)
    p2, = ax.plot(X_axis-0.25, rms_model, 'gs', label='g')
    ax.plot(X_axis-0.25, rms_george, 'gs', mfc='none', label='g')

    rms_model, rms_george = getRMS(inFile, band1 = 'r', band2 = 'w2', pc0_lim=pc0_lim)
    p3, = ax.plot(X_axis, rms_model, 'r^', label='r')
    ax.plot(X_axis, rms_george, 'r^', mfc='none', label='r')

    rms_model, rms_george = getRMS(inFile, band1 = 'i', band2 = 'w2', pc0_lim=pc0_lim)
    p4, = ax.plot(X_axis+0.25, rms_model, 'D', color='orange', label='i')
    ax.plot(X_axis+0.25, rms_george, 'D', color='orange', mfc='none', label='i')

    rms_model, rms_george = getRMS(inFile, band1 = 'z', band2 = 'w2', pc0_lim=pc0_lim)
    p5, = ax.plot(X_axis+0.5, rms_model, 'v', color='maroon', label='z')
    ax.plot(X_axis+0.5, rms_george, 'v', color='maroon', mfc='none', label='z')


    ax.plot([1.7,1.7],[0,1],'k:')
    ax.plot([3.2,3.2],[0,0.39],'k:')
    ax.plot([4.7,4.7],[0,1],'k:')

    ax.set_xlim(0.3,6.2)
    ax.set_ylim(0.03,0.49)
    
    ax.minorticks_on()
    ax.tick_params(which='major', length=7, width=1.5)
    ax.tick_params(which='minor', length=4, color='#000033', width=1.0)    

    ## additional Y-axis (on the right)
    y_ax = ax.twinx()
    y_ax.set_ylim(0.03,0.49)
    y_ax.set_yticklabels([])
    y_ax.minorticks_on()
    y_ax.tick_params(which='major', length=7, width=1.5, direction='in')
    y_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    
    ## additional X-axis (on the top)
    ##x_ax = ax.twiny()
    ##x_ax.set_xlim(0.3,6.2)
    ##x_ax.set_xticklabels([])
    ##x_ax.tick_params(which='major', length=7, width=1.5, direction='in')
    ##x_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    
    
    ##plt.xticks([1,2.5,4,5.5])
    ##plt.setp(ax.get_xticklabels(), visible=False)
    
    ax.text(2.2,0.4, r''+"%.0f" % (pc0_lim[0])+'$~ \leq P_1 < ~$'+"%.0f" % (pc0_lim[1]), fontsize=14, color='black', weight='bold')
    
    ax.set_ylabel('RMS [mag]', fontsize=14, labelpad=10)
    
    if pc0_lim[0]==2:
        lns = [p1, p2, p3, p4, p5]
        ax.legend(handles=lns, fontsize=14, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    

################################################################# 

inFile = 'ESN_HI_catal.csv'
fig = py.figure(figsize=(4.5, 9), dpi=100)   
fig.subplots_adjust(hspace=0, top=0.92, bottom=0.08, left=0.20, right=0.95)
gs = gridspec.GridSpec(4,1) 
p = 0


for i in [2,0,-2,-4]:
    ax = plt.subplot(gs[p]) ; p+=1 
    plotRMS(ax, inFile, pc0_lim=[i,i+2])


plt.xticks([1,2.5,4,5.5], ('50-60','60-70','70-80','80-90'), rotation=45)
ax.tick_params(axis='x', which='minor', bottom=False)

ax.set_xlabel('Inclination range [deg]', fontsize=14, labelpad=10)


fig.savefig("compare_rms.png")
fig.savefig("compare_rms.eps")
plt.show()

    



















