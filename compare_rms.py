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
def george_params(band1='r'):
    if band1=='u': theta = [3.668874  , 6.50517701, 0.59288974, 0.16381692]
    if band1=='g': theta = [3.15024649,  5.59633583, -0.54934107,  0.10942759]
    if band1=='r': theta = [3.00712491,  5.28702113, -1.04442791, 0.09683042]
    if band1=='i': theta = [2.80674111,  4.97871648, -1.58905037,  0.09901532]
    if band1=='z': theta = [2.7634206 ,  4.64052849, -2.06768134,  0.09825892]
    if band1=='w1':theta = [2.68218476e+00,  1.02518650e+01, -4.78962522e+00,  1.00000000e-02]
    
    return theta
################################################################# 
def getRMS(inFile, band1 = 'r', band2 = 'w2'):

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
    kernel = sigma * kernels.Matern52Kernel([l1,l2], ndim=2, axes=[0, 1])
    gp = george.GP(kernel)
    gp.compute(X, yerr)
    A_grg, var_A_grg = gp.predict(R, X, return_var=True)


    indx = np.where(-4<=pc0)
    R = R[indx]
    A_mdl = A_mdl[indx]
    A_grg = A_grg[indx]
    pc0 = pc0[indx]
    inc = inc[indx]

    indx = np.where(pc0<=-2)
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

X_axis = np.asarray([1,2.5,4,5.5])
inFile = 'ESN_HI_catal.csv'

fig = py.figure(figsize=(7, 5), dpi=100)    
ax = fig.add_subplot(111)

rms_model, rms_george = getRMS(inFile, band1 = 'u', band2 = 'w2')
ax.plot(X_axis, rms_model, 'bo')
ax.plot(X_axis, rms_george, 'bo', mfc='none')
    
rms_model, rms_george = getRMS(inFile, band1 = 'g', band2 = 'w2')
ax.plot(X_axis+0.15, rms_model, 'gs')
ax.plot(X_axis+0.15, rms_george, 'gs', mfc='none')

    
rms_model, rms_george = getRMS(inFile, band1 = 'r', band2 = 'w2')
ax.plot(X_axis+0.3, rms_model, 'r^')
ax.plot(X_axis+0.3, rms_george, 'r^', mfc='none')

rms_model, rms_george = getRMS(inFile, band1 = 'i', band2 = 'w2')
ax.plot(X_axis+0.45, rms_model, 'D', color='orange')
ax.plot(X_axis+0.45, rms_george, 'D', color='orange', mfc='none')


rms_model, rms_george = getRMS(inFile, band1 = 'z', band2 = 'w2')
ax.plot(X_axis+0.6, rms_model, 'v', color='maroon')
ax.plot(X_axis+0.6, rms_george, 'v', color='maroon', mfc='none')


plt.show()

    



















