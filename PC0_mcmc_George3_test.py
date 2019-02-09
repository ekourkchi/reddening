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
def kernel(X1, X2, l, sigma_f=1.0):
    
    l = np.asarray(l)
    L1 = np.zeros_like(X1)+l
    L2 = np.zeros_like(X2)+l
    
    X1_ = X1/L1
    X2_ = X2/L2
    
    ''' Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2. Args: X1: Array of m points (m x d). X2: Array of n points (n x d). Returns: Covariance matrix (m x n). '''
    sqdist = np.sum(X1_**2, 1).reshape(-1, 1) + np.sum(X2_**2, 1) - 2 * np.dot(X1_, X2_.T)
    return sigma_f**2 * np.exp(-0.5 * sqdist)
################################################################# 

def posterior_predictive2(X_s, X_train, Y_train, l, sigma_f=1.0, sigma_y=1e-8):

    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)
    
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, cov_s

################################################################# 

def nll_fn2(X_train, Y_train, Epc0, noise2):
    
    X = X_train.T
    pc0 = X[0]
    D = Y_train
    N = len(pc0)

    def step(theta):
        
        l1 = np.exp(theta[0])
        l2 = np.exp(theta[1])
        sigma = np.exp(theta[2])
        yerr = np.diagonal(np.sqrt(noise2))+theta[3]
        
        ##kernel = sigma * kernels.ExpSquaredKernel([l1,l2], ndim=2, axes=[0, 1])
        kernel = sigma * kernels.Matern52Kernel([l1,l2], ndim=2, axes=[0, 1])

        gp = george.GP(kernel)
        gp.compute(X_train, yerr)
        
        #sys.exit()
        # Compute determinant via Cholesky decomposition
        return -gp.lnlikelihood(D)
    return step

################################################################# 


################################################################# 
def data_trim(ndim, ind, limits, samples):
    
    
    for i in range(ndim):
       samples[:,i] = np.asarray(samples[:,i])
    index = np.where(samples[:,ind]<limits[1])
    pp = len(index[0])
    AA = np.zeros([pp,ndim])
    for i in range(ndim):
       AA[:,i] = samples[:,i][index[0]]    
    samples = AA


    index = np.where(samples[:,ind]>limits[0])
    pp = len(index[0])
    AA = np.zeros([pp,ndim])
    for i in range(ndim):
       AA[:,i] = samples[:,i][index[0]]    
    samples = AA

    return samples
################################################################# 
if len(sys.argv)<3:
    print "Please enter band1 and band2 as input ..."
    print "Exaple: python ...mcmc.py r w2"
    sys.exit()
else:
    band1 = str(sys.argv[1])
    band2 = str(sys.argv[2])
################################################################# 

_, Input, T = getBand('ESN_HI_catal.csv', band1=band1 , band2=band2)
r_w1    = Input[1]
pc0     = Input[2]
inc     = Input[3]

AB = T[2] ; table = T[5]
a0, b0 = AB[0], AB[1]

Er_w1 = table['Er_w1']
Epc0  = table['Epc0']
Einc  = table['inc_e']


a,b,c,d, alpha, beta, gamma, Ealpha, Ebeta = getReddening_params(band1=band1, band2=band2)

q2 = 10**(-1.*gamma)
F = log_a_b(inc, q2)
dF2 = Elogab2(inc, q2, Einc)
A = F*(a*pc0**3+b*pc0**2+c*pc0+d)
dA = np.sqrt(dF2*(a*pc0**3+b*pc0**2+c*pc0+d)**2+(F*(3*a*pc0**2+2*b*pc0+c)*Epc0)**2)


R = r_w1-(alpha*pc0+beta)

D = R - A

N = len(pc0)
dR2 = Er_w1**2+(alpha*Epc0)**2+(Ealpha*pc0)**2
noise2 = (dR2+dA**2)*np.eye(N)

X = np.ones(shape = (2,N))
X[0] = pc0
X[1] = inc
X = X.T
y = D

start_time = time.time()

### Maximum Likelihood
result = minimize(nll_fn2(X, y, Epc0, noise2), [1, 1, 1, 0.1], 
               bounds=((None, None), (None, None), (None, None), (0.01, 0.5)),
               method='L-BFGS-B')
print result

