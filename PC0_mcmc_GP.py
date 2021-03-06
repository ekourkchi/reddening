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
import corner
import emcee
import scipy.optimize as op
from scipy.linalg import cholesky, inv,det
from scipy.optimize import minimize
#from numpy.linalg import inv, det

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
    R = Y_train
    N = len(pc0)

    def step(theta):
        
        #print theta
        alpha = theta[0]
        beta = theta[1]
        M = alpha*pc0 + beta
        dM = alpha*Epc0
        Delta = R-M
        noise2_ = noise2+(alpha*Epc0)**2*np.eye(N)
        
        K = kernel(X_train, X_train, [theta[2], theta[3]], sigma_f=theta[4])+ noise2_
        
        #print np.where(K<0)
        
        #print det([[1,2],[3,4]])
        #print np.sum(np.diagonal(cholesky(K)))
        
        #sys.exit()
        # Compute determinant via Cholesky decomposition
        return np.sum(np.log(np.diagonal(cholesky(K)))) + \
               0.5 * Delta.T.dot(inv(K).dot(Delta)) + \
               0.5 * len(X_train) * np.log(2*np.pi)
    return step

################################################################# 

def lnlike(theta, inc, R, pc0, Epc0, noise2):
    
    N = len(pc0)
    X = np.ones(shape = (2,N))
    X[0] = pc0
    X[1] = inc
    X = X.T

    alpha = theta[0]
    beta = theta[1]
    M = alpha*pc0 + beta
    dM = alpha*Epc0    
    Delta = R-M
    noise2_ = noise2+(alpha*Epc0)**2*np.eye(N)
    
    K = kernel(X, X, [theta[2], theta[3]], sigma_f=theta[4])+ noise2_
    
    nll_fn2 = np.sum(np.log(np.diagonal(cholesky(K)))) + \
               0.5 * Delta.T.dot(inv(K).dot(Delta)) + \
               0.5 * len(X) * np.log(2*np.pi)
    
    return -nll_fn2

################################################################# 

def lnprior(theta):
    
    alpha = theta[0]
    beta = theta[1]
    l1 = theta[2]
    l2 = theta[3]
    sigma = theta[4]

    if alpha>0.2 or alpha<-0.05: return -np.inf 
    if beta>0.4 or beta<-0.1: return -np.inf 
    #if l1>1. or l1<-1: return -np.inf 
    #if l2>1. or l2<-1: return -np.inf 
    #if sigma>1. or sigma<-1: return -np.inf 

    return 0.0

################################################################# 
def lnprob(theta, inc, R, pc0, Epc0, noise2):
    
    lp = lnprior(theta)
    if not np.isfinite(lp) :
        return -np.inf    
    return lp + lnlike(theta, inc, R, pc0, Epc0, noise2)

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

R, Input, T = getBand('ESN_HI_catal.csv', band1=band1 , band2=band2)
r_w1    = Input[1][0:1500]
pc0     = Input[2][0:1500]
inc     = np.cos(Input[3][0:1500]*np.pi/180.)
R = R[0:1500]

AB = T[2] ; table = T[5]
a0, b0 = AB[0], AB[1]

Er_w1 = table['Er_w1'][0:1500]
Epc0  = table['Epc0'][0:1500]
Einc  = table['inc_e'][0:1500]

N = len(pc0)
dR2 = Er_w1**2+(a0*Epc0)**2
noise2 = dR2*np.eye(N)

X = np.ones(shape = (2,N))
X[0] = pc0
X[1] = inc
X = X.T
y = R

## Maximum Likelihood
result = minimize(nll_fn2(X, y, Epc0, noise2), [0.05, 0.3, 1, 1, 1], 
               bounds=((-0.05, 0.2), (-0.1, 0.4), (None, None), (None, None), (None, None)),
               method='L-BFGS-B')
print result



if False:    ## MCMC part

        ndim, nwalkers = 5, 16
        p0 = [0.01*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(inc, R, pc0, Epc0, noise2))

        sampler.run_mcmc(p0, 2000)
        samples = sampler.chain[:, 200:, :].reshape((-1, ndim))
        
        samples[:,2] = np.abs(samples[:,2])
        samples[:,3] = np.abs(samples[:,3])
        samples[:,4] = np.abs(samples[:,4])

        
        T = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        
        for i in range(ndim):
            samples = data_trim(ndim, i, [T[i][0]-3*T[i][2],T[i][0]+3*T[i][1]], samples)
        
      
        alpha,beta,l1,l2,sigma = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        

        truths=[alpha[0],beta[0],l1[0],l2[0],sigma[0]]
        fig = corner.corner(samples, labels=[r"$\alpha_{"+band1+"}$", r"$\beta_{"+band1+"}$", "l1", "l2", r"$\sigma_f$"], truths=truths, truth_color='r', quantiles=[0.16, 0.84],
                    levels=(1-np.exp(-1./8),1-np.exp(-0.5),1-np.exp(-0.5*4),1-np.exp(-0.5*9)),
                    show_titles=True, fill_contours=True, plot_density=True,
                    scale_hist=False,space=0, 
                    title_kwargs={"fontsize":18}, title_fmt=".3f")        
        

        plt.show()

