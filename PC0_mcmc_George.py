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
    R = Y_train
    N = len(pc0)

    def step(theta):
        
        alpha = theta[0]
        beta = theta[1]
        l1 = np.exp(theta[2])
        l2 = np.exp(theta[3])
        sigma = np.exp(theta[4])
        M = alpha*pc0 + beta
        dM = alpha*Epc0    
        yerr = np.diagonal(np.sqrt(noise2+(alpha*Epc0)**2*np.eye(N)))
        
        kernel = sigma * kernels.ExpSquaredKernel([l1,l2], ndim=2, axes=[0, 1])
        gp = george.GP(kernel)
        gp.compute(X_train, yerr)
        
        #sys.exit()
        # Compute determinant via Cholesky decomposition
        return -gp.lnlikelihood(R - M)
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
    l1 = np.exp(theta[2])
    l2 = np.exp(theta[3])
    sigma = np.exp(theta[4])
    M = alpha*pc0 + beta
    dM = alpha*Epc0    
    Delta = R-M
    yerr = np.diagonal(np.sqrt(noise2+(alpha*Epc0)**2*np.eye(N)))
    
    kernel = sigma * kernels.ExpSquaredKernel([l1,l2], ndim=2, axes=[0, 1])
    gp = george.GP(kernel)
    gp.compute(X, yerr)
    
    return gp.lnlikelihood(R - M)

################################################################# 

def lnprior(theta):
    
    alpha = theta[0]
    beta = theta[1]
    l1 = theta[2]
    l2 = theta[3]
    sigma = theta[4]

    if alpha>0.2 or alpha<-0.05: return -np.inf 
    if beta>0.4 or beta<-0.1: return -np.inf 
    if l1>-6: return -np.inf 
    if l2<-6: return -np.inf 
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
r_w1    = Input[1]
pc0     = Input[2]
inc     = np.cos(Input[3]*np.pi/180.)
#R = R

AB = T[2] ; table = T[5]
a0, b0 = AB[0], AB[1]

Er_w1 = table['Er_w1']
Epc0  = table['Epc0']
Einc  = table['inc_e']

N = len(pc0)
dR2 = Er_w1**2+(a0*Epc0)**2
noise2 = dR2*np.eye(N)

X = np.ones(shape = (2,N))
X[0] = pc0
X[1] = inc
X = X.T
y = R

start_time = time.time()

# Maximum Likelihood
result = minimize(nll_fn2(X, y, Epc0, noise2), [0.05, 0.3, 1, 1, 1], 
               bounds=((-0.5, 0.5), (-1, 1), (None, None), (None, None), (None, None)),
               method='L-BFGS-B')
print result




def esnRand():
    
    alpha = -0.05+0.25*np.random.randn()
    beta = -0.1+0.5*np.random.randn()
    l1 = -6-2*np.random.randn()
    l2 = -6+2*np.random.randn()
    sigma = -3+np.random.randn()
    return np.asarray([alpha, beta, l1, l2, sigma])

npz_file = "PC0_mcmc_"+band1+"_"+band2+".George.npz"

if False:    ## MCMC part

        ndim, nwalkers = 5, 16
        p0 = [esnRand() for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(inc, R, pc0, Epc0, noise2))

        sampler.run_mcmc(p0, 2000)
        samples = sampler.chain[:, 200:, :].reshape((-1, ndim))
        
        #samples[:,2] = np.abs(samples[:,2])
        #samples[:,3] = np.abs(samples[:,3])
        #samples[:,4] = np.abs(samples[:,4])
        np.savez_compressed(npz_file, array=samples)
        
        samples[:,0]+= a0
        samples[:,1]+= b0
        
        T = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        
        for i in range(ndim):
            samples = data_trim(ndim, i, [T[i][0]-3*T[i][2],T[i][0]+3*T[i][1]], samples)
        
      
        alpha,beta,l1,l2,sigma = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        

        truths=[alpha[0],beta[0],l1[0],l2[0],sigma[0]]
        fig = corner.corner(samples, labels=[r"$\alpha_{"+band1+"}$", r"$\beta_{"+band1+"}$", "$ln(l^2_1)$", "$ln(l^2_2)$", r"$ln(\sigma^2_f)$"], truths=truths, truth_color='r', quantiles=[0.16, 0.84],
                    levels=(1-np.exp(-1./8),1-np.exp(-0.5),1-np.exp(-0.5*4),1-np.exp(-0.5*9)),
                    show_titles=True, fill_contours=True, plot_density=True,
                    scale_hist=False,space=0, 
                    title_kwargs={"fontsize":16}, title_fmt=".3f")        
        
        
        fig.savefig("PC0_mcmc_"+band1+"_"+band2+".George.png")
        #plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
