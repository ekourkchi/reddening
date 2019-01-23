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

from redTools import *
from Kcorrect import *

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


################################################################# 

R, Input, T = getBand('ESN_HI_catal.csv', band1 = 'w1', band2 = 'w2')
r_w1    = Input[1]
pc0     = Input[2]
inc     = Input[3]
################################################################# 

def lnlike(theta, inc, R, pc0):
    
    
    a,b,c,d,alpha,beta,gamma = theta
    q2 = 10**(-1.*gamma)
    
    model = log_a_b(inc, q2)*(a*pc0**3+b*pc0**2+c*pc0+d)+(alpha*pc0+beta)
    
    yerr = R*0+0.05
    inv_sigma2 = 1.0/(yerr**2) 
    return -0.5*(np.sum((R-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
    
    a,b,c,d,alpha,beta,gamma = theta

    if gamma>10. or gamma<1.: return -np.inf 
    #if abs(alpha)>1.: return -np.inf 
    #if abs(beta)>1.: return -np.inf 


    return 0.0


def lnprob(theta, inc, R, pc0):
    
    lp = lnprior(theta)
    if not np.isfinite(lp) :
        return -np.inf    
    return lp + lnlike(theta, inc, R, pc0)


if True:
        ndim, nwalkers = 7, 50

        p0 = [np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(inc, R, pc0))


        #pos, prob, state = sampler.run_mcmc(p0, 50)
        #sampler.reset()
        sampler.run_mcmc(p0, 20000)


        ## removing the first 1000 samples
        samples = sampler.chain[:, 2000:, :].reshape((-1, ndim))

        
        
############################### Cleaning output


        #samples = data_trim(ndim, 0, [-0.05,0.05], samples)  # a
        #samples = data_trim(ndim, 1, [-0.2,0], samples)    # b  
        #samples = data_trim(ndim, 2, [-1.5,1.5], samples)     # c
        #samples = data_trim(ndim, 3, [-2,-2], samples)        # d
        #samples = data_trim(ndim, 4, [-1.5,1.5], samples)     # e
        #samples = data_trim(ndim, 5, [2.4,3.1], samples)        # alfa
        #samples = data_trim(ndim, 6, [-0.03,0.03], samples)    # A
        #samples = data_trim(ndim, 7, [-1.5,1.5], samples)     # B
        #samples = data_trim(ndim, 8, [-1.5,1.5], samples)    # C     
        #samples = data_trim(ndim, 9, [-2.5,2.5], samples)    # D     
        #samples = data_trim(ndim, 10, [1.5,1.5], samples)     # E     
        T = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        
        for i in range(ndim):
            #print i, T[i], [T[i][0]-T[i][2],T[i][0]+T[i][1]]
            samples = data_trim(ndim, i, [T[i][0]-3*T[i][2],T[i][0]+3*T[i][1]], samples)
        
      

        a,b,c,d,alpha,beta,gamma = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],                                     axis=0)))
                                    
        truths=[a[0],b[0],c[0],d[0],alpha[0],beta[0],gamma[0]]
        fig = corner.corner(samples, labels=["$a$","$b$", "$c$", "$d$","alpha", "beta", "gamma"], truths=truths, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}, title_fmt=".3f")
       
        print "a: ", a
        print "b: ", b
        print "c: ", c
        print "d: ", d
        print "alpha: ", alpha
        print "beta: ", beta
        print "gamma: ", gamma
   
        #print "sigma: ", sigma
        
        fig.savefig("corner_plot_01.png")
        plt.show()

