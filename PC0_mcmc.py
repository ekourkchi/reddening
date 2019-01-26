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

from matplotlib import rcParams
rcParams["font.size"] = 14
#rcParams["font.family"] = "sans-serif"
#rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

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
band1 = 'r'
band2 = 'w2'
################################################################# 

R, Input, T = getBand('ESN_HI_catal.csv', band1=band1 , band2=band2)
r_w1    = Input[1]
pc0     = Input[2]
inc     = Input[3]
################################################################# 

def lnlike(theta, inc, R, pc0):
    
    
    a,b,c,d,alpha,beta,gamma = theta
    #c,d,alpha,beta,gamma = theta
    q2 = 10**(-1.*gamma)
    
    model = log_a_b(inc, q2)*(a*pc0**3+b*pc0**2+c*pc0+d)+(alpha*pc0+beta)
    #model = log_a_b(inc, q2)*(c*pc0+d)+(alpha*pc0+beta)
    
    yerr = R*0+0.05
    inv_sigma2 = 1.0/(yerr**2) 
    return -0.5*(np.sum((R-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
    
    a,b,c,d,alpha,beta,gamma = theta
    #c,d,alpha,beta,gamma = theta

    if gamma>10. or gamma<0.8: return -np.inf 
    #if abs(alpha)>1.: return -np.inf 
    #if abs(beta)>1.: return -np.inf 


    return 0.0


def lnprob(theta, inc, R, pc0):
    
    lp = lnprior(theta)
    if not np.isfinite(lp) :
        return -np.inf    
    return lp + lnlike(theta, inc, R, pc0)


if True:
        ndim, nwalkers = 7, 200
        #ndim, nwalkers = 5, 200

        p0 = [np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(inc, R, pc0))


        #pos, prob, state = sampler.run_mcmc(p0, 50)
        #sampler.reset()
        sampler.run_mcmc(p0, 50000)


        ## removing the first 1000 samples
        samples = sampler.chain[:, 5000:, :].reshape((-1, ndim))

        
        
############################### Cleaning output

   
        T = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        
        for i in range(ndim):
            samples = data_trim(ndim, i, [T[i][0]-3*T[i][2],T[i][0]+3*T[i][1]], samples)
        
      

        a,b,c,d,alpha,beta,gamma = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],                                     axis=0)))

        
        #c,d,alpha,beta,gamma = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
 
        
        truths=[a[0],b[0],c[0],d[0],alpha[0],beta[0],gamma[0]]
        fig = corner.corner(samples, labels=["$a$","$b$", "$c$", "$d$",r"$\alpha$", r"$\beta$", r"$\gamma$"], truths=truths, truth_color='r', quantiles=[0.16, 0.84],
                    levels=(1-np.exp(-1./8),1-np.exp(-0.5),1-np.exp(-0.5*4),1-np.exp(-0.5*9)),
                    show_titles=True, fill_contours=True, plot_density=True,
                    scale_hist=False,space=0, 
                    title_kwargs={"fontsize":16}, title_fmt=".3f") 
        
        
        #truths=[c[0],d[0],alpha[0],beta[0],gamma[0]]
        #fig = corner.corner(samples, labels=["$c$", "$d$",r"$\alpha$", r"$\beta$", r"$\gamma$"], truths=truths, truth_color='r', quantiles=[0.16, 0.84],
                    #levels=(1-np.exp(-1./8),1-np.exp(-0.5),1-np.exp(-0.5*4),1-np.exp(-0.5*9)),
                    #show_titles=True, fill_contours=True, plot_density=True,
                    #scale_hist=False,space=0, 
                    #title_kwargs={"fontsize":16}, title_fmt=".3f")        
        
       
         
        
       
        print "a: ", a
        print "b: ", b
        print "c: ", c
        print "d: ", d
        print "alpha: ", alpha
        print "beta: ", beta
        print "gamma: ", gamma
   
        #print "sigma: ", sigma
        
        fig.savefig("PC0_mcmc_"+band1+".png")
        plt.show()

