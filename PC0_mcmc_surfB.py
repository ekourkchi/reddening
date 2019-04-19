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
if len(sys.argv)<3:
    print "Please enter band1 and band2 as input ..."
    print "Exaple: python ...mcmc.py r w2"
    sys.exit()
else:
    band1 = str(sys.argv[1])
    band2 = str(sys.argv[2])
################################################################# 

AB, cov, rms, table = faceON_surfaceB('ESN_HI_catal.csv', band1 = band1, band2 = band2)
a0, b0 = AB[0], AB[1]

r_w1    = table['r_w1']
mu50    = table['mu50']
inc     = table['inc']

Er_w1 = table['Er_w1']
Einc  = table['inc_e']
Emu50 = table['Emu50']

Epc0 = Emu50
pc0  = mu50

R = r_w1-(a0*pc0+b0)
dR2 = Er_w1**2+(a0*pc0)**2
################################################################# 

def lnlike(theta, inc, R, pc0):
   
    if band1!='w1':
        d,gamma = theta
        a = 0
        b=0
        c=0
        alpha=0
        beta = 0 
    else: 
        c,d,alpha,beta,gamma = theta
        a=0 ; b=0
        
    q2 = 10**(-1.*gamma)
    
    F = log_a_b(inc, q2)
    M = F*(a*pc0**3+b*pc0**2+c*pc0+d)+(alpha*pc0+beta)
    
    dF2 = Elogab2(inc, q2, Einc)
    dM2 = dF2*(a*pc0**3+b*pc0**2+c*pc0+d)**2+(F*(3*a*pc0**2+2*b*pc0+c)*Epc0)**2+(alpha*Epc0)**2
    
    
    inv_sigma2 = 1.0/(dR2+dM2)
    return -0.5*(np.sum((R-M)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
    
    if band1!='w1':
        d,gamma = theta
        a = 0
        b=0
        c=0
        alpha=0
        beta=0
    else: 
        c,d,gamma = theta
        a=-0.002 ; b=-0.007

    if gamma>20. : return -np.inf 
    #if d<-1: return -np.inf 

    #if alpha < -0.1 or alpha>0.1:return -np.inf 
    #if beta < -0.1 or beta>0.1: return -np.inf 
    #if a>0.01 or b>0.01: return -np.inf 


    return 0.0


def lnprob(theta, inc, R, pc0):
    
    lp = lnprior(theta)
    if not np.isfinite(lp) :
        return -np.inf    
    return lp + lnlike(theta, inc, R, pc0)

npz_file = "PC0_mcmc_surfB_"+band1+"_"+band2+".npz"

if band1!='w1': 
    ndim, nwalkers = 2, 20
else: 
    ndim, nwalkers = 5, 20

if True:    ## MCMC part


        p0 = [np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(inc, R, pc0))


        #pos, prob, state = sampler.run_mcmc(p0, 50)
        #sampler.reset()
        sampler.run_mcmc(p0, 7000)


        ## removing the first 1000 samples
        samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))
        
        np.savez_compressed(npz_file, array=samples)

        

if True:    ## Ploting part
############################### Cleaning output
        
        loaded = np.load(npz_file)
        samples = loaded['array']
        

        #if band1!='w1': 
            #samples[:,1]+= a0
            #samples[:,2]+= b0
        #else:
            #samples[:,2]+= a0
            #samples[:,3]+= b0
        
        T = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        
        for i in range(ndim):
            samples = data_trim(ndim, i, [T[i][0]-3*T[i][2],T[i][0]+3*T[i][1]], samples)
        
      

        if band1!='w1': 
            d,gamma = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        else:
            c,d,alpha,beta,gamma = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        

        if band1!='w1': 
            truths=[d[0],gamma[0]]
            fig = corner.corner(samples, labels=["$C^{(0)}_{"+band1+"}$", r"$\theta_"+band1+"$"], truths=truths, truth_color='r', quantiles=[0.16, 0.84],
                    levels=(1-np.exp(-1./8),1-np.exp(-0.5),1-np.exp(-0.5*4),1-np.exp(-0.5*9)),
                    show_titles=True, fill_contours=True, plot_density=True,
                    scale_hist=False,space=0, 
                    title_kwargs={"fontsize":18}, title_fmt=".3f", bins=25) 
        
        else:
            truths=[c[0],d[0],alpha[0],beta[0],gamma[0]]
            fig = corner.corner(samples, labels=["$C^{(1)}_{"+band1+"}$", "$C^{(0)}_{"+band1+"}$",r"$\alpha)_{"+band1+"}$", r"$\beta)_{"+band1+"}$", r"$\theta)_{"+band1+"}$"], truths=truths, truth_color='r', quantiles=[0.16, 0.84],
                    levels=(1-np.exp(-1./8),1-np.exp(-0.5),1-np.exp(-0.5*4),1-np.exp(-0.5*9)),
                    show_titles=True, fill_contours=True, plot_density=True,
                    scale_hist=False,space=0, 
                    title_kwargs={"fontsize":18}, title_fmt=".3f", bins=25)        
        

        with open("PC0_mcmc_surfB_"+band1+"_"+band2+".txt", 'w') as file:  

             #if band1!='w1':
                 #file.write("a: "+str(a)+"\n")
                 #file.write("b: "+str(b)+"\n")
             #file.write("c: "+str(c)+"\n")
             file.write("d: "+str(d)+"\n")
             #file.write("alpha: "+str(alpha)+"\n")
             #file.write("beta: "+str(beta)+"\n")
             file.write("gamma: "+str(gamma)+"\n")
             
        
        fig.savefig("PC0_mcmc_surfB_"+band1+"_"+band2+".png")
        #plt.show()

