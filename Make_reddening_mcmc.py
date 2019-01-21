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

inFile  = 'ESN_HI_catal.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

table = extinctionCorrect(table)
table = Kcorrection(table)

band1 = 'u'
band2 = 'w1'

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

index, = np.where(table['r_w1']>-2)
table = trim(table, index)
########################################################Face-ON

## Get the initial estimations using Face-on galaxies
## AB:    a0*logWimx+b0
## Delta: alfa*X**2+beta*X+gama
## table0: table of face-on galaxies
AB, Delta, table0 = faceON(table)

########################################################Inclined
### Inclined
if True:
    index, = np.where(table['Sqlt']>3)
    table1 = trim(table, index)

    index, = np.where(table1['Wqlt']>3)
    table1 = trim(table1, index)

    index, = np.where(table1['flag']==0)
    table1 = trim(table1, index)


#########################################################################  
pgc = table1['pgc']
logWimx = table1['logWimx']
logWimx_e = table1['logWimx_e']
inc = table1['inc']
r_w1 = table1['r_w1']
c21w = table1['c21w'] 
Er_w1 = table1['Er_w1']
Ec21w = table1['Ec21w']

R = Reddening(r_w1, logWimx, c21w, AB, Delta)

a0 = AB[0]  
alfa = Delta[0]
beta = Delta[1]

SigmaR2 = np.ones_like(R)
for i in range(len(R)):
    
    if c21w[i]>-beta/2./alfa:
        SigmaR2[i] = 0.1**2+((2.*alfa*c21w[i]+beta)**2)*Ec21w[i]**2+(a0*logWimx_e[i])**2
    else:
        SigmaR2[i] = 0.1**2+(a0*logWimx_e[i])**2

print "SimgaR2 median: ", np.median(SigmaR2)
#########################################################################  

def lnlike(theta, inc, R, logWimx, c21w):
    
    
    a,b,c,d,e,alfa,A,B,C,D,E = theta
    #D = 0
    q2 = 10**(-1.*alfa)
    
    #model = log_a_b(inc, q2)*(c*logWimx*c21w+d*logWimx+e*c21w+f)+(A*logWimx+B*c21w+C+D*c21w**2)
    
    model = log_a_b(inc, q2)*(a*c21w**2+b*logWimx*c21w+c*c21w+d*logWimx+e)+(A*c21w**2+B*logWimx*c21w+C*c21w+D*logWimx+E)
    
    
    inv_sigma2 = 1./SigmaR2
    return -0.5*(np.sum((R-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def lnprior(theta):
    
    a,b,c,d,e,alfa,A,B,C,D,E = theta

    if alfa>10. or alfa<1.:
      return -np.inf 

    return 0.0


def lnprob(theta, inc, R, logWimx, c21w):
    
    lp = lnprior(theta)
    if not np.isfinite(lp) :
        return -np.inf    
    return lp + lnlike(theta, inc, R, logWimx, c21w)

#########################################################################  
##nll = lambda *args: -lnlike(*args)
##result = op.minimize(nll, [0.3], args=(inc, R))
##print result["x"]


if True:
        inc__ = np.arange(45,90,0.1)

        R__ = log_a_b(inc__, 0.04)
        plt.plot(inc__, R__, '-', color='k')
        
       
        R__ = log_a_b(inc__, 0.02)
        plt.plot(inc__, R__, '-', color='b')        
        
        R__ = log_a_b(inc__, 0.004)
        plt.plot(inc__, R__, '-', color='g')         
        
        R__ = log_a_b(inc__, 0.002)
        plt.plot(inc__, R__, '-', color='r') 
        
        R__ = log_a_b(inc__, 0.001)
        plt.plot(inc__, R__, '-', color='maroon') 
        
        plt.show()


if False:
        ndim, nwalkers = 11, 300

        p0 = [np.random.randn(ndim) for i in range(nwalkers)]

        #Rerr = 0.1*R/R
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(inc, R, logWimx, c21w))


        #pos, prob, state = sampler.run_mcmc(p0, 50)
        #sampler.reset()
        sampler.run_mcmc(p0, 10000)


        #pos, prob, state = sampler.run_mcmc(p0, 100)
        #sampler.reset()
        #sampler.run_mcmc(pos, 1000)


        samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))
        #samples[:, ndim-1] = np.exp(samples[:, ndim-1])
        
        #samples = sampler.chain
        #samples[:, :, i]   # i: dimension number
        
        
############################### Cleaning output


        samples = data_trim(ndim, 0, [-0.05,0.05], samples)  # a
        samples = data_trim(ndim, 1, [-0.2,0], samples)    # b  
        #samples = data_trim(ndim, 2, [-1.5,1.5], samples)     # c
        #samples = data_trim(ndim, 3, [-2,-2], samples)        # d
        #samples = data_trim(ndim, 4, [-1.5,1.5], samples)     # e
        samples = data_trim(ndim, 5, [2.4,3.1], samples)        # alfa
        samples = data_trim(ndim, 6, [-0.03,0.03], samples)    # A
        #samples = data_trim(ndim, 7, [-1.5,1.5], samples)     # B
        #samples = data_trim(ndim, 8, [-1.5,1.5], samples)    # C     
        #samples = data_trim(ndim, 9, [-2.5,2.5], samples)    # D     
        #samples = data_trim(ndim, 10, [1.5,1.5], samples)     # E     
        
          


        a,b,c,d,e,alfa,A,B,C,D,E = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],                                     axis=0)))
                                    
        truths=[a[0],b[0],c[0],d[0],e[0],alfa[0],A[0],B[0],C[0],D[0],E[0]]
        fig = corner.corner(samples, labels=["$a$","$b$", "$c$", "$d$","$e$","alfa", "A", "B", "C", "D", "E"], truths=truths, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}, title_fmt=".3f")
       
        print "a: ", a
        print "b: ", b
        print "c: ", c
        print "d: ", d
        print "e: ", e
        print "alfa: ", alfa      
        print "A: ", A
        print "B: ", B  
        print "C: ", C    
        print "D: ", D
        print "E: ", E
        #print "sigma: ", sigma
        
        fig.savefig("corner_plot_01.png")
        plt.show()



        #theta0 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                    #zip(*np.percentile(samples, [16, 50, 84],
                                                        #axis=0)))
        #print theta0

        #fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        #samples = sampler.chain
        #labels = ["theta"]
        #for i in range(ndim):
            #ax = axes[i]
            #ax.plot(samples[:, :, i], "k", alpha=0.01)
            #ax.set_xlim(0, len(samples))
            #ax.set_ylabel(labels[i])
            #ax.yaxis.set_label_coords(-0.1, 0.5)

        #axes[-1].set_xlabel("step number");
        #plt.show()




