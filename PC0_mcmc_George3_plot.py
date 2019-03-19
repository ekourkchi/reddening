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
#import esn_corner as corner

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


npz_file = "PC0_mcmc_"+band1+"_"+band2+".George3.npz"

if True:    ## MCMC part
        ndim = 4
        
        loaded = np.load(npz_file)
        samples = loaded['array']
        
        samples = data_trim(ndim, 0, [1.5,10], samples)
        samples = data_trim(ndim, 1, [3,10], samples)
        if band1=='i': 
            samples = data_trim(ndim, 2, [-10,1.5], samples)
        samples = data_trim(ndim, 3, [0,10], samples)
        
        T = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        
        for i in range(ndim):
            samples = data_trim(ndim, i, [T[i][0]-3*T[i][2],T[i][0]+3*T[i][1]], samples)
        
      
        l1,l2,sigma, err = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        

        truths=[l1[0],l2[0],sigma[0], err[0]]
        fig = corner.corner(samples, labels=["$log(\ell^2_0)$", "$log(\ell^2_1)$", r"$log(\sigma^2_f)$", "Error"], truths=truths, truth_color='r', quantiles=[0.16, 0.84],
                    levels=(1-np.exp(-1./8),1-np.exp(-0.5),1-np.exp(-0.5*4),1-np.exp(-0.5*9)),
                    show_titles=True, fill_contours=True, plot_density=True,
                    scale_hist=False,space=0, 
                    title_kwargs={"fontsize":15}, title_fmt=".3f", bins=25)        
        
        
        fig.savefig("PC0_mcmc_"+band1+"_"+band2+"_George3.png")
        plt.show()

