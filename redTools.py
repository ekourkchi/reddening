#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

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
import copy 
import sklearn.datasets as ds
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

from Kcorrect import *
################################################################# 
def transform(inFile, band1 = 'r', band2 = 'w2'):
    
    table = getTable(inFile, band1=band1, band2=band2, faceOn=False)

    pgc = table['pgc']
    logWimx = table['logWimx']
    logWimx_e = table['logWimx_e']
    inc = table['inc']
    r_w1 = table['r_w1']
    c21w = table['c21w'] 
    Er_w1 = table['Er_w1']
    Ec21w = table['Ec21w']

    C82  = table['C82_w2']   # concentration 80%/20%
    mu50 = table['w2']+2.5*np.log10(2.*np.pi*(table['R50_w2']*60)**2)-2.5*np.log10(table['Wba'])

    z_scaler = StandardScaler()

    data = {'$Log( W_{mx}^i)$':logWimx, '$c21W2$':c21w, '$\mu 50$':mu50}
    n_comp = len(data)
    d = pd.DataFrame.from_dict(data)
    z_data = z_scaler.fit_transform(d)


    pca_trafo = PCA().fit(z_data)
    pca_data = pca_trafo.fit_transform(z_data)
    A = pca_trafo.explained_variance_ratio_                    # The importance of different PCAs components
    pca_inv_data = pca_trafo.inverse_transform(np.eye(n_comp)) # coefficients to make PCs from features


    delta = pca_data[:,0]-(pca_inv_data[0,0]*z_data[:,0]+pca_inv_data[0,1]*z_data[:,1]+pca_inv_data[0,2]*z_data[:,2])    
    
    return z_scaler, pca_trafo
################################################################# 
def getTable(inFile, band1 = 'r', band2 = 'w2', faceOn=False):
    
    inFile  = 'ESN_HI_catal.csv'
    table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

    table = extinctionCorrect(table)
    table = Kcorrection(table)

    if band2=='w1':
        text1 = band1+'-W1'      # example: cr-W1
        text2 = '$c21W_1$'       # example: c21w
    else: 
        text1 = band1+'-W2' 
        text2 = '$c21W_2$'

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

    index, = np.where(table['r_w1']>-5)
    table = trim(table, index)

    index, = np.where(table['Sqlt']>3)
    table = trim(table, index)

    index, = np.where(table['Wqlt']>3)
    table = trim(table, index)
    
    if faceOn:
        index, = np.where(table['flag']>0)
        table = trim(table, index)

        index, = np.where(table['flag']<3)
        table = trim(table, index)

    else: 

        index, = np.where(table['flag']==0)
        table = trim(table, index)
    
    return table
################################################################# 
def Fdelta(B, X):
    
    alfa = B[0]
    beta = B[1]
    gama = B[2]
    Xm = -beta/2./alfa
    X1 = np.ones_like(X)
    
    for i in range(len(X)):
        if X[i] >= Xm:
            X1[i] = Xm
        else: X1[i]=X[i]

    return alfa*X1**2+beta*X1+gama

################################################################# 
def extinct(ebv, band='r', method='SF11'):
    
    if method=='SFD98':
        if band=='u': extin = ebv*5.155  # original SFD98 values
        elif band=='g': extin = ebv*3.793
        elif band=='r': extin = ebv*2.751
        elif band=='i': extin = ebv*2.086
        elif band=='z': extin = ebv*1.479
        elif band=='w1': extin = ebv*0.171  # CCM values
        elif band=='w2': extin = ebv*0.096 
        else: extin = 0.
        return extin
        
    elif method=='YUAN13': #  Yuan et al. 2013 Table 2 values 
        if band=='u': extin = ebv*4.35
        elif band=='g': extin = ebv*3.31
        elif band=='r': extin = ebv*2.32
        elif band=='i': extin = ebv*1.72
        elif band=='z': extin = ebv*1.28
        elif band=='w1': extin = ebv*0.19
        elif band=='w2': extin = ebv*0.15
        else: extin = 0.
        return extin    
    
    elif method=='SF11':     #  Schlafly & Finkbeiner 2011 (737:103)    
        if band=='u': extin = ebv*4.239
        elif band=='g': extin = ebv*3.303
        elif band=='r': extin = ebv*2.285
        elif band=='i': extin = ebv*1.698
        elif band=='z': extin = ebv*1.263
        elif band=='w1': extin = ebv*0.19   # Yuan et al. 2013
        elif band=='w2': extin = ebv*0.15   # Yuan et al. 2013
        else: extin = 0.
        return extin      
    else: return 0.
################################################################# 
def extinction(ebv_lst, band='r', method='SF11'):
    
    N = len(ebv_lst)
    extinct_lst = np.zeros(N)
    for i in range(N):
        extinct_lst[i] = extinct(ebv_lst[i], band=band, method=method)
    
    return extinct_lst
################################################################# 
def trim(myDict, index):
    
    monDict = copy.deepcopy(myDict)
    for key in monDict:
        A = monDict[key]
        monDict[key] = A[index]
    
    return monDict


################################################################# 
def Reddening(r_w1, logWimx, c21w, AB, Delta):
    
    r_w1 = r_w1-Fdelta(Delta, c21w)
    R = r_w1-(AB[0]*logWimx+AB[1])
    
    return R
################################################################# 
### inc [deg]
def log_a_b(inc, q2):
    
    inc = inc*np.pi/180.
    
    b_a2 = (1.-q2)*(np.cos(inc))**2+q2
    a_b = np.sqrt(1./b_a2)
    
    return np.log(a_b)
################################################################# 
def faceON(table):

    index, = np.where(table['flag']>0)
    table0 = trim(table, index)

    index, = np.where(table0['flag']<3)
    table0 = trim(table0, index)
    
    #### for test not face-on
    ###index, = np.where(table['Sqlt']>3)
    ###table0 = trim(table, index)

    ###index, = np.where(table0['Wqlt']>3)
    ###table0 = trim(table0, index)

    ###index, = np.where(table0['flag']==0)
    ###table0 = trim(table0, index)

    ###index, = np.where(table0['inc']>80)
    ###table0 = trim(table0, index)

    ###index, = np.where(table0['inc']<90)
    ###table0 = trim(table0, index)
    #### end test
    
    logWimx = table0['logWimx']
    r_w1 = table0['r_w1']
    c21w = table0['c21w'] 
    Ec21w = table0['Ec21w']

    AB, cov  = np.polyfit(logWimx,r_w1, 1, cov=True, full = False)
    a0, b0 = AB[0],AB[1]
    a = 1./a0
    b = -1.*b0/a0
    print "AB: ", a0, b0
    print "Covariance AB: ", cov

    delta = r_w1-(a0*logWimx+b0)
   
    a, b, c  = np.polyfit(c21w,delta, 2)
    x = c21w
    y = delta
    mydata = odr.Data(x, y, wd=Ec21w, we=delta*0.+0.1)
    F = odr.Model(Fdelta)
    myodr = odr.ODR(mydata, F, beta0=[a,b,c])
    myoutput = myodr.run()
    print "Delta: ", myoutput.beta
    print"Std Error Delta: ", myoutput.sd_beta
    
    return [a0,b0], myoutput.beta, table0, cov, myoutput.sd_beta
################################################################# 
def extinctionCorrect(table):

    ### table is a structured array
    myTable = {}
    for name in table.dtype.names:
        myTable[name] = table[name]
    table = myTable
    ### table is now a dictionary
    
    table['u']  -= extinction(table['ebv'], band='u', method='SF11')
    table['g']  -= extinction(table['ebv'], band='g', method='SF11')
    table['r']  -= extinction(table['ebv'], band='r', method='SF11')
    table['i']  -= extinction(table['ebv'], band='i', method='SF11')
    table['z']  -= extinction(table['ebv'], band='z', method='SF11')
    table['w1'] -= extinction(table['ebv'], band='w1', method='SF11')
    table['w2'] -= extinction(table['ebv'], band='w2', method='SF11')
    table['u_'] -= extinction(table['ebv'], band='u', method='SF11')
    table['g_'] -= extinction(table['ebv'], band='g', method='SF11')
    table['r_'] -= extinction(table['ebv'], band='r', method='SF11')
    table['i_'] -= extinction(table['ebv'], band='i', method='SF11')
    table['z_'] -= extinction(table['ebv'], band='z', method='SF11')
    table['w1_']-= extinction(table['ebv'], band='w1', method='SF11')
    table['w2_']-= extinction(table['ebv'], band='w2', method='SF11')
    
    return table
    
################################################################# 
    
    
