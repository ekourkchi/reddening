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
import copy 


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
    
    #### for test not face-n
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
    
    
