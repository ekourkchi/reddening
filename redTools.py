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
from collections import OrderedDict
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from Kcorrect import *
################################################################# 
def transform(inFile, band1 = 'r', band2 = 'w2'):
    
    table = getTable(inFile, band1=band1, band2=band2, faceOn=False)

    logWimx = table['logWimx']
    c21w = table['c21w'] 
    C82  = table['C82_w2']   # concentration 80%/20%
    mu50 = table['mu50']

    z_scaler = StandardScaler()

    data = {'$Log( W_{mx}^i)$':logWimx, '$c21W2$':c21w, '$\mu 50$':mu50}
    order_of_keys = ['$Log( W_{mx}^i)$', '$c21W2$', '$\mu 50$']
    list_of_tuples = [(key, data[key]) for key in order_of_keys]
    data = OrderedDict(list_of_tuples)
    n_comp = len(data)
    d =  pd.DataFrame.from_dict(data) 
    z_data = z_scaler.fit_transform(d)

    pca_trafo = PCA().fit(z_data)
    
    return z_scaler, pca_trafo
################################################################# 
def getTable(inFile, band1 = 'r', band2 = 'w2', faceOn=False, inc_lim=[0,100]):
    
    ###inFile  = 'ESN_HI_catal.csv'
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

    index, = np.where(table['Wba']>0.01)
    table = trim(table, index)
    
    index, = np.where(table['inc']>=inc_lim[0])
    table = trim(table, index)
    
    index, = np.where(table['inc']<inc_lim[1])
    table = trim(table, index)
    
    table['c21w'] = table['m21'] - table[band2]
    table['r_w1'] = table[band1] - table[band2]

    table['Ec21w'] = np.sqrt(table['m21_e']**2+0.05**2)
    table['Er_w1'] = 0.*table['r_w1']+0.1
    
    table['mu50'] = table[band2]+2.5*np.log10(2.*np.pi*(table['R50_'+band2]*60)**2)-2.5*np.log10(table['Wba'])
    
    dWba2 = ((0.1/6./table['R50_'+band2])**2)*(1+table['Wba']**2)
    c2 = (2.5/np.log(10))**2
    table['Emu50']=np.sqrt(c2*(0.1/6./table['R50_'+band2])**2+c2*dWba2/table['Wba']**2+0.05**2)
    
    table['EC82'] = (5*np.sqrt(2.)/np.log(10))*(0.1/6./table['R50_'+band2])

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
    
    return np.log10(a_b)
### inc [deg]
def Elogab2(inc, q2, Einc):
    
    inc = inc*np.pi/180.
    Einc = Einc*np.pi/180.
    
    dF2 = (0.5/np.log(10.))**2
    dF2 *= ((np.sin(2*inc))**2)*(q2-1)**2
    dF2 = dF2 / ((np.cos(inc))**2+q2*(np.sin(inc))**2)**2 
    
    return dF2*(Einc**2)

################################################################# 
def faceON_pca(inFile, band1 = 'r', band2 = 'w2'):
    
    scaler, pca = transform(inFile, band1=band1, band2=band2)
    
    table = getTable(inFile, band1=band1, band2=band2, faceOn=True)
    
    index, = np.where(table['Wba']>0.001)
    table = trim(table, index)

    pgc = table['pgc']
    logWimx = table['logWimx']
    logWimx_e = table['logWimx_e']
    inc = table['inc']
    r_w1 = table['r_w1']
    c21w = table['c21w'] 
    Er_w1 = table['Er_w1']
    Ec21w = table['Ec21w']

    C82  = table['C82_w2']   # concentration 80%/20%
    mu50 = table[band2]+2.5*np.log10(2.*np.pi*(table['R50_'+band2]*60)**2)-2.5*np.log10(table['Wba'])
    
    data = {'$Log( W_{mx}^i)$':logWimx, '$c21W2$':c21w, '$\mu 50$':mu50}
    order_of_keys = ['$Log( W_{mx}^i)$', '$c21W2$', '$\mu 50$']
    list_of_tuples = [(key, data[key]) for key in order_of_keys]
    data = OrderedDict(list_of_tuples)
    d = pd.DataFrame.from_dict(data)
    z_data = scaler.transform(d)
    pca_data = pca.transform(z_data)

    pc0 = pca_data[:,0]
    pc1 = pca_data[:,1]
    pc2 = pca_data[:,2]

    a0, b0  = np.polyfit(pc0, r_w1, 1)
    delta = np.abs(r_w1-(a0*pc0+b0))
    indx = np.where(delta<1)
    r_w1_ = r_w1[indx]
    pc0_ = pc0[indx]    
    
    AB, cov  = np.polyfit(pc0_, r_w1_, 1, cov=True, full = False)
    a0, b0 = AB[0], AB[1]
    delta = np.abs(r_w1_-(a0*pc0_+b0))
    rms = np.sqrt(np.mean(np.square(delta)))
    
    return scaler, pca, AB, cov, rms
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
def getBand(inFile, band1 = 'r', band2 = 'w2'):

    scaler, pca, AB, cov, rms = faceON_pca(inFile, band1=band1, band2=band2)
    a0, b0 = AB[0], AB[1]
    
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
    mu50 = table['mu50']    
    data = {'$Log( W_{mx}^i)$':logWimx, '$c21W2$':c21w, '$\mu 50$':mu50}
    order_of_keys = ['$Log( W_{mx}^i)$', '$c21W2$', '$\mu 50$']
    list_of_tuples = [(key, data[key]) for key in order_of_keys]
    data = OrderedDict(list_of_tuples)
    n_comp = len(data)
    d = pd.DataFrame.from_dict(data)
    z_data = scaler.transform(d)
    pca_data = pca.transform(z_data)

    pc0 = pca_data[:,0]
    pc1 = pca_data[:,1]
    pc2 = pca_data[:,2]   
    
    Input = [pgc, r_w1, pc0, inc]
    Reddening = r_w1-(a0*pc0+b0)
    
    u = scaler.mean_
    s = scaler.scale_
    v = scaler.var_
    ## z = (x-u)/s
    ##u: mean  s:scale  var=s**2
    
    A = pca.explained_variance_ratio_              # The importance of different PCAs components

    pca_inv_data = pca.inverse_transform(np.eye(n_comp)) # coefficients to make PCs from features
    p0 = pca_inv_data[0,0]
    p1 = pca_inv_data[0,1]
    p2 = pca_inv_data[0,2]
    
    logWimx_e = table['logWimx_e']
    Ec21w = table['Ec21w']
    Emu50 = table['Emu50']
    
    table['Epc0'] = np.sqrt((p0*logWimx_e/s[0])**2+(p1*Ec21w/s[1])**2+(p2*Emu50/s[2])**2)
    
    #print "u: ", u
    #print "s: ", s
    #print "A: ", A
    #print "pca_inv_data: ", pca_inv_data
    
    
    return Reddening, Input, [scaler, pca, AB, cov, rms, table]    

################################################################# 
def redCorrect(inc, pc0, band1='r', band2='w2'):
    
    a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1=band1, band2=band2)
    
    return log_a_b(inc, 10**(-1.*theta))*(a*pc0**3+b*pc0**2+c*pc0+d)
    
################################################################# 
def getReddening_params(band1='r', band2='w2'):
    
    a=0;b=0;c=0;d=0;alpha=0;beta=0;gamma=0
    Ealpha=0;Ebeta=0 
    
    if band2=='w2':
        if band1=='u':
            a= -0.013
            b= -0.055
            c= 0.096
            d= 1.202
            alpha = 0.392
            beta = 0.473
            gamma = 2.935
            Ealpha = 0.006
            Ebeta = 0.008
        if band1=='g':
            a= -0.013
            b= -0.056
            c= 0.081
            d= 0.873
            alpha = 0.287
            beta = -0.424
            gamma = 3.131
            Ealpha = 0.004
            Ebeta = 0.006            
        if band1=='r':
            a= -0.012
            b= -0.054
            c= 0.069
            d= 0.692
            alpha = 0.215
            beta = -0.843
            gamma = 3.237
            Ealpha = 0.004
            Ebeta = 0.006             
        if band1=='i':
            a= -0.012
            b= -0.055
            c= 0.051
            d= 0.586
            alpha = 0.176
            beta = -1.071
            gamma = 3.286
            Ealpha = 0.004
            Ebeta = 0.005             
        if band1=='z':
            a= -0.011
            b= -0.055
            c= 0.023
            d= 0.474
            alpha = 0.157
            beta = -1.174
            gamma = 3.411
            Ealpha = 0.004
            Ebeta = 0.005             
        if band1=='w1':
            a= 0.
            b= 0.
            c= 0.015
            d= 0.039
            alpha = 0.008
            beta = -0.571
            gamma = 1.805
            Ealpha = 0.005
            Ebeta = 0.006 
    
    elif band2=='w1':
        if band1=='u':
            a= -0.015
            b= -0.050
            c= 0.139
            d= 1.181
            alpha = 0.377
            beta = 1.005
            gamma = 2.878
            Ealpha = 0.006
            Ebeta = 0.009 
        if band1=='g':
            a= -0.015
            b= -0.055
            c= 0.100
            d= 0.813
            alpha = 0.278
            beta = 0.133
            gamma = 3.170
            Ealpha = 0.004
            Ebeta = 0.006 
        if band1=='r':
            a= -0.014
            b= -0.053
            c= 0.079
            d= 0.623
            alpha = 0.204
            beta = -0.273
            gamma = 3.303
            Ealpha = 0.003
            Ebeta = 0.005 
        if band1=='i':
            a= -0.013
            b= -0.051
            c= 0.062
            d= 0.513
            alpha = 0.159
            beta = -0.496
            gamma = 3.342
            Ealpha = 0.003
            Ebeta = 0.005 
        if band1=='z':
            a= -0.011
            b= -0.045
            c= 0.049
            d= 0.402
            alpha = 0.128
            beta = -0.597
            gamma = 3.463
            Ealpha = 0.003
            Ebeta = 0.005 
   
    
    return a,b,c,d, alpha, beta, gamma, Ealpha, Ebeta   

################################################################# 
def george_params(band1='r'):
    if band1=='u': theta = [3.785  , 6.944, 0.904, 0.165]
    if band1=='g': theta = [3.255,  5.817, -0.312,  0.110]
    if band1=='r': theta = [3.078,  5.431, -0.883, 0.097]
    if band1=='i': theta = [2.910,  5.172, -1.390,  0.099]
    if band1=='z': theta = [2.861 ,  4.930, -1.904,  0.099]
    if band1=='w1':theta = [2.68218476e+00,  1.02518650e+01, -4.78962522e+00,  1.00000000e-02]
    
    return theta
################################################################# 
## p0 = 0.524
## p1 = 0.601
## p2 = -0.603
## u0 = 2.47 ; u1 = 1.63 ; u2 = 23.35
## sigma1 = 0.18 ; sigma2 = 1.15 ; sigma3 = 1.38
def get_PC(w2, m21, logWimx, Wba, R50_w2):
    
    we = 2.5*np.log10(2.*np.pi*(R50_w2*60)**2)-2.5*np.log10(Wba)
    mu50 = w2+we
    c21w = m21-w2
    P0 = 0.524*(logWimx-2.47)/0.18+0.601*(c21w-1.63)/1.15-0.603*(mu50-23.35)/1.38
    
    return logWimx, c21w, mu50, P0

def get_w(P0, m21, logWimx, Wba, R50_w2):
    
    we = 2.5*np.log10(2.*np.pi*(R50_w2*60)**2)-2.5*np.log10(Wba)
    tmp = P0 - 0.524*(logWimx-2.47)/0.18 - 0.601*(m21-1.63)/1.15 + 0.603*(we-23.35)/1.38
    w2 = tmp/((-0.601/1.15)-0.603/1.38)
    
    return w2
################################################################# 
## Shuffling the input data;
#################################################################
def esn_shuffle(array, seed=0):
        random.seed(seed)
        random.shuffle(array)
        return array
#################################################################   
def prepareSamples(table, N_test=200, Seed_test=0, N_cross=200, Seed_cross=100):
    
    ### table is a structured array
    myTable = {}
    for name in table.dtype.names:
        myTable[name] = table[name]
    table = myTable
    ### table is now a dictionary (hash table)    

    ## u0, g0, r0, i0, z0 are already K-/extinction- corrected
    pgc     = table['pgc']      # id
    inc     = table['inc']      # inclination [deg]
    pc0     = table['pc0']      # main principal component
    u0      = table['u']
    g0      = table['g']
    r0      = table['r']
    i0      = table['i']
    z0      = table['z']
    w1      = table['w1']
    w2      = table['w2']
    logWimx = table["logWimx"]  # inclination corrected HI 21 cm linewidth
    m21     = table["m21"]      # HI 21cm magnitude: m21 = -2.5*np.log10(Flux_21cm)+17.40
    mu50    = table["mu50"]     # W2 effective surface brightness (corrected for the axial ratio)
    Wba     = table["Wba"]      # WISE photometry axial ration, i.e. b/a
    R50     = table["R50_w2"]   # W2 half light radius

    table['u0'] = u0
    table['g0'] = g0
    table['r0'] = r0
    table['i0'] = i0
    table['z0'] = z0
    table['w10']= w1
    
    ## Reddening corrections
    u  = u0-redCorrect(inc, pc0, band1='u', band2='w2')
    g  = g0-redCorrect(inc, pc0, band1='g', band2='w2')
    r  = r0-redCorrect(inc, pc0, band1='r', band2='w2')
    i  = i0-redCorrect(inc, pc0, band1='i', band2='w2')
    z  = z0-redCorrect(inc, pc0, band1='z', band2='w2')
    w1 = w1-redCorrect(inc, pc0, band1='w1', band2='w2')
    
    
    table['u'] = u
    table['g'] = g
    table['r'] = r
    table['i'] = i
    table['z'] = z
    table['w1']= w1
        
    table['c21w']  = m21-w2
    table['g_w2']  = g-w2
    table['r_w2']  = r-w2
    table['i_w2']  = i-w2
    table['z_w2']  = z-w2
    table['w1_w2'] = w1-w2
    table['u_g']   = u-g
    table['g_r']   = g-r
    table['r_i']   = r-i
    table['g_i']   = g-i
    table['i_z']   = i-z
    table['r_z']   = r-z
    table['g_z']   = g-z


    data = {'$g-r$':g-r, '$r-i$':r-i, '$i-z$':i-z}
    order_of_keys = ['$g-r$', '$r-i$', '$i-z$']
    list_of_tuples = [(key, data[key]) for key in order_of_keys]
    data = OrderedDict(list_of_tuples)
    d = pd.DataFrame.from_dict(data)
    z_scaler = StandardScaler()
    z_data = z_scaler.fit_transform(d)
    pca = PCA().fit(z_data)
    pca_data = pca.transform(z_data)

    table["Cpc1"] = pca_data[:,0]
    table["Cpc2"] = pca_data[:,1]
    table["Cpc3"] = pca_data[:,2]   
    
    n = len(table['pgc'])
    indices = np.arange(n)
    indices = esn_shuffle(indices, seed=Seed_test)
    table = trim(table, indices)
    indices = np.arange(N_test)
    table_tst = trim(table, indices)
    indices = np.arange(N_test, n)
    table_tmp = trim(table, indices)


    n = len(table_tmp['pgc'])
    indices = np.arange(n)
    indices = esn_shuffle(indices, seed=Seed_cross)
    table_tmp = trim(table_tmp, indices)
    indices = np.arange(N_cross)
    table_cvl = trim(table_tmp, indices)
    indices = np.arange(N_cross, n)
    table_trn = trim(table_tmp, indices)
    
    table_tst["scaler"] = z_scaler
    table_cvl["scaler"] = z_scaler
    table_trn["scaler"] = z_scaler
    table_tst["Cpca"] = pca
    table_cvl["Cpca"] = pca
    table_trn["Cpca"] = pca
   
    return table_tst, table_cvl, table_trn
#################################################################   
def ML_data(table, features, output):
    
    n = len(features)  # number of features
    N = len(table['pgc'])   # training
    X = np.ones(shape = (n,N))
    for j in range(n):
        X[j] = table[features[j]]
        
    return X.T, table[output]
#################################################################   
def esn_RForest(table_tst, table_cvl, table_trn, features, output, \
    max_depth=2000, n_estimators=1000, max_features=3, \
        min_samples_leaf=5, bootstrap=True):
    
    x_trn, y_trn = ML_data(table_trn, features, output)
    x_cvl, y_cvl = ML_data(table_cvl, features, output)
    x_tst, y_tst = ML_data(table_tst, features, output)
    ################################

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, \
        max_features=max_features, min_samples_leaf=min_samples_leaf, \
            bootstrap=bootstrap)

    regr.fit(x_trn, y_trn)
    p_y_trn  = regr.predict(x_trn)
    p_y_cvl  = regr.predict(x_cvl)
    p_y_tst  = regr.predict(x_tst)
    
    return regr, x_trn, y_trn, p_y_trn, x_cvl, y_cvl, p_y_cvl, x_tst, y_tst, p_y_tst
#################################################################   
def predictor(w2, table, regr, features, output, index=0):
    
    outDict = {}
    
    M21 = table["m21"][index]
    M0  = table["logWimx"][index]
    WBA = table["Wba"][index]
    R50 = table["R50_w2"][index]
    INC = table["inc"][index]
    
    u0 = table['u0'][index]
    g0 = table['g0'][index]
    r0 = table['r0'][index]
    i0 = table['i0'][index]
    z0 = table['z0'][index]
    w1 = table['w1'][index]
    
    logWimx, c21w, mu50, P0 = get_PC(w2, M21, M0, WBA, R50)
    outDict["logWimx"] = logWimx
    outDict["mu50"] = mu50
    outDict["c21w"] = c21w
    
    outDict["pc0"] = P0
    
    u_t  = u0-redCorrect(INC, P0, band1='u', band2='w2')
    g_t  = g0-redCorrect(INC, P0, band1='g', band2='w2')
    r_t  = r0-redCorrect(INC, P0, band1='r', band2='w2')
    i_t  = i0-redCorrect(INC, P0, band1='i', band2='w2')
    z_t  = z0-redCorrect(INC, P0, band1='z', band2='w2')
    w1_t = w1-redCorrect(INC, P0, band1='w1', band2='w2')
    
    # Colors
    outDict["u_g"] = u_t-g_t
    outDict["g_r"] = g_t-r_t
    outDict["r_i"] = r_t-i_t
    outDict["i_z"] = i_t-z_t
    outDict["g_z"] = g_t-z_t
    outDict["g_i"] = g_t-i_t
    outDict["r_z"] = r_t-z_t
    
    gr = np.asarray([g_t])-np.asarray([r_t])
    ri = np.asarray([r_t])-np.asarray([i_t])
    iz = np.asarray([i_t])-np.asarray([z_t])
    data = {'$g-r$':gr, '$r-i$':ri, '$i-z$':iz}
    order_of_keys = ['$g-r$', '$r-i$', '$i-z$']
    list_of_tuples = [(key, data[key]) for key in order_of_keys]
    
    data = OrderedDict(list_of_tuples)
    d = pd.DataFrame.from_dict(data)
    z_scaler = table["scaler"]
    pca = table["Cpca"]
    z_data = z_scaler.fit_transform(d)
    pca_data = pca.transform(z_data)
    outDict["Cpc1"] = pca_data[0][0]
    outDict["Cpc2"] = pca_data[0][1]
    outDict["Cpc3"] = pca_data[0][2] 
    
    
    # Mags
    outDict["g_w2"] = g_t-w2
    outDict["r_w2"] = r_t-w2
    outDict["i_w2"] = i_t-w2
    outDict["z_w2"] = z_t-w2
    
    outDict["w1_w2"]= w1_t-w2
    
    inList=[]
    for f in features: inList.append(outDict[f])
        
    
    x_t = np.asarray([inList])
    y_t = regr.predict(x_t) 
    
    # guess - prediction
    delta = outDict[output]-y_t
    
    return delta
#################################################################      
def solver(f, x1, x2, threshold=0.01, N_iter=0):
    
    n = N_iter
    y1 = f(x1)
    y2 = f(x2)
    if np.sign(y1)*np.sign(y2)==1: return None
    elif np.abs(x2-x1)<threshold: 
        m = (y2-y1)/(x2-x1)
        b = y1-m*x1
        return -b/m,n+2

    d = (x2-x1)/5.
    x0 = x1
    n+=2
    while x0<x2:
        if np.sign(f(x0))*np.sign(f(x0+d))==-1: 
            return solver(f, x0, x0+d, threshold=threshold,N_iter=n+2)
        x0+=d
        n+=2    
#################################################################   

