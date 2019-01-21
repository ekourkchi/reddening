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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
################################################################# 
### inc [deg]
def log_a_b(inc, q2):
    
    inc = inc*np.pi/180.
    
    b_a2 = (1.-q2)*(np.cos(inc))**2+q2
    a_b = np.sqrt(1./b_a2)
    #a_b = 1./np.cos(inc)
    #a_b = A*(np.cos(inc))**2+B*(np.cos(inc))+(1.-A-B)
    return np.log(a_b)
###################################
def R_model(i, logWimx, c21w):
    

   
    
#######################################     
    c = -0.074
    d = 0.467
    e = 0.189
    f = -0.970
    
    alfa = 2.665
    
    A = -0.111
    B = 0
    C = 0.159
    D = 0 
    E = 0 

    
    q2 = 10**(-1.*alfa)   
  
######################################     
    #c = -0.074
    #d = 0.523
    #e = 0.171
    #f = -1.073
    
    #alfa = 2.651
    
    #A = -0.215
    #B = 0.034
    #C = 0.340
    #D = 0 
    #E = 0 

    #q2 = 10**(-1.*alfa)   
######################################     
######################################     
    c = -0.075
    d = 0.430
    e = 0.191
    f = -0.884
    
    alfa = 2.712
    
    A = -0.046
    B = 0
    C = 0
    D = 0 
    E = 0 
    
    q2 = 10**(-1.*alfa)   
######################################
    c = -0.062
    d = 0.349
    e = 0.157
    f = -0.753
    
    alfa = 3.712
    
    A = 0
    B = 0
    C = 0
    D = 0 
    E = 0 
    
    q2 = 10**(-1.*alfa)   
######################################     
######################################
    c = -0.0567
    d = 0.490
    e = 0.128
    f = -0.989
    
    alfa = 2.641
    
    A = -0.237
    B = 0.069
    C = 0.355
    D = -0.007
    E = 0 

    
    q2 = 10**(-1.*alfa)   
######################################      
######################################
    #c = -0.069
    #d = 0.515
    #e = 0.158
    #f = -1.047
    
    #alfa = 2.642
    
    #A = -0.294
    #B = 0.007
    #C = 0.489
    #D = -0.007
    #E = 0.026

    
    #q2 = 10**(-1.*alfa)   
######################################    
    
    model = log_a_b(i, q2)*(c*logWimx*c21w+d*logWimx+e*c21w+f)+(A*logWimx+B*c21w+C+D*c21w**2+E*logWimx*c21w)
    
    
    return model

################################################################# 
def Reddening(r_w1, logWimx, c21w):
    
    if True:
        m = 1.4192102549431043
        b = 2.459225503637881
        a0 = 1./m
        b0 = -1.*b*a0
    
        ## parabolic fix
        alfa = -0.02291157388715272 
        beta = 0.22941051602065887 
        gama = -0.3797350278273751
        r_w1 = r_w1-(alfa*c21w**2+beta*c21w+gama)

        return r_w1-(a0*logWimx+b0)
    

    
################################################################# 
def plot_Rinc_array(R, Input, scatter=False):
    
    fig = py.figure(figsize=(12, 13), dpi=100)    
    fig.subplots_adjust(wspace=0, top=0.95, bottom=0.1, left=0.07, right=0.98)
    gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1,1,1]) 
    p = 0
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1

    plot_Rinc(ax, R, Input, c21w_=[-2,1], logWimx_=[2.6,3.0], color='blue', scatter=True, xlabel=False, ylabel=True)
    plt.setp(ax.get_xticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[2.6,3.0], color='green', scatter=True, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)    
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[2,3], logWimx_=[2.6,3.0], color='orange', scatter=True, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[3,6], logWimx_=[2.6,3.0], color='red', scatter=True, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[-2,1], logWimx_=[2.4,2.6], color='blue', scatter=True, xlabel=False, ylabel=True)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ####################################################    
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[2.4,2.6], color='green', scatter=True, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[2,3], logWimx_=[2.4,2.6], color='orange', scatter=True, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[3,6], logWimx_=[2.4,2.6], color='red', scatter=True, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################    
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[-2,1], logWimx_=[2.2,2.4], color='blue', scatter=True, xlabel=False, ylabel=True)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ####################################################    
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[2.2,2.4], color='green', scatter=True, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################    
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[2,3], logWimx_=[2.2,2.4], color='orange', scatter=True, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[3,6], logWimx_=[2.2,2.4], color='red', scatter=True, xlabel=False, ylabel=False)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################     
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[-2,1], logWimx_=[1.8,2.2], color='blue', scatter=True, xlabel=True, ylabel=True)

    yticks = ax.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    xticks = ax.xaxis.get_major_ticks()
    xticks[-1].label1.set_visible(False)
    #plt.setp(ax.get_xticklabels(), visible=False)
    ####################################################    
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[1.8,2.2], color='green', scatter=True, xlabel=True, ylabel=False)

    xticks = ax.xaxis.get_major_ticks()
    xticks[-1].label1.set_visible(False)
    #plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[2,3], logWimx_=[1.8,2.2], color='orange', scatter=True, xlabel=True, ylabel=False)

    xticks = ax.xaxis.get_major_ticks()
    xticks[-1].label1.set_visible(False)
    #plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################
    ax = plt.subplot(gs[p]) ; p+=1
    plot_Rinc(ax, R, Input, c21w_=[3,6], logWimx_=[1.8,2.2], color='red', scatter=True, xlabel=True, ylabel=False)

    #xticks = ax.xaxis.get_major_ticks()
    #xticks[-1].label1.set_visible(False)
    #plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ####################################################     
    
    plt.subplots_adjust(hspace=.0, wspace=0)

    plt.show()
    

################################################################# 
def plot_Rinc(ax, R, Input, c21w_=[1,2], logWimx_=[2.4,2.6], color='red', scatter=False, xlabel=True, ylabel=True):
    
    pgc     = Input[0]
    r_w1    = Input[1]
    logWimx = Input[2]
    c21w    = Input[3]
    inc     = Input[4]
    
    #R = R - R_model(inc, logWimx, c21w)
    
    index = np.where(c21w<=c21w_[1])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(c21w>=c21w_[0])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(logWimx>=logWimx_[0])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(logWimx<=logWimx_[1])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]    
    
    


    
    if False:
        
        inc__ = np.arange(45,90,0.1)
        logWimx__ = np.ones_like(inc__)*(logWimx_[0]+logWimx_[1])*0.5
        c21w__ = np.ones_like(inc__)*(c21w_[0]+c21w_[1])*0.5
        R__ = R_model(inc__, logWimx__, c21w__)

        N = len(inc__)
        r_min = np.zeros(N)
        r_max = np.zeros(N)
        for ii in range(N):
            
            r_lst = []
            for iw in np.arange(logWimx_[0], logWimx_[1], 0.05):
                for ic in np.arange(c21w_[0], c21w_[1], 0.05):
                    r = R_model(inc__[ii], iw, ic)
                    r_lst.append(r)
            r_min[ii] = np.min(r_lst)
            r_max[ii] = np.max(r_lst)
        ax.fill_between(inc__, r_min, r_max, alpha=0.3, facecolor='blue')
        
    if True:
        
        X = np.atleast_2d(inc).T
        y = R
        dy = 0.1*np.abs(y)
        
        kernel = ConstantKernel() + Matern(length_scale=20, nu=5/2) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                n_restarts_optimizer=10)
        gp.fit(X, y)  
        
        x = np.atleast_2d(np.linspace(40,1000, 1000)).T
        y_pred, sigma = gp.predict(x, return_std=True)
        plt.plot(x, y_pred, 'g-', label=u'Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([y_pred - 1.000 * sigma,   # 1.9600
                                (y_pred + 1.000 * sigma)[::-1]]),
                alpha=.3, fc='g', ec='None', label='95% confidence interval')
        


    if scatter:
        ax.plot(inc, R, '.', color='black', markersize=3, alpha=0.3)
        
    xl = []
    yl= []
    yel=[]
    
    low = 45; high=90
    for i in np.arange(low,high,5):
        
        x = []
        y = []
        for ii in range(len(R)):
            xi = inc[ii]
            if xi>i and xi<=i+5:
                x.append(xi)
                y.append(R[ii])
        if len(x)>0:
            
            x = np.asarray(x)
            y = np.asarray(y)
            
            average   = np.median(y)
            stdev = np.std(y)
            
            index = np.where(y<average+2.*stdev)
            x = x[index]
            y = y[index]
            
            index = np.where(y>average-2.*stdev)
            x = x[index]
            y = y[index]        

            #ax.errorbar(np.median(x), np.median(y), yerr=np.std(y), fmt='o', color=color)
            
            xl.append(np.median(x))
            yl.append(np.median(y))
            yel.append(np.std(y))
            
            
    ax.tick_params(which='major', length=5, width=2.0, direction='in')
    ax.tick_params(which='minor', length=2, color='#000033', width=1.0, direction='in')
    ax.minorticks_on()

    ax.text(45,1.2, r'$c21w: $'+"(%.0f" % (c21w_[0])+', '+"%.1f)" % (c21w_[1]))
    ax.text(45,0.9, r'$Log(W^i_{mx}): $'+"(%.1f" % (logWimx_[0])+', '+"%.1f)" % (logWimx_[1]))

    ax.set_ylim([-0.9,1.4])     
    ax.set_xlim([41,99])   
    
    ax.plot([40,100],[0,0], 'k:')
    
    if xlabel: ax.set_xlabel(r'$Inc. [deg]$')
    if ylabel: 
        ax.set_ylabel(r'$\delta$') 
        #ax.set_ylabel(r'$\delta_{data}-\delta_{model}$') 
    
################################################################# 
def plot_RC21W(R, Input, inc_=[80,90], logWimx_=[2.4,2.6], color='red', scatter=False):
    
        
    pgc     = Input[0]
    r_w1    = Input[1]
    logWimx = Input[2]
    c21w    = Input[3]
    inc     = Input[4]
      
    index = np.where(inc<=inc_[1])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(inc>=inc_[0])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(logWimx>=logWimx_[0])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(logWimx<=logWimx_[1])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    if scatter:
        plt.plot(c21w, R, '.', color=color, markersize=2, alpha=0.3)

    xl = []
    yl= []
    yel=[]
    
    low = np.min(c21w) ; high = np.max(c21w)
    low = -1; high=7
    for i in np.arange(low,high,1):
        
        x = []
        y = []
        for ii in range(len(R)):
            xi = c21w[ii]
            if xi>=i and xi<i+1:
                #x.append(xi)
                x.append(i+0.5)
                y.append(R[ii])
        if len(x)>0:
            
            x = np.asarray(x)
            y = np.asarray(y)
            
            average   = np.median(y)
            stdev = np.std(y)
            
            index = np.where(y<average+2.*stdev)
            x = x[index]
            y = y[index]
            
            index = np.where(y>average-2.*stdev)
            x = x[index]
            y = y[index]        

            plt.errorbar(np.median(x), np.median(y), yerr=np.std(y), fmt='o', color=color)
            
            xl.append(np.median(x))
            yl.append(np.median(y))
            yel.append(np.std(y))


    plt.ylim([-1,1.5])     
    plt.xlim([-1,7]) 
    
    




################################################################# 
def plot_RlogW(R, Input, inc_=[80,90], c21w_=[1,2], color='red', scatter=False):
    
        
    pgc     = Input[0]
    r_w1    = Input[1]
    logWimx = Input[2]
    c21w    = Input[3]
    inc     = Input[4]
      
    index = np.where(inc<=inc_[1])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(inc>=inc_[0])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(c21w>=c21w_[0])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]

    index = np.where(c21w<=c21w_[1])
    r_w1 = r_w1[index]
    logWimx = logWimx[index]
    pgc = pgc[index]
    c21w = c21w[index]
    inc = inc[index]
    R = R[index]
    
    if scatter:
        plt.plot(logWimx, R, '.', color=color, markersize=2, alpha=0.3)

    xl = []
    yl= []
    yel=[]
    
    low = np.min(logWimx) ; high = np.max(logWimx)
    low = 1.7; high=3
    for i in np.arange(low,high,0.10):
        
        x = []
        y = []
        for ii in range(len(R)):
            xi = logWimx[ii]
            if xi>=i and xi<i+0.1:
                #x.append(xi)
                x.append(i+0.05)
                y.append(R[ii])
        if len(x)>0:
            
            x = np.asarray(x)
            y = np.asarray(y)
            
            average   = np.median(y)
            stdev = np.std(y)
            
            index = np.where(y<average+2.*stdev)
            x = x[index]
            y = y[index]
            
            index = np.where(y>average-2.*stdev)
            x = x[index]
            y = y[index]        

            plt.errorbar(np.median(x), np.median(y), yerr=np.std(y), fmt='o', color=color)
            
            xl.append(np.median(x))
            yl.append(np.median(y))
            yel.append(np.std(y))


    plt.ylim([-1,1.5])     
    plt.xlim([1.7,3]) 
    
################################################################# 

def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return  1.4192102549431043*x + B[0]
################################################################# 
inFile  = 'all_color_diff2.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

PGC = table['PGC']


########################################################### Begin
inFile  = 'ESN_HI_catal_v01.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

pgc = table['pgc']
logWimx = table['logWimx']
c21w = table['c21w']
inc = table['inc']
r = table['r']
w1 = table['w1']
flag = table['flag']
Sqlt = table['Sqlt']
Wqlt = table['Wqlt']

r_w1 = r-w1


#index = np.where(Sqlt>4)
#r_w1 = r_w1[index]
#logWimx = logWimx[index]
#pgc = pgc[index]
#c21w = c21w[index]
#flag = flag[index]
#inc = inc[index]
#Sqlt = Sqlt[index]
#Wqlt = Wqlt[index]

#index = np.where(Wqlt>4)
#r_w1 = r_w1[index]
#logWimx = logWimx[index]
#pgc = pgc[index]
#c21w = c21w[index]
#flag = flag[index]
#inc = inc[index]
#Sqlt = Sqlt[index]
#Wqlt = Wqlt[index]



index = np.where(inc>45)
#index = np.where(flag<3)
r_w1 = r_w1[index]
logWimx = logWimx[index]
pgc = pgc[index]
c21w = c21w[index]
flag = flag[index]
inc = inc[index]

#index = np.where(inc<55)
#index = np.where(flag>0)
#r_w1 = r_w1[index]
#logWimx = logWimx[index]
#pgc = pgc[index]
#c21w = c21w[index]
#flag = flag[index]
#inc = inc[index]


index = np.where(logWimx>1)
r_w1 = r_w1[index]
logWimx = logWimx[index]
pgc = pgc[index]
c21w = c21w[index]
flag = flag[index]
inc = inc[index]


index = np.where(r_w1<4)
r_w1 = r_w1[index]
logWimx = logWimx[index]
pgc = pgc[index]
c21w = c21w[index]
flag = flag[index]
inc = inc[index]


#for pp in pgc: print pp
#print len(pgc)


a0, b0  = np.polyfit(logWimx,r_w1, 1)
a = 1./a0
b = -1.*b0/a0
print a,b 

#a = 1.4192102549431043
#b = 2.553184176602721
#a0 = 1./a
#b0 = -1.*b*a0

#plt.plot([-1,1], [-a+b,a+b], 'g--')


###########   Test
#y = r_w1
#x = logWimx
#yerr = 0.1*np.ones_like(x)
#A = np.vstack((np.ones_like(x), x)).T
#C = np.diag(yerr * yerr)
#cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
#b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
#plt.plot([m_ls*1.7+b_ls,m_ls*3.0+b_ls], [1.7,3.0], '-', color='black')
###########



#delta = r_w1-(a0*logWimx+b0)
#print 'sigma1 = ', np.sqrt(np.median(delta**2))

#x = r_w1
#y = logWimx
#linear = odr.Model(f)
#mydata = odr.Data(x, y, wd=x*0.+1, we=y*0.+1)
#myodr = odr.ODR(mydata, linear, beta0=[0.1])
#myoutput = myodr.run()
#b = myoutput.beta[0]


#a = 1.4192102549431043
#b = 2.459225503637881
#a0 = 1./a
#b0 = -1.*b*a0
#plt.plot([-1,1], [-a+b,a+b], '--', color='black')
#print "BBB: ", b
##BBB:  2.459225503637881
#########################################################################  

## linear fix
#a = 0.1659241187077633 
#b = -0.35566041164213746
#for i in range(len(pgc)):
    #if c21w[i]<3:
        #r_w1[i] = r_w1[i]-(a*c21w[i]+b)
    #else: 
        #r_w1[i] = r_w1[i]-0.13

## parabolic fix
#a = -0.02291157388715272 
#b = 0.22941051602065887 
#c = -0.3797350278273751
#r_w1 = r_w1-(a*c21w**2+b*c21w+c)


#delta = r_w1-(a0*logWimx+b0)
#print 'sigma2 = ', np.sqrt(np.median(delta**2))

#for i in range(len(pgc)):
   #if c21w[i]<1  :
       #plt.plot(r_w1[i], logWimx[i], 'b.', markersize=3, alpha=0.4)
   #if c21w[i]>=1 and c21w[i]< 3:
       #plt.plot(r_w1[i], logWimx[i], 'g.', markersize=3, alpha=0.4)       
   #if c21w[i]>=3:
       #plt.plot(r_w1[i], logWimx[i], 'r.', markersize=3, alpha=0.4)    
       
#plt.xlabel('cr-w1')
#plt.ylabel(r'$Log( W_{mx}^i)$')

#plt.xlim([-2,4])
#plt.ylim([1.7,3])       
#plt.show()
#########################################################################  


Input = [pgc, r_w1, logWimx, c21w, inc]
R = Reddening(r_w1, logWimx, c21w)





#c21w_ = [0.,1.]
#plot_RlogW(R, Input, inc_=[85,90], c21w_=c21w_, color='red', scatter=False)
#plot_RlogW(R, Input, inc_=[75,80], c21w_=c21w_, color='green', scatter=False)
#plot_RlogW(R, Input, inc_=[65,70], c21w_=c21w_, color='blue', scatter=False)
#plot_RlogW(R, Input, inc_=[55,60], c21w_=c21w_, color='black', scatter=False)

#plt.xlabel(r'$Log( W_{mx}^i)$')
#plt.ylabel(r'$R$') 


#logWimx_ = [2.4,2.6]
#plot_RC21W(R, Input, inc_=[85,90], logWimx_=logWimx_, color='red', scatter=False)
#plot_RC21W(R, Input, inc_=[75,80], logWimx_=logWimx_, color='green', scatter=False)
#plot_RC21W(R, Input, inc_=[65,70], logWimx_=logWimx_, color='blue', scatter=False)
#plot_RC21W(R, Input, inc_=[55,60], logWimx_=logWimx_, color='black', scatter=False)

#plt.xlabel(r'$c21w$')
#plt.ylabel(r'$R$') 



plot_Rinc_array(R, Input, scatter=True)





#########################################################################  

#plt.plot(r_w1, c21w, 'b.')
    
#plt.xlim([-2,4])
#plt.ylim([-2,8])   
#plt.xlabel('cr-w1')
#plt.ylabel(r'$c21w$')     
#########################################################################  
#########################################################################  

#plt.plot(c21w, logWimx, 'g.', markersize=2, alpha=0.3)
    
#plt.xlim([-2,8])
#plt.ylim([1.7,3])  

#plt.ylabel(r'$Log( W_{mx}^i)$')
#plt.xlabel(r'$c21w$') 
#########################################################################  
#########################################################################  

#delta = r_w1-(a0*logWimx+b0)
#plt.plot(c21w, delta, '.', color='black', markersize=2, alpha=0.3)
#xl = []
#yl= []
#yel=[]
#for i in range(-1,6):
    
    #x = []
    #y = []
    #for ii in range(len(c21w)):
        #xi = c21w[ii]
        #if xi>=i and xi<i+1:
            #x.append(xi)
            #y.append(delta[ii])
    #if len(x)>0:
        #plt.errorbar(np.median(x), np.median(y), yerr=np.std(y), xerr=np.std(x), fmt='o', color='red')
        
        #xl.append(np.median(x))
        #yl.append(np.median(y))
        #yel.append(np.std(y))

#xl = np.asarray(xl)
#yl = np.asarray(yl)
#yel = np.asarray(yel)

#plt.xlim([-2,8])
#plt.ylim([-2,2])  

#plt.ylabel(r'$\Delta$')
#plt.xlabel(r'$c21w$') 


#index = np.where(delta<0.5)
#c21w = c21w[index]
#delta = delta[index]

#a, b, c  = np.polyfit(c21w,delta, 2)
#xx = np.arange(-1,6,0.01)
##plt.plot(xx, a*(xx**2)+b*xx+c, 'b--')
#print a, b, c

#index = np.where(c21w<3)
#c21w = c21w[index]
#delta = delta[index]


#a, b  = np.polyfit(c21w,delta, 1)
#plt.plot([-1,6], [-a+b,a*6+b], 'g--')
#print a,b 
###a = 0.1659241187077633 
###b = -0.35566041164213746


#a, b  = np.polyfit(xl,yl, 1, w=1./yel**2)
#plt.plot([-1,6], [-a+b,a*6+b], '--', color='black')

#########################################################################  





##a0, b0  = np.polyfit(r_w1, logWimx, 1)
##plt.plot([-1,1], [-a0+b0,a0+b0], 'b--')



#y = []
#x = []
#for i in range(len(r_w1_60)):
    #if logWimx_60[i]<a*r_w1_60[i]+b+0.5 and  logWimx_60[i]>a*r_w1_60[i]+b-0.5:
        #x.append(r_w1_60[i])
        #y.append(logWimx_60[i])

#x=np.asarray(x)
#y=np.asarray(y)

#a0, b0  = np.polyfit(y, x, 1)
#a = 1./a0
#b = -1.*b0/a0

#print a,b 
#plt.plot([-1,1], [-a+b,a+b], 'g--')

#plt.plot(x, y, 'g.')

#print linregress(x,y) #x and y are arrays or lists.




#from sklearn.linear_model import LinearRegression
#model = LinearRegression(fit_intercept=True)
#model.fit(x[:, np.newaxis], y)
#xfit = np.linspace(-1,1, 1000)
#yfit = model.predict(xfit[:, np.newaxis])
#plt.plot(xfit, yfit, '--')


#linear = odr.Model(f)
#mydata = odr.Data(x, y, wd=x*0.+1, we=y*0.+1)
#myodr = odr.ODR(mydata, linear, beta0=[0.4,2.5])
#myoutput = myodr.run()

#m = myoutput.beta[0]
#b = myoutput.beta[1]
#xfit = np.linspace(-1,1, 1000)
#yfit = m*xfit+b
#plt.plot(xfit, yfit, '--')


#for i in range(len(pgc_60)):
   #if pgc_60[i] in PGC:
       #plt.plot(r_w1_60[i], logWimx_60[i], 'b.')
#m = 0.964
#b = 2.349
#xfit = np.linspace(-1,1, 1000)
#yfit = m*xfit+b
#plt.plot(xfit, yfit, 'b--')












