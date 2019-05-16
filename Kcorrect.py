#!/usr/bin/python
# encoding=utf8
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import os
import subprocess
import math
import matplotlib.pyplot as plt
import numpy as np
import pylab as py

from calc_kcor import *

###########################################################
def calc_kcor_list(band, redshift, color_name, color_value):
    
    N = len(redshift)
    K_lst = np.zeros(N)
    
    for i in range(N):
        K_lst[i] = calc_kcor(band, redshift[i], color_name, color_value[i])
        
    return K_lst

###########################################################
def Kcorrect(table, band='r', corr=False):
    
    c = 299792.458 # km/s
    
    redshift = table['Vhel']/c
    
    if not corr:
        u = table['u']
        g = table['g']
        r = table['r']
        i = table['i']
        z = table['z']
    else: 
        u = table['u_']
        g = table['g_']
        r = table['r_']
        i = table['i_']
        z = table['z_']        
    
    
    if band=='u':
        k1 = calc_kcor_list('u', redshift, 'u - r', u-r)
        k2 = calc_kcor_list('u', redshift, 'u - i', u-i)
        k3 = calc_kcor_list('u', redshift, 'u - z', u-z)
        
        N = len(k1)
        k0 = np.zeros(N)
        for jj in range(N):
            if u[jj]>0: 
                if r[jj]>0 and i[jj]>0 and z[jj]>0:
                    k0[jj]=np.median([k1[jj], k2[jj], k3[jj]])
                elif r[jj]>0 and i[jj]>0 and z[jj]<=0:
                    k0[jj]=np.median([k1[jj], k2[jj]])
                elif r[jj]>0 and i[jj]<=0 and z[jj]>0:
                    k0[jj]=np.median([k1[jj], k3[jj]])
                elif r[jj]<=0 and i[jj]>0 and z[jj]>0:
                    k0[jj]=np.median([k2[jj], k3[jj]])
                elif r[jj]>0 and i[jj]<=0 and z[jj]<=0:
                    k0[jj]= k1[jj]
                elif r[jj]<=0 and i[jj]<=0 and z[jj]>0:
                    k0[jj]= k3[jj]
                elif r[jj]<=0 and i[jj]>0 and z[jj]<=0:
                    k0[jj]= k2[jj]
        return k0
    elif band=='g':
        k1 = calc_kcor_list('g', redshift, 'g - i', g-i)
        k2 = calc_kcor_list('g', redshift, 'g - z', g-z)
        k3 = calc_kcor_list('g', redshift, 'g - r', g-r)
        
        N = len(k1)
        k0 = np.zeros(N)
        for jj in range(N):
            if g[jj]>0:         
                if r[jj]>0 and i[jj]>0 and z[jj]>0:
                    k0[jj]=np.median([k1[jj], k2[jj], k3[jj]])
                elif r[jj]>0 and i[jj]>0 and z[jj]<=0:
                    k0[jj]=np.median([k1[jj], k3[jj]])
                elif r[jj]>0 and i[jj]<=0 and z[jj]>0:
                    k0[jj]=np.median([k2[jj], k3[jj]])
                elif r[jj]<=0 and i[jj]>0 and z[jj]>0:
                    k0[jj]=np.median([k1[jj], k2[jj]])
                elif r[jj]>0 and i[jj]<=0 and z[jj]<=0:
                    k0[jj]= k3[jj]
                elif r[jj]<=0 and i[jj]<=0 and z[jj]>0:
                    k0[jj]= k2[jj]
                elif r[jj]<=0 and i[jj]>0 and z[jj]<=0:
                    k0[jj]= k1[jj]                
        return k0
    
    elif band=='r':
        k1 = calc_kcor_list('r', redshift, 'g - r', g-r)
        k2 = calc_kcor_list('r', redshift, 'u - r', u-r)

        N = len(k1)
        k0 = np.zeros(N)
        for jj in range(N):
            if r[jj]>0:         
                if g[jj]>0:
                    k0[jj]=k1[jj]
                elif u[jj]>0:
                    k0[jj]=k2[jj]
                               
        return k0      
    
    elif band=='i':
        k1 = calc_kcor_list('i', redshift, 'g - i', g-i)
        k2 = calc_kcor_list('i', redshift, 'u - i', u-i)

        N = len(k1)
        k0 = np.zeros(N)
        for jj in range(N):
            if i[jj]>0:         
                if g[jj]>0:
                    k0[jj]=k1[jj]
                elif u[jj]>0:
                    k0[jj]=k2[jj]
                               
        return k0        
   
    elif band=='z':
        k1 = calc_kcor_list('z', redshift, 'g - z', g-z)
        k2 = calc_kcor_list('z', redshift, 'r - z', r-z)
        k3 = calc_kcor_list('z', redshift, 'u - z', u-z)

        N = len(k1)
        k0 = np.zeros(N)
        for jj in range(N):
            if z[jj]>0:         
                if g[jj]>0 and r[jj]>0:
                    k0[jj]=np.median([k1[jj], k2[jj]])
                elif g[jj]>0 and r[jj]<=0:
                    k0[jj]=k1[jj]
                elif g[jj]<=0 and r[jj]>0:
                    k0[jj]=k2[jj]
                elif u[jj]>0:
                    k0[jj]=k3[jj]
                               
        return k0  
    
    elif band=='w1' or band=='w2':
        return -2.27*redshift    # Huang et al. 2007 Fig. 6

###########################################################
def Kcorrection(table):
    
    # Chilingarian et al. 2010
    d_u = Kcorrect(table, band='u')
    d_g = Kcorrect(table, band='g')
    d_r = Kcorrect(table, band='r')
    d_i = Kcorrect(table, band='i')
    d_z = Kcorrect(table, band='z')
    d_w1 = Kcorrect(table, band='w1')
    d_w2 = Kcorrect(table, band='w2')
    table['u']-= d_u
    table['g']-= d_g
    table['r']-= d_r
    table['i']-= d_i
    table['z']-= d_z
    table['w1']-= d_w1
    table['w2']-= d_w2
    
    
    d_u = Kcorrect(table, band='u', corr=True)
    d_g = Kcorrect(table, band='g', corr=True)
    d_r = Kcorrect(table, band='r', corr=True)
    d_i = Kcorrect(table, band='i', corr=True)
    d_z = Kcorrect(table, band='z', corr=True)
    d_w1 = Kcorrect(table, band='w1', corr=True)
    d_w2 = Kcorrect(table, band='w2', corr=True)    
    table['u_']-= d_u
    table['g_']-= d_g
    table['r_']-= d_r
    table['i_']-= d_i
    table['z_']-= d_z
    table['w1_']-= d_w1
    table['w2_']-= d_w2
    
    return table
