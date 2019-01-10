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
def Kcorrect(table, band='r'):
    
    c = 299792.458 # km/s
    
    redshift = table['Vhel']/c
    
    u = table['u']
    g = table['g']
    r = table['r']
    i = table['i']
    z = table['z']
    
    if band=='u':
        k1 = calc_kcor_list('u', redshift, 'u - r', u-r)
        k2 = calc_kcor_list('u', redshift, 'u - i', u-i)
        k3 = calc_kcor_list('u', redshift, 'u - z', u-z)
        return (k1+k2+k3)/3.
    elif band=='g':
        k1 = calc_kcor_list('g', redshift, 'g - i', g-i)
        k2 = calc_kcor_list('g', redshift, 'g - z', g-z)
        return (k1+k2)/2. 
    elif band=='r':
        k1 = calc_kcor_list('r', redshift, 'g - r', g-r)
        return k1       
    elif band=='i':
        k1 = calc_kcor_list('i', redshift, 'g - i', g-i)
        return k1       
    elif band=='z':
        k1 = calc_kcor_list('z', redshift, 'g - z', g-z)
        k2 = calc_kcor_list('z', redshift, 'r - z', r-z)
        return (k1+k2)/2.   
    elif band=='w1' or band=='w2':
        return -2.27*redshift    # Huang et al. 2007 Fig. 6

###########################################################
def Kcorrection(table):
    
    # Chilingarian et al. 2010
    table['u']-= Kcorrect(table, band='u')
    table['g']-= Kcorrect(table, band='g')
    table['r']-= Kcorrect(table, band='r')
    table['i']-= Kcorrect(table, band='i')
    table['z']-= Kcorrect(table, band='z')
    table['w1']-= Kcorrect(table, band='w1')
    table['w2']-= Kcorrect(table, band='w2')
    table['u_']-= Kcorrect(table, band='u')
    table['g_']-= Kcorrect(table, band='g')
    table['r_']-= Kcorrect(table, band='r')
    table['i_']-= Kcorrect(table, band='i')
    table['z_']-= Kcorrect(table, band='z')
    table['w1_']-= Kcorrect(table, band='w1')
    table['w2_']-= Kcorrect(table, band='w2')
    
    return table
