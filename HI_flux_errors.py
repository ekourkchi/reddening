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

########################################################### Begin
########################################################### Begin
inFile  = '../ADHI.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)


pgc    = table['PGC']
Vh_av  = table['Vh_av']
Wmx_av = table['Wmx_av']
eW_av  = table['eW_av']
N_av   = table['N_av']
Wmx1   = table['Wmx1']
e_W1   = table['e_W1']
SN1    = table['SN1']
Flux1  = table['Flux1']
Wmx2   = table['Wmx2']
e_W2   = table['e_W2']
SN2    = table['SN2']
Flux2  = table['Flux2']
Wmx3   = table['Wmx3']
e_W3   = table['e_W3']
SN3    = table['SN3']
Flux3  = table['Flux3']

N = len(pgc)

Wmx  = np.zeros(N)
eWmx = np.zeros(N)
F_av  = np.zeros(N)
NN_av = np.zeros(N)

for i in range(N):
    
    n = 0
    W_tot = 0
    eWmx_tot = 0
    #F_tot = 0
    if e_W1[i]<= 20 and e_W1[i]>=0:
        n+=1
        W_tot+=1.*Wmx1[i]/e_W1[i]**2
        #F_tot+=Flux1[i]
        eWmx_tot+=1./e_W1[i]**2
    if e_W2[i]<= 20 and e_W2[i]>=0:
        n+=1
        W_tot+=1.*Wmx2[i]/e_W2[i]**2
        #F_tot+=Flux2[i]
        eWmx_tot+=1./e_W2[i]**2
    if e_W3[i]<= 20 and e_W3[i]>=0:
        n+=1
        W_tot+=1.*Wmx3[i]/e_W3[i]**2
        #F_tot+=Flux3[i]  
        eWmx_tot+=1./e_W3[i]**2

    if n>0:
        Wmx[i]  = W_tot/eWmx_tot
        #F_av[i]  = F_tot/n
        eWmx[i] = math.sqrt(1./eWmx_tot)
        NN_av[i] = n
        
    n=0
    F_tot = 0
    if Flux1[i]>0:
        n+=1
        F_tot+=Flux1[i]
    if Flux2[i]>0:
        n+=1
        F_tot+=Flux2[i]
    if Flux3[i]>0:
        n+=1
        F_tot+=Flux3[i]
    if n>0: F_av[i]  = 1.*F_tot/n
########################################################### Begin
inFile  = '../ALFALFA100.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

pgc_alfalfa = table['PGC']
F_alfalfa = table['F']
SNR_alfalfa = table['SNR']

########################################################### Begin
inFile  = '../Cornel_HI.csv'
table   = np.genfromtxt(inFile , delimiter='|', filling_values=-1, names=True, dtype=None)

pgc_cornel = table['PGC']
Fc_cornel = table['Fc']

F_std = []
F_av_  = []
for i in range(len(pgc)):
    if pgc[i] in pgc_alfalfa and pgc[i] in pgc_cornel:
        
        ind1, = np.where(pgc_cornel==pgc[i])
        ind2, = np.where(pgc_alfalfa==pgc[i])
        
        if SNR_alfalfa[ind2][0]>=10:
            F = [(Fc_cornel[ind1]+F_alfalfa[ind2])/2., F_av[i]]
            F_av_.append(np.mean(F))
            F_std.append(np.std(F))



F_av = np.asarray(F_av_)
F_std = np.asarray(F_std)

x = np.median(F_std/F_av)
print x
plt.plot(F_av, F_std/F_av, 'r.', alpha=0.2)
 
plt.plot([0,200],[x,x])


plt.ylim([0,2])
plt.show()        
