import sys
import time
import os
import subprocess
import math
import numpy as np
from astropy.table import Table, Column 

from redTools import *
from Kcorrect import *

################################################################# 

band2 = 'w2'

band1 = 'g'
_, Input, T = getBand('ESN_HI_catal.csv', band1=band1 , band2=band2)
pc0     = Input[2]
inc     = Input[3]
table = T[5]
pgc_g = table['pgc']
a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1=band1, band2=band2)
A_g = log_a_b(inc, 10**(-1.*theta))*(a*pc0**3+b*pc0**2+c*pc0+d)
P0_g = pc0
inc_g = inc

band1 = 'r'
_, Input, T = getBand('ESN_HI_catal.csv', band1=band1 , band2=band2)
pc0     = Input[2]
inc     = Input[3]
table = T[5]
pgc_r = table['pgc']
a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1=band1, band2=band2)
A_r = log_a_b(inc, 10**(-1.*theta))*(a*pc0**3+b*pc0**2+c*pc0+d)
P0_r = pc0
inc_r = inc
tbl = table

band1 = 'i'
_, Input, T = getBand('ESN_HI_catal.csv', band1=band1 , band2=band2)
pc0     = Input[2]
inc     = Input[3]
table = T[5]
pgc_i = table['pgc']
a,b,c,d, alpha, beta, theta, Ealpha, Ebeta = getReddening_params(band1=band1, band2=band2)
A_i = log_a_b(inc, 10**(-1.*theta))*(a*pc0**3+b*pc0**2+c*pc0+d)
P0_i = pc0
inc_i = inc

PGC = []
INC = []
P = []
U =[]; G=[]; R=[]; I=[]; Z=[]; W1=[]; W2=[]
m21 = []
logWimx = []
mu50 = []
R50_g = []
R50_r = []
R50_i = []
R50_z = []
R50_w1 = []
R50_w2 = []
Sba = []
Wba = []
Ty = []


for i in range(len(pgc_r)):
    
    if pgc_r[i] in pgc_g and pgc_r[i] in pgc_i:
        idx_g = np.where(pgc_g==pgc_r[i])
        idx_i = np.where(pgc_i==pgc_r[i])
        P0 = np.mean([P0_g[idx_g][0], P0_r[i], P0_i[idx_i][0]])
        PGC.append(pgc_r[i])
        INC.append(inc_r[i])
        P.append(P0)
        U.append(tbl['u'][i])
        G.append(tbl['g'][i])
        R.append(tbl['r'][i])
        I.append(tbl['i'][i])
        Z.append(tbl['z'][i])
        W1.append(tbl['w1'][i])
        W2.append(tbl['w2'][i])
        m21.append(tbl['m21'][i])
        logWimx.append(tbl['logWimx'][i])
        mu50.append(tbl['mu50'][i])
        R50_g.append(tbl['R50_g'][i])
        R50_r.append(tbl['R50_r'][i])
        R50_i.append(tbl['R50_i'][i])
        R50_z.append(tbl['R50_z'][i])
        R50_w1.append(tbl['R50_w1'][i])
        R50_w2.append(tbl['R50_w2'][i])
        Sba.append(tbl['Sba'][i])
        Wba.append(tbl['Wba'][i])
        Ty.append(tbl['Ty'][i])
        

PGC=np.asarray(PGC)
INC=np.asarray(INC)
P=np.asarray(P)
U=np.asarray(U)
G=np.asarray(G)
R=np.asarray(R)
I=np.asarray(I)
Z=np.asarray(Z)
W1=np.asarray(W1)
W2=np.asarray(W2)
m21 =np.asarray(m21)
logWimx = np.asarray(logWimx)
mu50 = np.asarray(mu50)
R50_g = np.asarray(R50_g)
R50_r = np.asarray(R50_r)
R50_i = np.asarray(R50_i)
R50_z = np.asarray(R50_z)
R50_w1 = np.asarray(R50_w1)
R50_w2 = np.asarray(R50_w2)
Sba = np.asarray(Sba)
Wba = np.asarray(Wba)
Ty = np.asarray(Ty)

myTable = Table()
myTable.add_column(Column(data=PGC, name='pgc'))
myTable.add_column(Column(data=INC, name='inc'))
myTable.add_column(Column(data=P, name='pc0', format='%0.3f'))
myTable.add_column(Column(data=U, name='u', format='%0.3f'))
myTable.add_column(Column(data=G, name='g', format='%0.3f'))
myTable.add_column(Column(data=R, name='r', format='%0.3f'))
myTable.add_column(Column(data=I, name='i', format='%0.3f'))
myTable.add_column(Column(data=Z, name='z', format='%0.3f'))
myTable.add_column(Column(data=W1, name='w1', format='%0.3f'))
myTable.add_column(Column(data=W2, name='w2', format='%0.3f'))
myTable.add_column(Column(data=logWimx, name='logWimx', format='%0.3f'))
myTable.add_column(Column(data=m21, name='m21', format='%0.2f'))
myTable.add_column(Column(data=mu50, name='mu50', format='%0.3f'))
myTable.add_column(Column(data=R50_g, name='R50_g', format='%0.2f'))
myTable.add_column(Column(data=R50_r, name='R50_r', format='%0.2f'))
myTable.add_column(Column(data=R50_i, name='R50_i', format='%0.2f'))
myTable.add_column(Column(data=R50_z, name='R50_z', format='%0.2f'))
myTable.add_column(Column(data=R50_w1, name='R50_w1', format='%0.2f'))
myTable.add_column(Column(data=R50_w2, name='R50_w2', format='%0.2f'))
myTable.add_column(Column(data=Sba, name='Sba', format='%0.2f'))
myTable.add_column(Column(data=Wba, name='Wba', format='%0.2f'))
myTable.add_column(Column(data=Ty, name='Ty', format='%0.1f'))


myTable.write('ESN_INC_P0_MAG.csv', format='ascii.fixed_width',delimiter=',', bookend=False, overwrite=True)   
