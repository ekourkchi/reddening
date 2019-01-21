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

from redTools import *
from Kcorrect import *
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
inFile  = 'ESN_HI_catal.csv'
table   = np.genfromtxt(inFile , delimiter=',', filling_values=-1, names=True, dtype=None)

table = extinctionCorrect(table)
table = Kcorrection(table)

band1 = 'r'
band2 = 'w1'

if band2=='w1':
    text1 = band1+'-W1'            # example: cr-W1
    text2 = '$c21W_1$'              # example: c21w
else: 
    text1 = band1+'-W2' 
    text2 = '$c21W_2$'


table['c21w'] = table['m21'] - table[band2]
table['r_w1'] = table[band1] - table[band2]
r_w1 = table['r_w1']
c21w = table['c21w']

pgc = table['pgc']
logWimx = table['logWimx']

inc = table['inc']
r = table['r']
w1 = table['w1']

flag = table['flag']
Sqlt = table['Sqlt']
Wqlt = table['Wqlt']

text1 = 'cr-w1'
text2 = r'$c21w$'

########################################################### END

r_w1 = r-w1


#index = np.where(inc>45)
index = np.where(flag<3)
r_w1 = r_w1[index]
logWimx = logWimx[index]
pgc = pgc[index]
c21w = c21w[index]
flag = flag[index]
inc = inc[index]

#index = np.where(inc<55)
index = np.where(flag>0)
r_w1 = r_w1[index]
logWimx = logWimx[index]
pgc = pgc[index]
c21w = c21w[index]
flag = flag[index]
inc = inc[index]


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



a0, b0  = np.polyfit(logWimx,r_w1, 1)
a = 1./a0
b = -1.*b0/a0
print a,b 

#########################################################################  

fig = py.figure(figsize=(11, 5), dpi=100)   
fig.subplots_adjust(wspace=0.2, top=0.95, bottom=0.1, left=0.07, right=0.98)
gs = gridspec.GridSpec(1, 2) 

ax = plt.subplot(gs[0]) 
ax.plot([-1,4], [-a+b,a*4+b], 'g--')


delta = r_w1-(a0*logWimx+b0)
print 'sigma2 = ', np.sqrt(np.median(delta**2))

for i in range(len(pgc)):
   if c21w[i]<1  :
       ax.plot(r_w1[i], logWimx[i], 'b.', markersize=5, alpha=1)
   if c21w[i]>=1 and c21w[i]< 2:
       ax.plot(r_w1[i], logWimx[i], 'g.', markersize=5, alpha=1)  
   if c21w[i]>=2 and c21w[i]< 3:
       ax.plot(r_w1[i], logWimx[i], '.', color='green', markersize=5, alpha=1)         
   if c21w[i]>=3:
       ax.plot(r_w1[i], logWimx[i], 'r.', markersize=5, alpha=1)    
       
ax.set_xlabel(text1)
ax.set_ylabel(r'$Log( W_{mx}^i)$')

ax.set_xlim([-2,2])
ax.set_ylim([1.7,3])       
       
#########################################################################  




#########################################################################
ax = plt.subplot(gs[1])

delta = r_w1-(a0*logWimx+b0)
ax.plot(c21w, delta, '.', color='black', markersize=3, alpha=0.5)
xl = []
yl= []
yel=[]
for i in np.arange(-1,5,0.3):
    
    x = []
    y = []
    for ii in range(len(c21w)):
        xi = c21w[ii]
        if xi>=i and xi<i+1:
            x.append(xi)
            y.append(delta[ii])
    if len(x)>0:
        #ax.errorbar(np.median(x), np.median(y), yerr=np.std(y), xerr=np.std(x), fmt='o', color='red')
        
        xl.append(np.median(x))
        yl.append(np.median(y))
        yel.append(np.std(y))

xl = np.asarray(xl)
yl = np.asarray(yl)
yel = np.asarray(yel)

plt.xlim([-2,8])
plt.ylim([-1.5,1.5])  

plt.ylabel(r'$\Delta$'+'('+text1+')')
plt.xlabel(text2) 


#index = np.where(delta<0.5)
#c21w = c21w[index]
#delta = delta[index]

a, b, c  = np.polyfit(c21w,delta, 2)
xx = np.arange(-1,10,0.01)
#ax.plot(xx, a*(xx**2)+b*xx+c, 'r-')
print a, b, c


for i in np.arange(-1,5,1):
    
    x = []
    y = []
    for ii in range(len(c21w)):
        xi = c21w[ii]
        if xi>=i and xi<i+1:
            x.append(xi)
            y.append(delta[ii])
    #if len(x)>0:
        #ax.errorbar(np.median(x), np.median(y), yerr=np.std(y), xerr=np.std(x), fmt='o', color='red')

ax.plot([-2,8], [0,0], 'k:')



index = np.where(delta<0.5)
c21w_  = c21w[index]
delta_ = delta[index]
a, b, c  = np.polyfit(c21w_,delta_, 2)
xx = np.arange(-2,8,0.01)
plt.plot(xx, a*(xx**2)+b*xx+c, 'r--')


#########################################################################  
#X = np.atleast_2d(xl).T
#y = yl
#dy = yel


X = np.atleast_2d(c21w).T
y = delta
dy = 0.1*np.ones_like(y)



## ????
#kernel = C(1.0, (1e-3, 1e3)) * RBF(5, (1e-3, 1e3))
#kernel = ConstantKernel() + Matern(length_scale=3, nu=3/2) + WhiteKernel(noise_level=0.1)


                              
kernel = ConstantKernel() + 1.0 * RBF(100)+ WhiteKernel(noise_level=1.)

gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2)#,
                              #n_restarts_optimizer=10)
gp.fit(X, y)


x = np.atleast_2d(np.linspace(-2,8, 2000)).T
y_pred, sigma = gp.predict(x, return_std=True)



plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.000 * sigma,   # 1.9600
                        (y_pred + 1.000 * sigma)[::-1]]),
         alpha=.3, fc='b', ec='None', label='95% confidence interval')


#########################################################################  

plt.show()




########################################################### 4Fati
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



index = np.where(inc>45)
r_w1 = r_w1[index]
logWimx = logWimx[index]
c21w = c21w[index]
flag = flag[index]
inc = inc[index]
pgc = pgc[index]
Sqlt = Sqlt[index]
Wqlt = Wqlt[index]

index = np.where(Sqlt>3)
r_w1 = r_w1[index]
logWimx = logWimx[index]
c21w = c21w[index]
flag = flag[index]
inc = inc[index]
pgc = pgc[index]
Sqlt = Sqlt[index]
Wqlt = Wqlt[index]

index = np.where(Wqlt>3)
r_w1 = r_w1[index]
logWimx = logWimx[index]
c21w = c21w[index]
flag = flag[index]
inc = inc[index]
pgc = pgc[index]
Sqlt = Sqlt[index]
Wqlt = Wqlt[index]

index = np.where(logWimx>1)
r_w1 = r_w1[index]
logWimx = logWimx[index]
c21w = c21w[index]
flag = flag[index]
inc = inc[index]
pgc = pgc[index]
Sqlt = Sqlt[index]
Wqlt = Wqlt[index]

index = np.where(r_w1<4)
r_w1 = r_w1[index]
logWimx = logWimx[index]
c21w = c21w[index]
flag = flag[index]
inc = inc[index]
Sqlt = Sqlt[index]
Wqlt = Wqlt[index]

x = np.atleast_2d(c21w).T
delta_pred, sigma = gp.predict(x, return_std=True)
r_w1 = r_w1-delta_pred
R = r_w1-(a0*logWimx+b0)


myTable = Table()
myTable.add_column(Column(data=pgc, name='pgc'))
myTable.add_column(Column(data=logWimx, name='logWimx'))
myTable.add_column(Column(data=r_w1, name='r_w1'))
myTable.add_column(Column(data=c21w, name='c21w'))
myTable.add_column(Column(data=inc, name='inc'))
myTable.add_column(Column(data=R, name='reddening'))

#myTable.write('reddening_v01.csv', format='ascii.fixed_width',delimiter=',', bookend=False, overwrite=True) 
########################################################### END









