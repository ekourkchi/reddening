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
from string import ascii_letters
import pandas as pd
import seaborn as sns
import matplotlib
import sklearn.datasets as ds
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from redTools import *
from Kcorrect import *


################################################################# 
def add_axis(ax, xlim, ylim):
    
    x1, x2 = xlim[0], xlim[1]
    y1, y2 = ylim[0], ylim[1]
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)

    ax.minorticks_on()
    ax.tick_params(which='major', length=7, width=1.5)
    ax.tick_params(which='minor', length=4, color='#000033', width=1.0)     
    
    # additional Y-axis (on the right)
    y_ax = ax.twinx()
    y_ax.set_ylim(y1, y2)
    y_ax.set_yticklabels([])
    y_ax.minorticks_on()
    y_ax.tick_params(which='major', length=7, width=1.5, direction='in')
    y_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')

    # additional X-axis (on the top)
    x_ax = ax.twiny()
    x_ax.set_xlim(x1, x2)
    x_ax.set_xticklabels([])
    x_ax.minorticks_on()
    x_ax.tick_params(which='major', length=7, width=1.5, direction='in')
    x_ax.tick_params(which='minor', length=4, color='#000033', width=1.0, direction='in')
    
    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(12)   

################################################################# 


#### All but face-on
inFile  = 'ESN_HI_catal.csv'
band1 = 'r'
band2 = 'w2'

## get transformations from non-Face-ON galaxies
scaler, pca = transform(inFile, band1=band1, band2=band2)
################################


#### Face ON
table = getTable(inFile, band1=band1, band2=band2, faceOn=True)

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

print len(logWimx)
data = {'$Log( W_{mx}^i)$':logWimx, '$c21W2$':c21w, '$\mu 50$':mu50}
d = pd.DataFrame.from_dict(data)
z_data = scaler.transform(d)
pca_data = pca.transform(z_data)
################################

fig = py.figure(figsize=(4.5, 11), dpi=100)   
fig.subplots_adjust(hspace=0, top=0.97, bottom=0.07, left=0.20, right=0.95)
gs = gridspec.GridSpec(5,1) 
p = 0


if band2=='w1':
        text1 = r'\overline{'+band1+r'}-\overline{W}1'            # example: cr-W1
        text2 = 'C_{21W1}'              # example: c21w
else: 
        text1 = '\overline{'+band1+'}-\overline{W}2'
        if band1=='w1': text1 = r'\overline{W}1-\overline{W}2'
        text2 = 'C_{21W2}'

pc0 = pca_data[:,0]
pc1 = pca_data[:,1]
pc2 = pca_data[:,2]

data = {'r-w1':r_w1, '$Log( W_{mx}^i)$':logWimx, '$c21W2$':c21w, '$\mu 50$':mu50, 'C82':C82, 'pc0':pc0}
d = pd.DataFrame.from_dict(data)
corr = d.corr()
############################################################################1st TOP panel
ax = plt.subplot(gs[p]) ; p+=1 
a0, b0  = np.polyfit(pc0, r_w1, 1)
delta = np.abs(r_w1-(a0*pc0+b0))
indx = np.where(delta<1)
r_w1_ = r_w1[indx]
pc0_ = pc0[indx]
#ax.plot(r_w1,pc0, 'g.', alpha=0.7)
for i in range(len(r_w1)):
    if c21w[i]<1: p1, = ax.plot(r_w1[i], pc0[i], 'b.', markersize=5, alpha=0.7, label=r"$"+text2+" < 1$")
    if c21w[i]>=1 and c21w[i]< 3:
            p2, = ax.plot(r_w1[i], pc0[i], 'g.', markersize=5, alpha=0.7, label=r"$1 < "+text2+" < 3$")
    if c21w[i]>=3:
            p3, = ax.plot(r_w1[i], pc0[i], 'r.', markersize=5, alpha=0.7, label=r"$3 < "+text2+"$")   
a0, b0  = np.polyfit(pc0_, r_w1_, 1)
y = np.linspace(-5,5,50)
x = a0*y+b0
ax.plot(x,y, 'k--')

ax.set_ylabel('$PC0$', fontsize=15, labelpad=7)
add_axis(ax,[-2,2],[-4.2,4.2])
plt.setp(ax.get_xticklabels(), visible=False)
lns = [p1, p2, p3]
ax.legend(handles=lns, loc=1, fontsize=10)

delta = np.abs(r_w1_-(a0*pc0_+b0))
rms = np.sqrt(np.mean(np.square(delta)))
Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
x0 = 0.45*Xlm[0]+0.55*Xlm[1]
y0 = 0.65*Ylm[0]+0.35*Ylm[1]
ax.text(x0,y0, r'$RMS=$'+'%.2f'%rms+' mag', fontsize=12, color='k')

x0 = 0.45*Xlm[0]+0.55*Xlm[1]
y0 = 0.80*Ylm[0]+0.20*Ylm[1]
ax.text(x0,y0, r'$Corr.=$'+'%.2f'%corr['r-w1']['pc0'], fontsize=12, color='k')
############################################################################2nd panel
ax = plt.subplot(gs[p]) ; p+=1

a0, b0  = np.polyfit(mu50, r_w1, 1)
delta = np.abs(r_w1-(a0*mu50+b0))
indx = np.where(delta<1.)
r_w1_ = r_w1[indx]
mu50_ = mu50[indx]
#ax.plot(r_w1,mu50, 'g.', alpha=0.7)
for i in range(len(r_w1)):
    if c21w[i]<1: p1, = ax.plot(r_w1[i], mu50[i], 'b.', markersize=5, alpha=0.7, label=r"$"+text2+" < 1$")
    if c21w[i]>=1 and c21w[i]< 3:
            p2, = ax.plot(r_w1[i], mu50[i], 'g.', markersize=5, alpha=0.7, label=r"$1 < "+text2+" < 3$")
    if c21w[i]>=3:
            p3, = ax.plot(r_w1[i], mu50[i], 'r.', markersize=5, alpha=0.7, label=r"$3 < "+text2+"$") 
a0, b0  = np.polyfit(mu50_, r_w1_, 1)
y = np.linspace(19,27,50)
x = a0*y+b0
ax.plot(x,y, 'k--')

add_axis(ax,[-2,2],[26.8,19.2])
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_ylabel(r'$\langle \mu \rangle_e$', fontsize=15, labelpad=7)

delta = np.abs(r_w1_-(a0*mu50_+b0))
rms = np.sqrt(np.mean(np.square(delta)))
Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
x0 = 0.45*Xlm[0]+0.55*Xlm[1]
y0 = 0.65*Ylm[0]+0.35*Ylm[1]
ax.text(x0,y0, r'$RMS=$'+'%.2f'%rms+' mag', fontsize=12, color='k')

x0 = 0.45*Xlm[0]+0.55*Xlm[1]
y0 = 0.80*Ylm[0]+0.20*Ylm[1]
ax.text(x0,y0, r'$Corr.=$'+'%.2f'%corr['r-w1'][u'$\mu 50$'], fontsize=12, color='k')

############################################################################3rd TOP panel
ax = plt.subplot(gs[p]) ; p+=1


a0, b0  = np.polyfit(c21w, r_w1, 1)
delta = np.abs(r_w1-(a0*c21w+b0))
indx = np.where(delta<1.)
r_w1_ = r_w1[indx]
c21w_ = c21w[indx]
#ax.plot(r_w1,c21w, 'g.', alpha=0.7)
for i in range(len(r_w1)):
    if c21w[i]<1: p1, = ax.plot(r_w1[i], c21w[i], 'b.', markersize=5, alpha=0.7, label=r"$"+text2+" < 1$")
    if c21w[i]>=1 and c21w[i]< 3:
            p2, = ax.plot(r_w1[i], c21w[i], 'g.', markersize=5, alpha=0.7, label=r"$1 < "+text2+" < 3$")
    if c21w[i]>=3:
            p3, = ax.plot(r_w1[i], c21w[i], 'r.', markersize=5, alpha=0.7, label=r"$3 < "+text2+"$") 
a0, b0  = np.polyfit(c21w_, r_w1_, 1)
y = np.linspace(-2,7,50)
x = a0*y+b0
ax.plot(x,y, 'k--')

add_axis(ax,[-2,2],[-1.5,7])
ax.set_ylabel('$C_{21W2}$', fontsize=16, labelpad=7)
plt.setp(ax.get_xticklabels(), visible=False)
delta = np.abs(r_w1_-(a0*c21w_+b0))
rms = np.sqrt(np.mean(np.square(delta)))
Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
x0 = 0.45*Xlm[0]+0.55*Xlm[1]
y0 = 0.65*Ylm[0]+0.35*Ylm[1]
ax.text(x0,y0, r'$RMS=$'+'%.2f'%rms+' mag', fontsize=12, color='k')

x0 = 0.45*Xlm[0]+0.55*Xlm[1]
y0 = 0.80*Ylm[0]+0.20*Ylm[1]
ax.text(x0,y0, r'$Corr.=$'+'%.2f'%corr['r-w1'][u'$c21W2$'], fontsize=12, color='k')

#labels=ax.get_yticks().tolist()
#for i in range(len(labels)): labels[i] = str(np.int(labels[i]))
#labels[4] = ''; 
#ax.set_yticklabels(labels)
############################################################################4 TOP panel
ax = plt.subplot(gs[p]) ; p+=1

a0, b0  = np.polyfit(logWimx, r_w1, 1)
delta = np.abs(r_w1-(a0*logWimx+b0))
indx = np.where(delta<1.)
r_w1_ = r_w1[indx]
logWimx_ = logWimx[indx]
#ax.plot(r_w1,logWimx, 'g.', alpha=0.7)
for i in range(len(r_w1)):
    if c21w[i]<1: p1, = ax.plot(r_w1[i], logWimx[i], 'b.', markersize=5, alpha=0.7, label=r"$"+text2+" < 1$")
    if c21w[i]>=1 and c21w[i]< 3:
            p2, = ax.plot(r_w1[i], logWimx[i], 'g.', markersize=5, alpha=0.7, label=r"$1 < "+text2+" < 3$")
    if c21w[i]>=3:
            p3, = ax.plot(r_w1[i], logWimx[i], 'r.', markersize=5, alpha=0.7, label=r"$3 < "+text2+"$")         
a0, b0  = np.polyfit(logWimx_, r_w1_, 1)
y = np.linspace(1,3,50)
x = a0*y+b0
ax.plot(x,y, 'k--')

add_axis(ax,[-2,2],[1.6,2.9])
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_ylabel('$Log( W_{mx}^i)$', fontsize=16, labelpad=7)

delta = np.abs(r_w1_-(a0*logWimx_+b0))
rms = np.sqrt(np.mean(np.square(delta)))
Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
x0 = 0.45*Xlm[0]+0.55*Xlm[1]
y0 = 0.65*Ylm[0]+0.35*Ylm[1]
ax.text(x0,y0, r'$RMS=$'+'%.2f'%rms+' mag', fontsize=12, color='k')

x0 = 0.45*Xlm[0]+0.55*Xlm[1]
y0 = 0.80*Ylm[0]+0.20*Ylm[1]
ax.text(x0,y0, r'$Corr.=$'+'%.2f'%corr['r-w1'][u'$Log( W_{mx}^i)$'], fontsize=12, color='k')

#labels=ax.get_yticks().tolist()
#labels[0] = ''; 
#ax.set_yticklabels(labels)

############################################################################5th TOP panel
ax = plt.subplot(gs[p]) ; p+=1

for i in range(len(r_w1)):
    if c21w[i]<1: p1, = ax.plot(r_w1[i], C82[i], 'b.', markersize=5, alpha=0.7, label=r"$"+text2+" < 1$")
    if c21w[i]>=1 and c21w[i]< 3:
            p2, = ax.plot(r_w1[i], C82[i], 'g.', markersize=5, alpha=0.7, label=r"$1 < "+text2+" < 3$")
    if c21w[i]>=3:
            p3, = ax.plot(r_w1[i], C82[i], 'r.', markersize=5, alpha=0.7, label=r"$3 < "+text2+"$") 

add_axis(ax,[-2,2],[1.5,8.8])
Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
x0 = 0.45*Xlm[0]+0.55*Xlm[1]
y0 = 0.80*Ylm[0]+0.20*Ylm[1]
ax.text(x0,y0, r'$Corr.=$'+'%.2f'%corr['r-w1']['C82'], fontsize=12, color='k')
ax.set_xlabel('$'+text1+'$', fontsize=16, labelpad=7)
ax.set_ylabel('$C_{82}$', fontsize=16, labelpad=7)


#plt.show()
plt.savefig('r_w2_features_Fon.eps', dpi=600)
