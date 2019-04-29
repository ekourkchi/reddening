import numpy as np
import emcee
import matplotlib.pyplot as pl
import corner
import scipy.optimize as op
################################################################# 
def lnlike(theta, x, y, xerr, yerr):
    m, b = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + (m*xerr)**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

################################################################# 
def lnprior(theta):
    m, b = theta
    if True:  # don't care (flat prior)
        return 0.0
    return -np.inf
################################################################# 
def lnprob(theta, x, y, xerr, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, xerr, yerr)
################################################################# 
## nwalker: # of MCMC random walkers
## nsteps: # the lenght of MCMC chain
## ignore: how many first steps to ignore
def linMC(x, y, xerr, yerr, nwalkers=100, nsteps=1000, ignore=100):

    [m_guess, b_guess], cov  = np.polyfit(x,y, 1, w=1./yerr**2, cov=True, full = False)
    
    nll = lambda *args: -lnlike(*args)
    # maximum likelihood
    result = op.minimize(nll, [m_guess, b_guess], args=(x, y, xerr, yerr))
    m_ml, b_ml = result["x"]
    
    ndim = 2
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, xerr, yerr))
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:, ignore:, :].reshape((-1, ndim))
    
    m_mcmc, b_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
                             
    return m_mcmc, b_mcmc, samples
#################################################################

def linSimul(samples, xl, size=500, percentile=[16, 50, 84]):

    Y = np.ones(shape = (size,len(xl)))
    i = 0 
    for m, b, in samples[np.random.randint(len(samples), size=size)]:
        Y[i] = m*xl+b
        i+=1

    U = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                zip(*np.percentile(Y, [16, 50, 84],
                                                    axis=0)))
    yl = np.asarray([item[0] for item in U])
    yue = np.asarray([item[1] for item in U])
    yle = np.asarray([item[2] for item in U])   
    
    return yl, yue, yle
    
 #################################################################
   
    
################################################################# 
def lnlike1D(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

################################################################# 
def lnprob1D(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike1D(theta, x, y, yerr)
################################################################# 
## nwalker: # of MCMC random walkers
## nsteps: # the lenght of MCMC chain
## ignore: how many first steps to ignore
def linMC1D(x, y, yerr, nwalkers=100, nsteps=1000, ignore=100):

    [m_guess, b_guess], cov  = np.polyfit(x,y, 1, w=1./yerr**2, cov=True, full = False)
    #m_guess, b_guess = -8, 14
    
    nll = lambda *args: -lnlike1D(*args)
    # maximum likelihood
    result = op.minimize(nll, [m_guess, b_guess], args=(x, y, yerr))
    m_ml, b_ml = result["x"]
    
    ndim = 2
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob1D, args=(x, y, yerr))
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:, ignore:, :].reshape((-1, ndim))
    
    m_mcmc, b_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
                             
    return m_mcmc, b_mcmc, samples
#################################################################


    
    
    
    
    
    
    
    
    
    
    
    
    

