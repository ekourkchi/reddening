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

    A, cov  = np.polyfit(x,y, 1, w=1./yerr**2, cov=True, full = False)
    
    m_guess = A[0]
    b_guess = A[1]
    
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

    A   = np.polyfit(x,y, 1, w=1./yerr**2, cov=False, full = False)
    #m_guess, b_guess = -8, 14
    m_guess = A[0]
    b_guess = A[1]
    
        
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

################################################################# 
def lnlikelikeS1(theta, x, y, xerr, yerr):
    b = theta
    model = x + b
    inv_sigma2 = 1.0/(yerr**2 + (xerr)**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

################################################################# 
def lnprobS1(theta, x, y, xerr, yerr):
    lp = lnpriorS1(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelikeS1(theta, x, y, xerr, yerr)


def lnpriorS1(theta):
    b = theta
    if True:  # don't care (flat prior)
        return 0.0
    return -np.inf
################################################################# 
## nwalker: # of MCMC random walkers
## nsteps: # the lenght of MCMC chain
## ignore: how many first steps to ignore
def linMCSlope1(x, y, xerr, yerr, nwalkers=100, nsteps=1000, ignore=100):

    A, cov  = np.polyfit(x,y, 1, w=1./yerr**2, cov=True, full = False)
    
    b_guess = A[0]
    
    nll = lambda *args: -lnlikelikeS1(*args)
    # maximum likelihood
    result = op.minimize(nll, [b_guess], args=(x, y, xerr, yerr))
    b_ml = result["x"]
    
    ndim = 1
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobS1, args=(x, y, xerr, yerr))
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:, ignore:, :].reshape((-1, ndim))
    
    b_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
                             
    return b_mcmc, samples
#################################################################
    
    
    
    
################################################################# 
def lnlike_H0(theta, mu, V, mu_err, V_err):
    H0 = theta
    
    D = 10**((mu-25.)/5.)
    
    model = H0 * D
    d_model = H0 * D * (np.log(10)/5.) * mu_err
    
    inv_sigma2 = 1.0/(V_err**2 + (d_model)**2)
    return -0.5*(np.sum((V-model)**2*inv_sigma2 - np.log(inv_sigma2)))

################################################################# 
def lnprior_H0(theta):
    H0 = theta
    if True:  # don't care (flat prior)
        return 0.0
    return -np.inf
################################################################# 
def lnprob_H0(theta, x, y, xerr, yerr):
    lp = lnprior_H0(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_H0(theta, x, y, xerr, yerr)
################################################################# 
## nwalker: # of MCMC random walkers
## nsteps: # the lenght of MCMC chain
## ignore: how many first steps to ignore
def linMC_H0(mu, V, mu_err, V_err, nwalkers=100, nsteps=1000, ignore=100):
    
    
    H0 = 68.
    
    nll = lambda *args: -lnlike_H0(*args)
    # maximum likelihood 
    result = op.minimize(nll, [H0], args=(mu, V, mu_err, V_err))
    
    ndim = 1
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_H0, args=(mu, V, mu_err, V_err))
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:, ignore:, :].reshape((-1, ndim))
    
    H0 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
                             
    return H0, samples
#################################################################    
    
    
    
    
    
    
    

