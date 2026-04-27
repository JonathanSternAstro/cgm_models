import sys, pdb, string,os, glob
import numpy as np
from numpy import log10 as log, log as ln
from scipy import interpolate
import astropy, astropy.cosmology
from astropy import units as un, constants as cons
from astropy.cosmology import Planck15 as cosmo
import pylab as pl, matplotlib
from matplotlib import cm

matplotlib.rc('font', family='serif', size=10)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True

### model parameters
model_name='fiducial' # no '_' in name!
alpha = 2.3
mach=5.6
Z2Zsun = lambda r2Rvir: 0.3 * (r2Rvir/0.2)**-0.6
f_gas = 0.6
Rvir = 140*un.kpc
z = 1.5
b = 1
X = 0.7

def mean_log_n_volume_weighted(r2Rvir, z,f_gas=f_gas, alpha=alpha,X=X):
    x_BN = cosmo.Om(z) - 1
    Delta_c = 18 * np.pi ** 2 + 82 * x_BN - 39 * x_BN ** 2
    rho_Rvir = (1-alpha/3)*f_gas*Delta_c*cosmo.Ob0/cosmo.Om0*cosmo.critical_density(z)
    rho = rho_Rvir * r2Rvir**-alpha
    nH = X*rho/cons.m_p
    return log(nH.to('cm**-3').value)


def sigma_s(mach=mach,b=b):
    return ln(1+b**2*mach**2)**0.5

def PDF_volume_weighted(log_n,mean_log_n_volume_weighted,sigma_s): #per bin size of 1 in natural logarithm
    s = (log_n - mean_log_n_volume_weighted)*ln(10)
    mu_s = -sigma_s**2/2.
    return (2*np.pi)**-0.5/sigma_s * np.e**(-(s-mu_s)**2/(2*sigma_s**2))

all_log_ns = np.linspace(-7,3,101)
dln_ns = 0.1 * ln(10)
all_r2Rvirs = 10.**np.linspace(-1.3,0.5,18001)
log_ns_v, r2Rvirs_v = np.meshgrid(all_log_ns, all_r2Rvirs)
r2Rvir_midbins = (r2Rvirs_v[1:,:]+r2Rvirs_v[:-1,:])/2.
log_ns_v_midbins = (log_ns_v[1:,:]+log_ns_v[:-1,:])/2.
nHs = 10.**log_ns_v_midbins*un.cm**-3


weights_lognormal_volume_weighted = PDF_volume_weighted(log_ns_v_midbins,mean_log_n_volume_weighted(r2Rvir_midbins,z),sigma_s(mach))

Rperp2Rvirs = 10.**np.linspace(-1.,0.3,41)

Rperp2Rvir=Rperp2Rvirs[9] #0.2
weights_los       = 2 * ((r2Rvirs_v[1:,:]**2 - Rperp2Rvir**2)**0.5 - (r2Rvirs_v[:-1,:]**2 - Rperp2Rvir**2)**0.5) * Rvir
weights_los[np.isnan(weights_los)] = 0
dNs = (nHs*weights_lognormal_volume_weighted*weights_los).sum(axis=0).to('cm**-2').value * dln_ns
pl.plot(np.cumsum(dNs),all_log_ns,label='%.2f'%Rperp2Rvir,c='k')
pl.xlabel(r'$N_{\rm H}\ [{\rm cm}^{-2}]$')
pl.ylabel(r'$\log\ n_{\rm H}\ [{\rm cm}^{-3}]$')
pl.savefig('nH_vs_NH.png',bbox_inches='tight',dpi=300)