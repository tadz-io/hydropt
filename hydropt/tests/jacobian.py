from ..hydropt import PolynomialForward, PACE_POLYNOM_05
from ..bio_opts import TwoCompModel
from sklearn.preprocessing import PolynomialFeatures
import lmfit
import numpy as np
#import jax.numpy as np
from xarray import DataArray
from jax import jacfwd
import jax
import itertools
import warnings

HSI_WBANDS = np.arange(400, 711, 5)
wbands = HSI_WBANDS

iop_model = TwoCompModel()
fwd_model = PolynomialForward(iop_model)

powers = PolynomialFeatures(degree=5).fit([[1, 1]]).powers_
cfs = DataArray(PACE_POLYNOM_05).values

def cdom_iop(a_440):
    '''
    IOP model for CDOM
    '''
    return np.array([a_440*np.exp(-0.017*(wbands-440)), np.zeros(len(wbands))])

def nap_iop(spm):
    '''
    IOP model for NAP
    '''
    # vectorized
    return spm*np.array([(.041*.75*np.exp(-.0123*(wbands-443))), 0.014*0.57*(550/wbands)])

def grad_cdom_iop():
    '''
    Gradient of CDOM IOP model
    ''' 
    d_a = np.exp(-.017*(wbands-440))
    d_b = np.zeros(len(d_a))
    return np.array([d_a, d_b])

def grad_nap_iop():
    '''
    Gradient of NAP IOP model
    '''
    d_a = .03075*np.exp(-.0123*(wbands-443))
    d_b = .014*0.57*(550/wbands)
    
    return np.array([d_a, d_b])

def total_iop(ag, spm):
    '''
    total_iop calculates total absorption and scatter coefficients (excluding water)
    
    ag - CDOM absorption at 440nm
    spm - SPM concentration in g/m3
    pico - chl concentration in mg/m3 for picos
    nano - chl concentration in mg/m3 for nanos
    micro - chl concentration in mg/m3 for micros
    '''
    
    return cdom_iop(ag), nap_iop(spm)

def der_2d_polynomial(x, c, p):
    '''
    derivative of 2 variable polynomial
    
    x: points at wich to evaluate the derivative; a nx2 array were n is number of wavebands
    c: coefficients of the polynomial terms; a nxm matrix where n is the number wavebands and m the number of polynomial terms
    p: exponents of the n polynomial terms; a mx2 matrix
    '''
    if c.shape[0] is not x.shape[0]:
        warnings.warn('matrix dimensions of x and c do not match!')
    # get derivative terms of polynomial features
    d_x1 = lambda x, p: p[0]*x[0]**(float(p[0]-1))*x[1]**p[1]
    d_x2 = lambda x, p: p[1]*x[1]**(float(p[1]-1))*x[0]**p[0]
    # evaluate terms at [x]
    ft = np.array([[d_x1(x,p), d_x2(x,p)] for (x,p) in zip(itertools.cycle([x.T]), p)])
    # dot derivative matrix with polynomial coefficients and get diagonal
    dx = np.array([np.dot(c,ft[:,0,:]).diagonal(), np.dot(c,ft[:,1,:]).diagonal()])
    
    return dx

def hydrolight_polynom(x, degree=5):
    '''
    Forward model using polynomial fit to Hydrolight simulations
    
    x[0]: total log absorption at wavelength i
    x[1]: total log backscatter at wavelength i
    '''
    # get polynomial features
    ft = PolynomialFeatures(degree=degree).fit_transform(x.T)
    # get polynomial coefficients
    # calculate log(Rrs)
    log_Rrs = np.dot(cfs, ft.T).diagonal()
    
    #return np.exp(log_Rrs)
    return log_Rrs

def jacobian(x):
    '''
    Jacobian at [x] of the remote-sensing reflectance (Rrs),
    using the Hydrolight polynomial as forward model, w.r.t concentration of constituents
    
    x[0]: cdom absorption at 440 nm
    x[1]: spm concentration in g/m^3
    x[2]: pico concentration in mg/m^3
    x[3]: nano concentration in mg/m^3
    x[4]: micro concentration in mg/m^3
    '''
    if isinstance(x, lmfit.Parameters):
        x = [x['cdom'].value, x['nap'].value]
    
    # gradients of IOP models at every waveband
    grad_iops = np.array([grad_cdom_iop(), grad_nap_iop()])
    #iops = np.sum(total_iop(*x), axis=0)
    iops = fwd_model.iop_model.sum_iop(cdom=x[0], nap=x[1])
    # calculate rrs
    #rrs = np.exp(hydrolight_polynom(np.log(iops)))
    rrs = fwd_model.refl_model.forward(iops)
    # (dln(a)/da) * (dRrs/dln(Rrs)) = Rrs/a; (dln(b)/db) * (dRrs/dln(Rrs)) = Rrs/b; do element-wise multiplication
    grad_log_iops = np.array([rrs/iops[0], rrs/iops[1]]) * grad_iops
    # evaluate jacobian of hydrolight polynomial at x
    grad_polynom = der_2d_polynomial(np.log(iops.T), cfs, powers)
    #initialize empty jacobian matrix
    jac = np.empty([grad_polynom.shape[1], grad_log_iops.shape[0]])  
    for c, (i,j) in enumerate(zip(grad_polynom.T, grad_log_iops.T)):
        jac[c] = np.dot(i,j)
        
    return jac

def cost_function(y):
    '''
    
    create scalar cost function by summing residuals
    y - target
    '''

    def minimizer(x0):
        ag = x0['cdom'].value
        spm = x0['nap'].value
        # calculate log of total IOPs
        rrs = fwd_model.forward(cdom=ag, nap=spm)
        # calculate residuals
        residual = rrs-y
        
        return residual
    
    return minimizer

iops_t = np.sum(total_iop(ag=.02, spm=.3), axis=0)
rrs_0 = fwd_model.forward(cdom=.02, nap=.3)

x0 = lmfit.Parameters()
x0.add('cdom', value=.01, min=1E-9, max=10)
x0.add('nap', value=.1, min=1E-9, max=200)

xhat = lmfit.minimize(cost_function(rrs_0), x0, Dfun=jacobian)
pass