from ..hydropt import PolynomialForward, PACE_POLYNOM_05, der_2d_polynomial
from ..bio_opts import TwoCompModel
from sklearn.preprocessing import PolynomialFeatures
import lmfit
import numpy as np
#import jax.numpy as np
from xarray import DataArray
from jax import jacfwd
import jax

HSI_WBANDS = np.arange(400, 711, 5)

powers = PolynomialFeatures(degree=5).fit([[1, 1]]).powers_
cfs = DataArray(PACE_POLYNOM_05).values

iop_model = TwoCompModel()
fwd_model = PolynomialForward(iop_model)
fwd_model_l = lambda x: fwd_model.forward(**x)
rrs_0 = fwd_model.forward(nap=.3, cdom=.02)

nap_grad = np.array([
    .03075*np.exp(-.0123*(HSI_WBANDS-443)),
    .014*0.57*(550/HSI_WBANDS)])

cdom_grad = np.array([
    np.exp(-.017*(HSI_WBANDS-440)),
    np.zeros(len(HSI_WBANDS))])

def refl_grad(x):
    d_p = der_2d_polynomial(np.log(x.T), cfs, powers)
    d_a = 1/x[0]
    d_bb = 1/x[1]
    rrs = fwd_model.refl_model.forward(x)
    dx = rrs*d_p*np.array([d_a, d_bb])

    return dx


def residuals(x):
    residual = rrs_0-fwd_model.forward(**dict(x.valuesdict()))

    return residual

def jacobian_num(x):
    x = {
        'nap': x['nap'].value,
        'cdom': x['cdom'].value}
    jac = jacfwd(fwd_model_l)(x)
    jac_array = np.array([jac.get('nap'), jac.get('cdom')])

    return jac_array.T

def jacobian(x):
    '''
    figure out order of components [nap, cdom]*grad or [cdom, nap]*grad
    '''
    jac = np.empty([63, 2])
    xp = {
        'nap': x['nap'].value,
        'cdom': x['cdom'].value}
    #grad_iops = fwd_model.iop_model.get_gradient(**xp)
    grad_iops = np.array([nap_grad, cdom_grad])
    iops = fwd_model.iop_model.sum_iop(**xp)
    #grad_refl = fwd_model.refl_model.gradient(iops)
    grad_refl = refl_grad(iops)
    
    for c, (i,j) in enumerate(zip(grad_refl.T, grad_iops.T)):
        jac[c] = np.dot(i,j)
        #jac = jax.ops.index_update(jac, jax.ops.index[c], np.dot(i, j))
        
    return jac

def jacobian_v2(x):

    if isinstance(x, lmfit.Parameters):
        x = [x['nap'].value, x['cdom'].value]
    
    # gradients of IOP models at every waveband
    grad_iops = np.array([nap_grad, cdom_grad])
    iops = fwd_model.iop_model.sum_iop(nap=x[0], cdom=x[1])
    # calculate rrs
    rrs = fwd_model.refl_model.forward(iops)
    # (dln(a)/da) * (dRrs/dln(Rrs)) = Rrs/a; (dln(b)/db) * (dRrs/dln(Rrs)) = Rrs/b; do element-wise multiplication
    grad_log_iops = np.array([rrs/iops[0],rrs/iops[1]]) * grad_iops
    # evaluate jacobian of hydrolight polynomial at x
    grad_polynom = der_2d_polynomial(np.log(iops.T), cfs, powers)
    #initialize empty jacobian matrix
    jac = np.empty([grad_polynom.shape[1], grad_log_iops.shape[0]])  
    for c, (i,j) in enumerate(zip(grad_polynom.T, grad_log_iops.T)):
        jac[c] = np.dot(i,j)
        #jac = jax.ops.index_update(jac, jax.ops.index[c], np.dot(i, j))
    
    return jac

x0 = lmfit.Parameters()
x0.add('nap', value=.2, min=1E-9, max=20)
x0.add('cdom', value=.01, min=1E-9, max=10)

# p = jacfwd(fwd_model_l)({
#     'nap': .2,
#     'cdom': .01})

q = jacobian(x0)
q2 = jacobian_v2(x0)

xhat = lmfit.minimize(residuals, x0, Dfun=jacobian_v2)
print(residuals(x0))