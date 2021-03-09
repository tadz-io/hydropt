from ..iops import FiveCompModel, ThreeCompModel
from ..hydropt import PolynomialForward, InversionModel
from jax import jacfwd
from scipy import optimize
import lmfit


iop_model = FiveCompModel()
fwd_model = PolynomialForward(iop_model)
inv_model = InversionModel(fwd_model, lmfit.minimize)
rrs_0 = fwd_model.forward(
    nap=.3,
    cdom=2E-2, pico=.5, nano=.3, micro=1)

x0 = lmfit.Parameters()
x0.add('cdom', value=.13, min=1E-9, max=10)
x0.add('nap', value=.01, min=1E-9, max=20)
x0.add('pico', value=.1, min=1E-9, max=10)
x0.add('nano', value=.1, min=1E-9, max=10)
x0.add('micro', value=.1, min=1E-9, max=10)
# #x0 = {'nap':1, 'chl': 1, 'cdom': 1}



q = inv_model.invert(x0, rrs_0)
pass

# '''
# Testing jacfwd from jax lib

# - import jax.numpy as np in hydropt module
# - change return type from ForwardModel().forward() - should return numpy.array, not pd.Series
# - change assignment method in ForwardModel().jacobian() - use jax.ops.index_update
# '''
# #
# bom = TwoCompModel()
# fwd_model = PolynomialForward(bom)
# inv_model = InversionModel(fwd_model, lmfit.minimize)
# fwd_l = lambda w: inv_model._fwd_model.forward(**w)

# p = jacfwd(fwd_l)({
#     'nap': 0.01,
#     'cdom': 0.13})  

# q = inv_model._fwd_model.jacobian(nap=0.01, cdom=0.13)
# pass
