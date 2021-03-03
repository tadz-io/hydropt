from ..bio_opts import TwoCompModel
from ..hydropt import PolynomialForward
from jax import jacfwd

'''
Testing jacfwd from jax lib

- import jax.numpy as np in hydropt module
- change return type from ForwardModel().forward() - should return numpy.array, not pd.Series
'''
#
bom = TwoCompModel()
fwd_model = PolynomialForward(bom)
fwd_l = lambda w: fwd_model.forward(**w)

p = jacfwd(fwd_l)({
    'nap': 22.0,
    'cdom': 22.0})  

q = fwd_model.jacobian(nap=22.0, cdom=22.0)
pass
