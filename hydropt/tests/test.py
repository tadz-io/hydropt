from ..bio_opts import TwoCompModel
from ..hydropt import PolynomialForward
from jax import jacfwd
#
bom = TwoCompModel()
fwd_model = PolynomialForward(bom)
fwd_l = lambda w: fwd_model.forward(**w)

import jax.numpy as np
p = jacfwd(fwd_l)({
    'nap': 22.0,
    'cdom': 22.0})  

q = fwd_model.jacobian(nap = 22.0, cdom=22.0)
pass
