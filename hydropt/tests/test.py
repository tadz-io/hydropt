from ..iops import ThreeCompModel
from ..hydropt import PolynomialForward, InversionModel, forward_to_dataset
from scipy import optimize
import lmfit

iop_model = ThreeCompModel()
fwd_model = PolynomialForward(iop_model)
inv_model = InversionModel(fwd_model, lmfit.minimize)
rrs_0 = forward_to_dataset(inv_model._fwd_model.forward)(nap=1, chl=1, cdom=1)

x0 = lmfit.Parameters()
x0.add('cdom', value=.01, min=1E-9, max=10)
x0.add('nap', value=.1, min=1E-9, max=20)
x0.add('chl', value=.01, min=1E-9, max=10)
#x0 = {'nap':1, 'chl': 1, 'cdom': 1}

inv_model.invert(x0, rrs_0)
print(fwd_model.forward(nap=1, cdom=1, chl=1))
print(fwd_model.forward(nap=1, cdom=1, chl=1))
pass
