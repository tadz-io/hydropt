from ..iops import ThreeCompModel
from ..hydropt import PolynomialForward

iop_model = ThreeCompModel()
fwd_model = PolynomialForward(iop_model)
fwd_model.forward(nap=1, cdom=1, chl=1)
