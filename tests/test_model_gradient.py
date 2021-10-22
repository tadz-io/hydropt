import unittest
import numpy as np
from hydropt.hydropt import PolynomialForward, PolynomialReflectance, BioOpticalModel
#from jax import jacfwd
from hydropt.bio_optics import clear_nat_water, phyto, HSI_WBANDS

def jacfwd(x):
    '''dummy function'''
    pass

bio_opt_model = BioOpticalModel()
bio_opt_model.set_iop(
    wavebands=HSI_WBANDS,
    water=clear_nat_water,
    chl = phyto)
@unittest.skip('under development')
class TestPolynomialReflectance(unittest.TestCase):
    def setUp(self, model=PolynomialReflectance(), n=5):
        self.fwd_model = model.forward
        self.gradient = model.gradient
        self.n = n
        self.dims = (2, len(model._parameters))

    def test_jacobian(self):
        for i in range(self.n):
            # set seed
            np.random.seed(i)
            iops = np.random.rand(*self.dims)
            # analytical computation of jacobian
            jac_ana = self.gradient(iops)
            # numerical approximation of jacobian
            jac_num = jacfwd(self.fwd_model)(iops)
            # diagonalize and stack
            jac_num = np.vstack([
                np.diag(jac_num[:,0,:]),
                np.diag(jac_num[:,1,:])])
            np.testing.assert_allclose(jac_num, jac_ana, rtol=.01)

@unittest.skip('under development')
class TestForwardModel(unittest.TestCase):
    def setUp(self, model=PolynomialForward(bio_opt_model)):
        # to do: set up some example BioOpticalModels to init PolynomialForward()
        self.model = model
        self.fwd_model = lambda chl: model.forward(chl=chl)

    def _test_jacobian(self):
        jac_ana = self.model.jacobian(chl=.1)
        # replace None's with parameter values
        jac_num = jacfwd(self.fwd_model, (0))(chl=.1)


if __name__ == '__main__':
    unittest.main()