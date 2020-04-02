import numpy as np
import pandas as pd
import pkg_resources
from sklearn.preprocessing import PolynomialFeatures

PACE_POLYNOM_05 = pkg_resources.resource_filename('hydropt', 'data/PACE_polynom_05.csv')

class Hydropt:
    def __init__(self, iop_model):
        self.iop_model = iop_model
        self.model_coef = None
        self.model_powers = None
        self.set_model_coef()
        
    def set_model_coef(self):
        model_coef = pd.read_csv(PACE_POLYNOM_05, index_col = 0)
        # check if wavebands match
        if np.array_equal(model_coef.index, self.iop_model._wavebands):
            self.model_coef = model_coef
            self.model_powers = PolynomialFeatures(degree=5).fit([[1,1]]).powers_
        else:
            raise ValueError('wavebands do not match number of model coefficients')
            
    def hydrolight_polynom(self, x, degree=5):
        '''
        Forward model using polynomial fit to Hydrolight simulations

        x[0]: total absorption at wavelength i
        x[1]: total backscatter at wavelength i

        returns Rrs
        '''
        # log absorption, backscatter
        x_log = np.log(x)
        # get polynomial features
        ft = PolynomialFeatures(degree=degree).fit_transform(x_log.T)
        # get polynomial coefficients
        c = self.model_coef
        # calculate log(Rrs)
        log_rrs = np.dot(c, ft.T).diagonal()
        # calculate Rrs
        rrs = np.exp(log_rrs)

        return rrs
    
    def forward(self, **kwargs):
        # calculate total absorption/backscatter
        iop = self.iop_model.sum_iop(**kwargs)
        
        return self.hydrolight_polynom(iop)