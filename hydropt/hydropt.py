import numpy as np
import warnings
import pandas as pd
from xarray import DataArray
import types
#import pkg_resources
#from .iops import IOP_model
from abc import ABC, abstractmethod
from sklearn.preprocessing import PolynomialFeatures

''' TO DO:
# - move model coefficients to XML/JSON file and include bounds                                     
# - to use jax/autograd replace sklearn PolynomialFeatures with:
    
    def _polynomial_features(self, x):
        x = x.T
        f = np.array([(x[0]**i)*(x[1]**j) for (i,j) in self.model_powers]).T

        return f
'''


# PACE_POLYNOM_05 = pkg_resources.resource_filename('hydropt', 'data/PACE_polynom_05.csv')
# OLCI_POLYNOM_04 = pkg_resources.resource_filename('hydropt', 'data/OLCI_polynom_04.csv')

PACE_POLYNOM_05 = pd.read_csv('/Users/tadzio/Documents/code_repo/hydropt_4_sent3/data/processed/model_coefficients/PACE_polynom_05.csv', index_col=0)

class Interpolator:
    '''
    see descriptors: 
    https://stackoverflow.com/questions/55511445/how-to-pass-self-to-a-method-like-object-in-python
    '''
    def __init__(self, dims, method='linear'):
        self.dims = dims
        self.method = method

    def interpolate(self, da, xnew):
        return da.interp(**{self.dims: xnew}, method=self.method)

    def __call__(self, _instance, xnew):
        instance = _instance.__class__()
        # copy attributes
        instance.__dict__.update(_instance.__dict__)
        # replace _parameters w. interpolated values
        setattr(instance, '_parameters', self.interpolate(instance._parameters, xnew))

        return instance

    def __get__(self, instance, owner):
        return types.MethodType(self, instance) if instance else self
        

class WavebandError(ValueError):
    pass

''' Abstract base classes

Definition of all abstract base classes

ForwardModel: model to calculate Rrs from IOPs
'''

class ReflectanceModel(ABC):
    ''' 
    rename ForwardModel -> ReflectanceModel
    '''
    @property
    @abstractmethod
    def _domain(self):
        pass
    
    @abstractmethod
    def interpolate(self, xnew):
        '''return: instance of class'''
            
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def gradient(self):
        pass
    
    def plot(self):
        pass

''' Context classes

ForwardModel: ...
'''

class ForwardModel:
    '''
    BioOpticalModel -> ForwardModel
    '''
    def __init__(self, iop_model, refl_model):
        self.iop_model = iop_model
        self.refl_model = refl_model
        self.__cache = True
        # method does nothing yet -> pass to interpolate()
        self._method = 'linear'

    @property
    def iop_model(self):
        return self.__iop_model

    @iop_model.setter
    def iop_model(self, m):
        self.__iop_model = m
        self.__cache = True

    @property
    def refl_model(self):      
        return self.__refl_model

    @refl_model.setter
    def refl_model(self, m):
        self.__refl_model = m
        self.__cache = True
    
    def forward(self, **x):
        if self.__cache:
            self.refl_model = self.refl_model.interpolate(self.iop_model.wavebands)
            self.__cache = False
            
        iops = self.iop_model.sum_iop(**x)
        self._validate_bounds(x)
        
        return self.refl_model.forward(iops)
    
    def _validate_bounds(self, x):
        pass


class PolynomialReflectance(ReflectanceModel):

    _parameters = DataArray(PACE_POLYNOM_05)
    interpolate = Interpolator(dims='wavelength')
    _domain = None
    gradient = None

    def forward(self, x):
        c = self._parameters
        x = np.log(x)
        # get polynomial features
        ft = PolynomialFeatures(degree=5).fit_transform(x.T)
        # calculate log(Rrs)
        log_rrs = np.dot(c, ft.T).diagonal()

        return np.exp(log_rrs)

class PolynomialForward(ForwardModel):
    def __init__(self, iop_model):
        super().__init__(iop_model, PolynomialReflectance())
 
class InversionModel:
    def __init__(self, fwd_model, minimizer):
        pass
    
# def forward(self, comp_conc):
#     # calculate (and validate) IOPs
#     iops = self.iop_model(comp_conc)
#     self.validate_iops(iops)
#     # calculate rrs and interpolate
#     self._current_rrs = self.forward_model(iops)
#     self.interpolate(rrs)(self.iop_model._wavebands)
#     self._current_iops = iops

#     return self._current_rrs


# class Hydropt:
#     def __init__(self, iop_model):
#         self.iop_model = iop_model
#         self.model_coef = None
#         self.model_powers = None
#         self.set_model_coef()
        
#     def set_model_coef(self):
#         model_coef = pd.read_csv(OLCI_POLYNOM_04, index_col = 0)
#         # check if wavebands match
#         if np.array_equal(model_coef.index, self.iop_model._wavebands):
#             self.model_coef = model_coef
#             self.model_powers = PolynomialFeatures(degree=4).fit([[1,1]]).powers_
#         else:
#             raise ValueError('wavebands do not match number of model coefficients')
            
#     def hydrolight_polynom(self, x, degree=4, ignore_warnings=False):
#         '''
#         Forward model using polynomial fit to Hydrolight simulations

#         x[0]: total absorption at wavelength i
#         x[1]: total backscatter at wavelength i

#         returns Rrs
#         '''
#         # check if IOPs are outside of model bounds
#         if not ignore_warnings:
#             self._validate_iop_bounds(x)
#         # log absorption, backscatter
#         x_log = np.log(x)
#         # get polynomial features
#         ft = PolynomialFeatures(degree=degree).fit_transform(x_log.T)
#         # get polynomial coefficients
#         c = self.model_coef
#         # calculate log(Rrs)
#         log_rrs = np.dot(c, ft.T).diagonal()
#         # calculate Rrs
#         rrs = np.exp(log_rrs)

#         return rrs
    
#     def _validate_iop_bounds(self, x):
#         bounds_exceded = np.any([
#             np.any(x<OLCI_IOP_LOWER_BOUNDS*.2),
#             np.any(x>OLCI_IOP_UPPER_BOUNDS*1.1)])
        
#         if bounds_exceded:
#             warnings.warn('''IOP(s) exceed bounds of polynomial model.
#             Caution must be taken when extrapolating outside of bounds''')
    
#     def forward(self, ignore_warnings=False, **kwargs):
#         # calculate total absorption/backscatter
#         iop = self.iop_model.sum_iop(**kwargs)
        
#         return self.hydrolight_polynom(iop, ignore_warnings=ignore_warnings)