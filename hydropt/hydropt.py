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

def diff(x, y, f):
    return y - f(x)

def l1_loss(x, y, f):
    return np.abs(y - f(x))

def l2_loss(x, y, f):
    return (y - f(**x))**2

def chi_squared(x, y, f, sigma):
    return ((y - f(x))/sigma)**2
 
class InversionModel:
    def __init__(self, fwd_model, minimizer, loss_function=l2_loss):
        self._fwd_model = fwd_model
        self._minimizer = minimizer
        self._loss = loss_function

    def invert(self, x, y):
        ''' 
        invert() should take *args (sigma etc..) and pass to _minimizer(args=...)
        https://stackoverflow.com/questions/51883058/l1-norm-instead-of-l2-norm-for-cost-function-in-regression-model
        '''
        key, x0 = zip(*[(k,v) for (k, v) in x.items()])
        loss_func = lambda x, y, f: self._loss(dict(zip(key, x)), y, f)
        # to do: implement jac (scipy.optimize)/Dfun (lmfit)
        xhat = self._minimizer(loss_func, x0, args=(y, self._fwd_model.forward), method='lm')

        return xhat

    @property
    def iop_model(self):
        return self._fwd_model.iop_model.iop_model
