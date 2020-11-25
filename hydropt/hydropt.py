import numpy as np
import warnings
import pandas as pd
from xarray import DataArray, Dataset
from .band_models import BandModel
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
        
        return pd.Series(self.refl_model.forward(iops), index=self.iop_model.wavebands)
    
    def _validate_bounds(self, x):
        pass


class PolynomialReflectance(ReflectanceModel):

    _parameters = DataArray(PACE_POLYNOM_05)
    _domain = None
    interpolate = Interpolator(dims='wavelength')
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


def _residual(x, y, f, w):
    '''weighted residuals'''
    
    return (y - f(**x))/(1/np.sqrt(w))


class ValidationDataset:

    def __init__(self, fwd_model):
        self.fwd_model = fwd_model

    def create(self, wt):
        ''' create validation set

        wt - str indicating watertype: c1, c2 etc..
        '''

        pass

def _to_dataset(func):
    '''
    make this a validation set decorator

    to do: take extra argument of observed concentration and calculate
    rrs -> add noise -> invert using .invert() -> create validation set

    let this function create the synthetic dataset and invert it -> return the validation set:
    observed vs. predicted
    '''
    def wrapper(*args, **kwargs):
        '''
        m - InversionModel instance
        x, y, w - see InversionModel
        '''
        stat_vars = ['chisqr', 'redchi', 'aic', 'bic']
        fwd_model = args[0]._fwd_model.forward 
        iop_model = args[0]._fwd_model.iop_model
        # do inversion
        out = func(*args, **kwargs)
        # get estimates
        x_hat = {i: float(j) for i,j in out.params.items()}
        rrs_hat = fwd_model(**x_hat)
        iop_hat = iop_model.get_iop(**x_hat)
        # organize data in dict
        data = {
            'rrs': (['wavelength'], rrs_hat),
            'iops': (['comp', 'iop', 'wavelength'], iop_hat),
            'conc': (['comp'], [i for i in x_hat.values()]),
            'weights': (['wavelength'], kwargs.get('w', np.repeat(1, len(rrs_hat))))}
        # add stats
        data.update({i: getattr(out, i) for i in stat_vars})   
        # iop_hat = {k: v for k,v in zip(x_hat.keys(), iop_model.get_iop(**x_hat))}
        # calculate standard-error
        try:
            data.update({'std_error': (['comp'], np.sqrt(getattr(out, 'covar').diagonal()))})
        except AttributeError:
            data.update({'std_error': (['comp'], np.repeat(np.nan, len(x_hat)))})
        # set coordinates
        coords = {
            'wavelength': iop_model.wavebands,
            'comp': [i for i in x_hat.keys()],
            'iop': ['absorption', 'backscatter']}
        
        return Dataset(data, coords=coords)
    
    return wrapper

 
class InversionModel:
    def __init__(self, fwd_model, minimizer, loss=_residual, band_model='rrs'):
        self._fwd_model = fwd_model
        self._minimizer = minimizer
        self._loss = loss
        self._band_model = BandModel(band_model)

    @_to_dataset
    def invert(self, y, x, w=1):
        ''' 
        x - initial guess
        y - rrs to invert
        w - weights to wavebands

        decorate invert() to parse output to dict/DataFrame
        specify decorater class during init or as property
        https://stackoverflow.com/questions/51883058/l1-norm-instead-of-l2-norm-for-cost-function-in-regression-model
        '''
        key, x0 = zip(*[(k, float(v)) for (k, v) in x.items()])
        #loss_func = lambda x, y, f: self._loss(dict(zip(key, x)), y, f, w)
        loss_func = lambda x, y, f: self._loss(dict(x.valuesdict()), y, f, w)
        # apply band-transformation on y and model
        args = self._band_model((y, self._fwd_model.forward))
        # to do: implement jac (scipy.optimize)/Dfun (lmfit)
        xhat = self._minimizer(loss_func, x, args=args)

        return xhat

    @property
    def iop_model(self):
        return self._fwd_model.iop_model.iop_model
