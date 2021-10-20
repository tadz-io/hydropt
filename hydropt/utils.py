import warnings
import itertools
import numpy as np
from xarray import Dataset

def interpolate_to_wavebands(data, wavelength, index='wavelength'):
    '''
    data - pandas df where index is wavelength and columns are SIOPs
    wavelength - wavelengths used to interpolate SIOPs
    '''
    data.reset_index(inplace=True)
    # add OLCI/MERIS wavebands to phytop SIOP wavelengths
    wband_merge = np.unique(sorted(np.append(data[index], wavelength)))

    data.set_index(index, inplace=True)
    #.interpolate(method='slinear', fill_value=0, limit_direction='both')\
    data = data.reindex(wband_merge)\
        .astype(float)\
        .interpolate()\
        .reindex(wavelength)

    warnings.warn('changed interpolation method')

    return data

def waveband_wrapper(f, wb):
    def inner(*args, **kwargs):
        kwargs['wb'] = wb
        return f(*args, **kwargs)
    return inner

def der_2d_polynomial(x, c, p):
    '''
    derivative of 2 variable polynomial

    x: points at wich to evaluate the derivative; a nx2 array were n is number of wavebands
    c: coefficients of the polynomial terms; a nxm matrix where n is the number wavebands and m the number of polynomial terms
    p: exponents of the n polynomial terms; a mx2 matrix
    '''
    if c.shape[0] is not x.shape[0]:
        warnings.warn('matrix dimensions of x and c do not match!')
    # derivative of powers and truncate to positive floats
    p_trunc = lambda p: float(max(0, p-1))
    # get derivative terms of polynomial features
    d_x1 = lambda x, p: p[0]*x[0]**(p_trunc(p[0]))*x[1]**p[1]
    d_x2 = lambda x, p: p[1]*x[1]**(p_trunc(p[1]))*x[0]**p[0]
    # evaluate terms at [x]
    ft = np.array([[d_x1(x,p), d_x2(x,p)] for (x,p) in zip(itertools.cycle([x.T]), p)])
    # dot derivative matrix with polynomial coefficients and get diagonal
    dx = np.array([np.dot(c,ft[:,0,:]).diagonal(), np.dot(c,ft[:,1,:]).diagonal()])
    
    return dx

def recurse(x, f_args=np.nan):
    
    def inner(x):
        n = []
        if isinstance(x, np.ndarray):
            return x

        for v in x:
            if callable(v):
                out = inner(v(f_args))
                n.append(out)
            else:
                raise ValueError('models should return np.array')
        
        return n
    
    out = inner(x)

    return np.array(out)

def to_xarray_dataset(func):
    '''
    wrapper for output Inversionmodel.invert()
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

def update_lmfit_parameters(x):
    x_in = list(x.params.values())
    x_out = []
    for i in x_in:
        i.init_value = i.value
        x_out.append(i)    
    x.params.add_many(*x_out)
    
    return x.params

def lmfit_results_to_array(x, parameters=[]):
    p_list = ['params'].extend(parameters)
    x_array = np.hstack(np.array([np.array(getattr(x, i)) for i in p_list]))
    
    return x_array
