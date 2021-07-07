import warnings
import itertools
import numpy as np

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
    # get derivative terms of polynomial features
    d_x1 = lambda x, p: p[0]*x[0]**(float(p[0]-1))*x[1]**p[1]
    d_x2 = lambda x, p: p[1]*x[1]**(float(p[1]-1))*x[0]**p[0]
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