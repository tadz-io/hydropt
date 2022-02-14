import warnings
import itertools
import numpy as np
from numpy.lib.index_tricks import ndindex

class LmfitResultsToArray:
    def __init__(self, results):
        for k,v in results.__dict__.items():
            setattr(self, k, v)

def apply_along_axis(func1d, axis, arr, progress_bar=None, *args, **kwargs):
    '''
    Modified from numpy's apply_along_axis function
    '''
    # add arr = np.asanyarray()
    # check if type(axis) == str
    nd = arr.ndim
    # arr, with the iteration axis at the end
    in_dims = list(range(nd))
    inarr_view = np.transpose(arr, in_dims[:axis] + in_dims[axis+1:] + [axis])
    # compute indices for the iteration axes
    inds = iter(ndindex(inarr_view.shape[:-1]))
    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration as e:
        raise ValueError(
            'Cannot apply_along_axis when any iteration dimensions are 0'
        ) from None
    res = func1d(inarr_view[ind0], *args, **kwargs)
    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    buff = np.zeros(inarr_view.shape[:-1] + res.shape, res.dtype)
    # permutation of axes such that out = buff.transpose(buff_permute)
    buff_dims = list(range(buff.ndim))
    buff_permute = (
        buff_dims[0 : axis] +
        buff_dims[buff.ndim-res.ndim : buff.ndim] +
        buff_dims[axis : buff.ndim-res.ndim]
    ) 
    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    if progress_bar:
        inds = progress_bar(list(inds))
    for ind in inds:
        buff[ind] = func1d(inarr_view[ind], *args, **kwargs)

    # finally, rotate the inserted axes back to where they belong
    return np.transpose(buff, buff_permute)

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
    dx = np.array([
        np.dot(c, ft[:,0,:]).diagonal(),
        np.dot(c, ft[:,1,:]).diagonal()])

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

def update_lmfit_parameters(x):
    x_in = list(x.params.values())
    x_out = []
    for i in x_in:
        i.init_value = i.value
        x_out.append(i)
    x.params.add_many(*x_out)
    
    return x.params

def lmfit_results_to_array(x, parameters=[]):
    p_list = ['params']
    if 'std' in parameters:
        try:
            std = standard_error(x.covar)
        except AttributeError:
            std = np.empty(len(x.params))
            std[:] = np.nan
        x.std = std
    p_list.extend(parameters)
    x_array = np.hstack([np.array(getattr(x, i)) for i in p_list])
    
    return x_array

def standard_error(cov):
    '''
    Calculate standard error from covariance matrix
    
    Args:
        cov: nxn covariance matrix
    Returns:
        standard error as a 1xn array
    '''
    return np.sqrt(cov.diagonal())