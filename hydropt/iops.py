import numpy as np
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt

DATA_PATH = pkg_resources.resource_filename('hydropt', 'data/')
PHYTO_SIOP = pkg_resources.resource_filename('hydropt', 'data/phyto_siop.csv')

OLCI_WBANDS = np.array([400,412.5,442.5,490,510,560,620,665,673.75,681.25,708.75])
HSI_WBANDS = np.arange(400, 711, 5)
WBANDS = HSI_WBANDS

def interpolate_to_wavebands(data, wavelength, index='wavelength'):
    '''
    data - pandas df where index is wavelength and columns are SIOPs
    wavelength - wavelengths used to interpolate SIOPs
    '''
    data.reset_index(inplace=True)
    # add OLCI/MERIS wavebands to phytop SIOP wavelengths
    wband_merge = np.unique(sorted(np.append(data[index], wavelength)))
    
    data.set_index(index, inplace=True)
    data = data.reindex(wband_merge)\
        .astype(float)\
        .interpolate(method='slinear', fill_value=0, limit_direction='both')\
        .reindex(wavelength)
    
    return data

def waveband_wrapper(f, wb):
    def inner(*args, **kwargs):
        kwargs['wb'] = wb
        return f(*args, **kwargs)
    return inner

# load phytoplankton basis vector
a_phyto_base_full = pd.read_csv(PHYTO_SIOP, sep=';', index_col=0)
# interpolate to hyperspectral wavebands
a_phyto_base_HSI = interpolate_to_wavebands(data=a_phyto_base_full, wavelength=HSI_WBANDS)

def nap(*args, wb):
    '''
    IOP model for NAP
    '''
    # vectorized
    def iop(spm=args[0]):
        return spm*np.array([(.041*.75*np.exp(-.0123*(wb-443))), .014*0.57*(550/wb)])
    
    def gradient():
        d_a = .03075*np.exp(-.0123*(wb-443))
        d_b = .014*0.57*(550/wb)
        
        return np.array([d_a, d_b])
    
    return iop, gradient

def cdom(*args, wb):
    '''
    IOP model for CDOM
    '''
    def iop(a_440=args[0]):
        return np.array([a_440*np.exp(-0.017*(wb-440)), np.zeros(len(wb))])
    
    def gradient():
        '''
        Gradient of CDOM IOP model
        ''' 
        d_a = np.exp(-.017*(wb-440))
        d_b = np.zeros(len(d_a))
        
        return np.array([d_a, d_b])
    
    return iop, gradient

def phyto(*args):
    '''
    IOP model for phytoplankton w. 
    packaging effect - according to Prieur&Sathyenadrath (1981)
    basis vector - according to Ciotti&Cullen 2002
    '''   
    def iop(chl=args[0]):
        # calculate absorption
        a = 0.06*np.power(chl, .65)*a_phyto_base_HSI.absorption.values
        # calculate backscatter according to 0.1-tadzio-IOP_backscatter
        # notebook in hydropt-4-sent3
        b = np.repeat(.014*0.18*np.power(chl, .471), len(a))
        
        return np.array([a, b])
    
    def gradient():
        '''dummy gradient function'''
        return np.zeros([2,63])
    
    return iop, gradient

class IOP_model:
    def __init__(self):
        self._wavebands = None
        self.iop_model = None
    
    def set_iop(self, wavebands, **kwargs):
        if self.check_wavelen(wavebands, **kwargs):
            self._wavebands = wavebands
            self.iop_model = {k: v for (k,v) in kwargs.items()}
    
    def get_iop(self, **kwargs):
        iops = []
        for k, value in kwargs.items():
            iops.append([self.iop_model.get(k)(value)[0]()])
        
        iops = np.vstack(iops)
        
        return iops
   
    def get_gradient(self, **kwargs):
        grads = []
        for k, value in kwargs.items():
            grads.append([self.iop_model.get(k)(value)[1]()])
        
        grads = np.vstack(grads)
        
        return grads
    
    def sum_iop(self, **kwargs):
        iops = self.get_iop(**kwargs).sum(axis=0)
        
        return iops
    
    def plot(self, **kwargs):
        n = len(kwargs)
        fig, axs = plt.subplots(1,n, figsize=(14,4))
        # clean-up loop code
        for (k,v), ax in zip(kwargs.items(), axs):
            ax2 = ax.twinx()
            ax.plot(self._wavebands, self.get_iop(**{k:v})[0][0])
            ax.set_xlabel('wavelength')
            ax.set_ylabel('absorption')
            ax.set_title(k)
            ax2.plot(self._wavebands, self.get_iop(**{k:v})[0][1], color='blue')
            ax2.set_ylabel('backscatter')
        plt.tight_layout()
        #return fig
        
    @staticmethod
    def check_wavelen(wavebands, **kwargs):
        models = [i for i in kwargs.values()]
        # check dimensions of models; skip gradients for now
        dims = [i(1)[0]().shape[1] for i in models]
        # check if all dimension match
        if len(set(dims)) != 1:
            raise ValueError('length of IOP vectors do not match')
        elif dims[0] != len(wavebands):
            raise ValueError('number of wavebands do not match with length of IOP vectors')
        
        return True
    
class ThreeCompModel(IOP_model):
    def __init__(self):
        self.set_iop(HSI_WBANDS,
                     nap=waveband_wrapper(nap, wb=HSI_WBANDS),
                     cdom=waveband_wrapper(cdom, wb=HSI_WBANDS),
                     chl=phyto)