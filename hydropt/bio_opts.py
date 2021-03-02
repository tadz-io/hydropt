#import numpy as np
import jax.numpy as np
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt

OLCI_WBANDS = np.array([400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25, 708.75])
HSI_WBANDS = np.arange(400, 711, 5)
WBANDS = OLCI_WBANDS

def waveband_wrapper(f, wb):
    def inner(*args, **kwargs):
        kwargs['wb'] = wb
        return f(*args, **kwargs)
    return inner

def nap(*args, wb):
    '''
    IOP model for NAP
    '''
    # vectorized
    def iop(spm=args[0]):
        return spm*np.array([(.041*.75*np.exp(-.0123*(wb-443))), .014*0.57*(550/wb)])
    
    def gradient(*args):
        d_a = .03075*np.exp(-.0123*(wb-443))
        d_bb = .014*0.57*(550/wb)
        
        return np.array([d_a, d_bb])
    
    return iop, gradient

def cdom(*args, wb):
    '''
    IOP model for CDOM
    '''
    def iop(a_440=args[0]):
        return np.array([a_440*np.exp(-0.017*(wb-440)), np.zeros(len(wb))])
    
    def gradient(*args):
        '''
        Gradient of CDOM IOP model
        ''' 
        d_a = np.exp(-.017*(wb-440))
        d_bb = np.zeros(len(d_a))
        
        return np.array([d_a, d_bb])
    
    return iop, gradient

class BioOpticalModel:
    def __init__(self):
        self._wavebands = None
        self.iop_model = {}
        self.gradient = {}
    
    @property
    def wavebands(self):
        return self._wavebands

    def set_iop(self, wavebands, **kwargs):
        if self.check_wavelen(wavebands, **kwargs):
            self._wavebands = wavebands
            try:
                self.iop_model.update({k: v(None)[0] for (k, v) in kwargs.items()})
                self.gradient.update({k: v(None)[1] for (k, v) in kwargs.items()})
            except:
                self.iop_model.update({k: v for (k, v) in kwargs.items()})
    
    def get_iop(self, **kwargs):
        iops = []
        for k, value in kwargs.items():
            iops.append([self.iop_model.get(k)(value)])
        
        iops = np.vstack(iops)
        
        return iops

    def get_gradient(self, **kwargs):
        grads = []
        for k, value in kwargs.items():
            grads.append([self.gradient.get(k)(value)])
        
        grads = np.vstack(grads)
        
        return grads
    
    def sum_iop(self, **kwargs):
        iops = self.get_iop(**kwargs).sum(axis=0)
        
        return iops
    
    def plot(self, **kwargs):
        n = len(kwargs)
        fig, axs = plt.subplots(1,n, figsize=(14, 4))
        # to do: clean-up loop code
        # pass kwargs to plt.plot
        for (k,v), ax in zip(kwargs.items(), axs):
            ax.plot(self._wavebands, self.get_iop(**{k:v})[0][0], label='absorption')
            ax.plot(self._wavebands, self.get_iop(**{k:v})[0][1], label='backscatter')
            ax.set_xlabel('wavelength (nm)')
            ax.set_ylabel('IOP ($m^{-1}$)')
            ax.set_title(k)     
            ax.legend()

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

class TwoCompModel(BioOpticalModel):
    def __init__(self):
        super().__init__()
        self.set_iop(HSI_WBANDS,
                     nap=waveband_wrapper(nap, wb=HSI_WBANDS),
                     cdom=waveband_wrapper(cdom, wb=HSI_WBANDS))