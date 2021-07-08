import numpy as np
import pandas as pd
import pkg_resources
from hydropt.utils import interpolate_to_wavebands

PHYTO_SIOP = pkg_resources.resource_filename('hydropt', 'data/phyto_siop.csv')
PHYTO_SC_SIOP = pkg_resources.resource_filename('hydropt', 'data/psc_absorption_se_uitz_2008.csv')
H2O_IOP_DEFAULT_STREAM = pkg_resources.resource_filename('hydropt', '/data/water_mason016.csv')
H2O_IOP_DEFAULT = pd.read_csv(H2O_IOP_DEFAULT_STREAM, sep=',', index_col='wavelength')

OLCI_WBANDS = np.array([400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25, 708.75])
HSI_WBANDS = np.arange(400, 711, 5)
WBANDS = OLCI_WBANDS

# load phytoplankton basis vector
a_phyto_base_full = pd.read_csv(PHYTO_SIOP, sep=';', index_col=0)
# load phytoplankton size class basis vectors (pico, nano, micro)
a_psc_base_full = pd.read_csv(PHYTO_SC_SIOP)
# interpolate to hyperspectral wavebands
a_phyto_base_HSI = interpolate_to_wavebands(data=a_phyto_base_full, wavelength=HSI_WBANDS)
a_phyto_base_OLCI = interpolate_to_wavebands(data=a_phyto_base_full, wavelength=OLCI_WBANDS)

pico_siop = a_psc_base_full[['wavelength','pico', 'pico_se']]
pico_siop.set_index('wavelength', inplace=True)

nano_siop = a_psc_base_full[['wavelength','nano', 'nano_se']]
nano_siop.set_index('wavelength', inplace=True)

micro_siop = a_psc_base_full[['wavelength','micro', 'micro_se']]
micro_siop.set_index('wavelength', inplace=True)
# set absorption and std to 0 for 710 nm
pico_siop.loc[710,:] = [0,0]
nano_siop.loc[710,:] = [0,0]
micro_siop.loc[710,:] = [0,0]

a_pico_base = interpolate_to_wavebands(data=pico_siop, wavelength=WBANDS)
a_nano_base = interpolate_to_wavebands(data=nano_siop, wavelength=WBANDS)
a_micro_base = interpolate_to_wavebands(data=micro_siop, wavelength=WBANDS)

def clear_nat_water(*args):
    '''
    IOP model for clear natural water
    '''
    def iop(*args):
        return H2O_IOP_DEFAULT.T.values

    def gradient(*args):
        return np.full(H2O_IOP_DEFAULT.T.shape, np.nan)

    return iop, gradient

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

def phyto(*args):
    '''
    IOP model for phytoplankton w. 
    packaging effect - according to Prieur&Sathyenadrath (1981)
    basis vector - according to Ciotti&Cullen 2002
    '''   
    def iop(chl=args[0]):
        # calculate absorption
        #a = 0.06*np.power(chl, .65)*a_phyto_base_HSI.absorption.values
        a = 0.06*chl*a_phyto_base_HSI.absorption.values
        # calculate backscatter according to 0.1-tadzio-IOP_backscatter
        # notebook in hydropt-4-sent3
        #bb = np.repeat(.014*0.18*np.power(chl, .471), len(a))
        bb = np.repeat(.014*0.18*chl, len(a))

        return np.array([a, bb])
    
    def gradient(*args):
        '''dummy gradient function'''
        return np.zeros([2,63])
    
    return iop, gradient

def phyto_olci(*args):
    '''
    IOP model for phytoplankton w. 
    packaging effect - according to Prieur&Sathyenadrath (1981)
    basis vector - according to Ciotti&Cullen 2002
    '''   
    def iop(chl=args[0]):
        # calculate absorption
        a = 0.06*np.power(chl, .65)*a_phyto_base_OLCI.absorption.values
        # calculate backscatter according to 0.1-tadzio-IOP_backscatter
        # notebook in hydropt-4-sent3
        bb = np.repeat(.014*0.18*np.power(chl, .471), len(a))
        
        return np.array([a, bb])
    
    def gradient(*args):
        '''dummy gradient function'''
        return np.zeros([2,11])
    
    return iop, gradient

def pico(*args):
    '''
    pico IOP model
    
    chl - concentration in mg/m3
    '''
    def iop(chl=args[0]):
        # chl specific absorption 
        a_star = a_pico_base['pico'].values
        bb_star = .0038*(WBANDS/470)**-1.4

        return chl*np.array([a_star.reshape(-1), bb_star])
    
    def gradient(*args):
        d_a = a_pico_base['pico'].values
        d_bb = .0038*(WBANDS/470)**-1.4

        return np.array([d_a.reshape(-1), d_bb])
    
    return iop, gradient

def nano(*args):
    '''
    nano IOP model
    
    chl - concentration in mg/m3
    '''
    def iop(chl=args[0]):
        # chl specific absorption 
        a_star = a_nano_base['nano'].values
        bb_star = .0038*(WBANDS/470)**-1.4

        return chl*np.array([a_star.reshape(-1), bb_star])

    def gradient(*args):
        d_a = a_nano_base['nano'].values
        d_bb = .0038*(WBANDS/470)**-1.4

        return np.array([d_a.reshape(-1), d_bb])
    
    return iop, gradient

def micro(*args):
    '''
    micro IOP model
    
    chl - concentration in mg/m3
    '''
    def iop(chl=args[0]):
        # chl specific absorption 
        a_star = a_micro_base['micro'].values
        bb_star = .0004*(WBANDS/470)**.4

        return chl*np.array([a_star.reshape(-1), bb_star])
    
    def gradient(*args):
        d_a = a_micro_base['micro'].values
        d_bb = .0004*(WBANDS/470)**.4

        return np.array([d_a.reshape(-1), d_bb])
    
    return iop, gradient
