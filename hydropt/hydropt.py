import numpy as np
import pandas as pd
import itertools
import warnings
from tqdm import tqdm_notebook
from sklearn.preprocessing import PolynomialFeatures
from lmfit import minimize, Minimizer, Parameters

OLCI_WBANDS = np.array([400,412.5,442.5,490,510,560,620,665,673.75,681.25,708.75])
HSI_WBANDS = np.linspace(400, 710, 100)

siop_psc = pd.read_csv('hydropt/data/psc_absorption_se_uitz_2008.csv')

pico_siop = siop_psc[['wavelength','pico', 'pico_se']]
pico_siop.set_index('wavelength', inplace=True)

nano_siop = siop_psc[['wavelength','nano', 'nano_se']]
nano_siop.set_index('wavelength', inplace=True)

micro_siop = siop_psc[['wavelength','micro', 'micro_se']]
micro_siop.set_index('wavelength', inplace=True)

def interpolate_to_wavebands(data, wavelength='wavelength'):
    '''
    interpolates SIOPs to OLCI/MERIS wavebands
    '''
    data.reset_index(inplace=True)
    # add OLCI/MERIS wavebands to phytop SIOP wavelengths
    wband_merge = np.unique(sorted(np.append(data[wavelength], OLCI_WBANDS)))
    
    data.set_index(wavelength, inplace=True)
    data = data.reindex(wband_merge)\
        .astype(float)\
        .interpolate(method='slinear', fill_value='extrapolate', limit_direction='both')\
        .reindex(OLCI_WBANDS)
    
    return data

# do interpolation to OLCI wavebands
pico_siop_olci = interpolate_to_wavebands(pico_siop)
nano_siop_olci = interpolate_to_wavebands(nano_siop)
micro_siop_olci = interpolate_to_wavebands(micro_siop)

# set absorption @ 708nm to zero -> linear extrapolation gives negative values
pico_siop_olci.loc[708.75] = 0
nano_siop_olci.loc[708.75] = 0
micro_siop_olci.loc[708.75] = 0

def pico_iop(chl):
    '''
    pico IOP model
    
    chl - concentration in mg/m3
    '''
    # chl specific absorption 
    a_star = pico_siop_olci['pico'].values
    b_star = (.0038*(OLCI_WBANDS/470)**-1.4)/.014
    
    return chl*np.array([a_star.reshape(-1), b_star])

def nano_iop(chl):
    '''
    nano IOP model
    
    chl - concentration in mg/m3
    '''
    # chl specific absorption 
    a_star = nano_siop_olci['nano'].values
    b_star = (.0038*(OLCI_WBANDS/470)**-1.4)/.014
    
    return chl*np.array([a_star.reshape(-1), b_star])

def micro_iop(chl):
    '''
    micro IOP model
    
    chl - concentration in mg/m3
    '''
    # chl specific absorption 
    a_star = micro_siop_olci['micro'].values
    b_star = (.0004*(OLCI_WBANDS/470)**.4)/.014
    
    return chl*np.array([a_star.reshape(-1), b_star])

def cdom_iop(a_440):
    '''
    IOP model for CDOM
    '''
    return np.array([a_440*np.exp(-0.017*(OLCI_WBANDS-440)), np.zeros(len(OLCI_WBANDS))])

def nap_iop(spm):
    '''
    IOP model for NAP
    '''
    # vectorized
    return spm*np.array([(.041*.75*np.exp(-.0123*(OLCI_WBANDS-443))), 0.57*(550/OLCI_WBANDS)])

def total_iop(ag, spm, pico, nano, micro):
    '''
    total_iop calculates total absorption and scatter coefficients (excluding water)
    
    ag - CDOM absorption at 440nm
    spm - SPM concentration in g/m3
    pico - chl concentration in mg/m3 for picos
    nano - chl concentration in mg/m3 for nanos
    micro - chl concentration in mg/m3 for micros
    '''
    
    return cdom_iop(ag), nap_iop(spm), pico_iop(pico), nano_iop(nano), micro_iop(micro)

# gradient for CDOM
def grad_cdom_iop():
    '''
    Gradient of CDOM IOP model
    ''' 
    d_a = np.exp(-.017*(OLCI_WBANDS-440))
    d_b = np.zeros(len(d_a))
    return np.array([d_a, d_b])

# gradient for NAP
def grad_nap_iop():
    '''
    Gradient of NAP IOP model
    '''
    d_a = .03075*np.exp(-.0123*(OLCI_WBANDS-443))
    d_b = 0.57*(550/OLCI_WBANDS)
    
    return np.array([d_a, d_b])

# gradient for pico
def grad_pico_iop():
    '''
    Gradient of pico IOP model
    '''
    # chl specific absorption 
    d_a = pico_siop_olci['pico'].values
    d_b = (.0038*(OLCI_WBANDS/470)**-1.4)/.014
    
    return np.array([d_a.reshape(-1), d_b])

# gradient for nano
def grad_nano_iop():
    '''
    Gradient nano IOP model
    '''
    # chl specific absorption 
    d_a = nano_siop_olci['nano'].values
    d_b = (.0038*(OLCI_WBANDS/470)**-1.4)/.014
    
    return np.array([d_a.reshape(-1), d_b])

# gradient for micro
def grad_micro_iop():
    '''
    Gradient micro IOP model
    '''
    # chl specific absorption 
    d_a = micro_siop_olci['micro'].values
    d_b = (.0004*(OLCI_WBANDS/470)**.4)/.014
    
    return np.array([d_a.reshape(-1), d_b])

## example of nap IOP model
# def nap(*args):
#     '''
#     IOP model for NAP
#     '''
#     # vectorized
#     def iop(spm=args):
#         return spm*np.array([(.041*.75*np.exp(-.0123*(OLCI_WBANDS-443))), 0.57*(550/OLCI_WBANDS)])
    
#     def gradient():
#         d_a = .03075*np.exp(-.0123*(OLCI_WBANDS-443))
#         d_b = 0.57*(550/OLCI_WBANDS)
        
#         return np.array([d_a, d_b])
    
#     return iop, gradient

class IOP_model:
    def __init__(self):
        self.name = None
        self._wavebands = None
        self.iop_model = []
    
    def set_iop(self, wavebands, *args):
        #get iop models
        models = [i for i in args]
        # get length of siop vector
        n_waveband = f(1)[0]().shape[1]
        if n_waveband != len(wavebands):
            raise ValueError('number of wavebands do not match with length of IOP vectors')
        else:
            self._wavebands = wavebands
            self.iop_model = self.iop_model.append(f)
    
    def get_iop(self, *args):
        return self.iop_model(*args)[0]()
    
    def get_gradient(self, *args):
        return self.iop_model(*args)[1]()
    
    def check_wavelen(self, wavebands, *args):
        models = [i for i in args]
        # check dimensions of models; skip gradients for now
        dims = [i(1)[0]().shape[1] for i in models]
        # check if all dimension match
        if len(set(dims)) != 1:
            raise ValueError('length of IOP vectors do not match')
        elif len(dims) == len(wavebands):
            raise ValueError('number of wavebands do not match with length of IOP vectors')
        
        return True

class Hydropt:
    def __init__(self, y):
        self.y = y
        self._wavebands = None
        self._id_wbands = None
        self.parameters = None
        self.polynom_coef = None
        self.polynom_powers = None
        self.set_polynom_coef()
        self.set_wavebands(OLCI_WBANDS)
    
    def set_wavebands(self, wavebands):
        # get indices for matching wavebands
        id_wbands = [key for key, val in enumerate(OLCI_WBANDS)
                          if val in set(wavebands)]
        # check if wavebands match OLCI wavebands
        if len(id_wbands) != len(wavebands):
            warnings.warn('provided wavebands do not match OLCI wavebands')
        else:
            self._wavebands = wavebands
            self._id_wbands = id_wbands
        
    def set_parameters(self, parameters):
        self.parameters = parameters
    
    def set_polynom_coef(self):
        # read polynomial coefficients
        polynom_coef = pd.read_csv('data/coeff_poly_deg_4_ext.csv', index_col = 0)
        # check if number of polynomial features match
        if (polynom_coef.shape[1] == PolynomialFeatures(degree=4).fit([[1,1]]).n_output_features_):
            #set polynomial coefficients
            self.polynom_coef = polynom_coef
            #get exponents of polynom terms
            self.polynom_powers = PolynomialFeatures(degree=4).fit([[1,1]]).powers_
        else:
            warnings.warn("polynomial features do not match")
    
    def hydrolight_polynom(self, x, degree=4):
        '''
        Forward model using polynomial fit to Hydrolight simulations

        x[0]: total log absorption at wavelength i
        x[1]: total log backscatter at wavelength i
        
        returns log(Rrs)
        '''
        # get polynomial features
        ft = PolynomialFeatures(degree=degree).fit_transform(x.T)
        # get polynomial coefficients
        c = self.polynom_coef
        # calculate log(Rrs)
        log_Rrs = np.dot(c, ft.T).diagonal()

        #return np.exp(log_Rrs)
        return log_Rrs
    
    def forward(self, ag, spm, pico, nano, micro):
        '''
        Forward model: from component concentration to Rrs
        forward() uses hydrolight_polynom() and IOP spectral models
        to calculate Rrs
        
        ag - CDOM absorption at 440nm
        spm - SPM concentration in g/m3
        pico - chl concentration in mg/m3 for picos
        nano - chl concentration in mg/m3 for nanos
        micro - chl concentration in mg/m3 for micros
        
        returns Rrs
        '''
        # calculate log of total IOPs
        log_iop = np.log(np.nansum(total_iop(ag,spm,pico,nano,micro), axis=0))
        # calculate Rrs at wavebands
        rrs = np.exp(self.hydrolight_polynom(log_iop))[self._id_wbands]
        
        return rrs
    
    def der_2d_polynomial(self, x):
        '''
        derivative of 2 variable polynomial

        x: points at wich to evaluate the derivative; a nx2 array were n is number of wavebands
        '''
        c = self.polynom_coef
        p = self.polynom_powers
        # check if dimensions match of polynomial and x
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
    
    def jacobian_hydrolight_rrs(self, x):
        '''
        Jacobian at [x] of the remote-sensing reflectance (Rrs),
        using the Hydrolight polynomial as forward model, w.r.t concentration of constituents

        x[0]: cdom absorption at 440 nm
        x[1]: spm concentration in g/m^3
        x[2]: pico concentration in mg/m^3
        x[3]: nano concentration in mg/m^3
        x[4]: micro concentration in mg/m^3
        '''
        if isinstance(x, Parameters):
            x = [x['ag'].value, x['spm'].value, x['pico'].value, x['nano'].value, x['micro'].value]

        # gradients of IOP models at every waveband
        grad_iops = np.array([grad_cdom_iop(), grad_nap_iop(), grad_pico_iop(), grad_nano_iop(), grad_micro_iop()])
        iops = np.sum(total_iop(*x), axis=0)
        # calculate rrs
        rrs = np.exp(self.hydrolight_polynom(np.log(iops)))
        # (dln(a)/da) * (dRrs/dln(Rrs)) = Rrs/a
        # (dln(b)/db) * (dRrs/dln(Rrs)) = Rrs/b; 
        # do element-wise multiplication
        grad_log_iops = np.array([rrs/iops[0],rrs/iops[1]]) * grad_iops
        # evaluate jacobian of hydrolight polynomial at x
        grad_polynom = self.der_2d_polynomial(np.log(iops.T))
        # initialize empty jacobian matrix
        jac = np.empty([grad_polynom.shape[1], grad_log_iops.shape[0]])  
        for c, (i,j) in enumerate(zip(grad_polynom.T, grad_log_iops.T)):
            jac[c] = np.dot(i,j)

        return jac[self._id_wbands]
    
    def levenberg_marquardt(self, y):
        # tolerance in estimation
        etol = np.Inf
        all_x_i = [self.parameters]
        all_f_i = [self.loss_function(y)(self.parameters)]

        def store(param, n_iter, resid):
            all_x_i.append(param)
            all_f_i.append(resid)

        def standard_error(cov):
            '''
            cov - covariance matrix
            '''
            return np.sqrt(cov.diagonal())

        def chi_squared(residuals, sigma_sq):
            '''
            residuals - residuals of fit
            sigma_sq - expected variance
            '''
            return np.sum([i**2/sigma_sq for i in residuals])
        
        # call optimization routine
        x_hat = minimize(self.loss_function(y),
                         self.parameters,
                         iter_cb=None,
                         Dfun = self.jacobian_hydrolight_rrs)
        # get parameter estimates
        x_hat_params = [x_hat.params.get(i).value for i in ['ag','spm','pico','nano','micro']]
        # calculat relative errors
        try:
            x_hat_err = standard_error(x_hat.covar)/x_hat_params
        except AttributeError:
            x_hat_err = [etol for i in range(len(x0))]
        ag, spm, pico, nano, micro = [i if j<etol else np.nan for i,j in zip(x_hat_params, x_hat_err)]
        # create dictionary of parameter estimates
        out = {
            'cdom': ag,
            'nap': spm,
            'pico': pico,
            'nano': nano,
            'micro': micro}

        return out
    
    def loss_function(self, y):
        '''

        create scalar cost function by summing residuals
        y - target
        '''

        def minimizer(x0):
            ag = x0['ag'].value
            spm = x0['spm'].value
            pico = x0['pico'].value
            nano = x0['nano'].value
            micro = x0['micro'].value  
            # calculate log of total IOPs
            log_iop = np.log(np.sum(total_iop(ag,spm,pico,nano,micro), axis=0))
            # calculate Rrs at wavebands
            rrs = np.exp(self.hydrolight_polynom(log_iop))[self._id_wbands]
            # calculate residuals
            residual = rrs-y

            return residual

        return minimizer
    
    def invert(self):     
        if len(self.y.shape) == 2:
            out = []
            for i in tqdm_notebook(self.y, total=self.y.shape[0]):
                out.append(self.levenberg_marquardt(i))
        else:
            out = self.levenberg_marquardt(self.y)
            
        return out
