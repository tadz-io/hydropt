import numpy as np

class BandModelRegister:
    def __init__(self):
        self._models = {}

    def register(self, name, func):
        self._models[name] = func

    def get(self, model):
        func = self._models.get(model)
        if not func:
            raise ValueError(model)

        return func

def reflectance(rrs):
    return rrs
    
def normalize_reflectance(rrs):
    '''
    rrs - pandas Series
    '''
    rrs_normalized = rrs/np.trapz(rrs, x=rrs.index)

    return rrs_normalized


band_models = BandModelRegister()
band_models.register('rrs', reflectance)
band_models.register('normalize_rrs', normalize_reflectance)

class BandModel:
    def __init__(self, band_model='rrs'):
        self.band_model = band_models.get(band_model)

    def __call__(self, arg):
        '''
        arg - tuple passed to args in InversionModel().minimizer
        '''
        # transform y
        y_t = self.band_model(arg[0])
        # transform function
        f_t = lambda **x: self.band_model(arg[1](**x))
        
        return tuple([y_t, f_t])