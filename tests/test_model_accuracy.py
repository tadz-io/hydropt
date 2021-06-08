import unittest
import numpy as np
import pandas as pd
from hydropt.hydropt import PolynomialReflectance

POLYNOM_PATH = './hydropt/data/PACE_polynom_04_h2o.csv'
RTS_NADIR_SUN30 = './tests/fixtures/rts_nadir_sun30.csv'

def _relative_error(x, y):
    return np.abs((x-y)/y)

def _rmsre(x, y):
    '''root mean squared relative error'''
    n = len(x)
    diff_sq = ((x-y)/y)**2
    rmsre = np.sqrt(np.sum((diff_sq))/n)
    
    return rmsre

class TestDimensionPolynom(unittest.TestCase):
    def setUp(self):
        self.polynom = pd.read_csv(POLYNOM_PATH, index_col=0)

    def test_shape_polynom(self):
        self.assertTupleEqual(self.polynom.shape, (63, 15), "Should be (63, 15)")

class TestModelAccuracy(unittest.TestCase):
    def setUp(self):
        self.data = self._preprocess_data(RTS_NADIR_SUN30, by='no')

    def test_relative_error(self):
        # take average per waveband
        delta = _relative_error(self.data.rrs, self.data.rrs_hat).\
                    groupby('wavelength').\
                    mean()
        np.testing.assert_array_less(delta, .01,
            'relative error of less than 1% expected')


    def test_rmsre(self):
        rmsre = _rmsre(self.data.rrs, self.data.rrs_hat)
        self.assertLessEqual(rmsre, .01, 'RMSRE of less than 1% expected')

    def _preprocess_data(self, path, by):
        data = pd.read_csv(path, index_col=by)
        fwd_model = PolynomialReflectance().forward
        rrs_hat = data\
                    .groupby(by)\
                    .apply(lambda x: pd.Series(
                        fwd_model([x.a, x.bb]), index=x.wavelength))\
                    .melt(value_name='rrs_hat', ignore_index=False)
        
        return self._merge(data, rrs_hat, on=[by, 'wavelength'])

    @staticmethod
    def _merge(x, y, on):
        x, y = [i.reset_index().set_index(on) for i in [x,y]]

        return pd.merge(x, y, left_index=True, right_index=True)

pass

if __name__ == '__main__':
    unittest.main()