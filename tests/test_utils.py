import unittest
import numpy as np
from lmfit import minimizer, Parameters
from hydropt.utils import LmfitResultsToArray, apply_along_axis
from hydropt.array_wrappers import LabeledArray

lmfit_result_kws = dict(
    method=None,
    nfev=None,
    ndata=None,
    nvarys=None,
    chisqr=0,
    redchi=0,
    aic=0,
    bic=0,
    params=Parameters()
)

class TestApplyAlongAxis(unittest.TestCase):
    def setUp(self, dims=(2,3,4)):
        self.dims = dims
        self.ndarray = np.array(
            list(range(np.prod(dims)))).reshape(dims)

    def test_apply_mean(self):
        for ax in range(len(self.dims)):
            np.testing.assert_array_equal(
                apply_along_axis(np.mean, axis=ax, arr=self.ndarray),
                np.mean(self.ndarray, axis=ax)
            )
    
    @unittest.skip
    def test_apply_minmax(self):
        minmax = lambda x: np.array([min(x), max(x)])
        for ax in range(len(self.dims)):
            np.testing.assert_array_equal(
                apply_along_axis(minmax, axis=ax, arr=self.ndarray),
                np.array([
                    np.min(self.ndarray, axis=ax),
                    np.max(self.ndarray, axis=ax)])
            )

class TestLmfitResultsArray(unittest.TestCase):
    def setUp(self):
        self.minimizer_result = minimizer.MinimizerResult(**lmfit_result_kws)
    
    def test_attrs(self):
        self.assertDictEqual(
            LmfitResultsToArray(self.minimizer_result).__dict__,
            lmfit_result_kws
        )
        

class TestLabeledArray(unittest.TestCase):
    def test_empty_label(self):
        self.assertEqual(LabeledArray([]).label, ())

    def test_label(self):
        self.assertEqual(
            LabeledArray([0, 1], label=('label1', 'label2')).label,
            ('label1', 'label2'))

    def test_unequal_dims(self):
        with self.assertRaises(ValueError):
            LabeledArray([0, 1], label=(''))

if __name__ == '__main__':
    unittest.main()