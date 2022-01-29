import unittest
import numpy as np
from hydropt.utils import apply_along_axis

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


if __name__ == '__main__':
    unittest.main()