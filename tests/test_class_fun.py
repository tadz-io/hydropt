from hydropt.utils import recurse
import unittest
import numpy as np
from hydropt.hydropt import BioOpticalModel

def dummy_model_one(*args):
    return np.ones([2,5])

def dummy_model_two(*args):
    return np.ones([3,5])

def dummy_model_three(*args):
    return np.ones([2,6])

def dummy_model_four(*args):
    def inner_one(*args):
        return np.ones([2,5])

    def inner_two(*args):
        return np.ones([2,5])

    return inner_one, inner_two

test_cases = {
    'equal_dims':
        [
            {
                'm1': dummy_model_four,
                'm2': dummy_model_four
            },
            {
                'm1': dummy_model_one,
                'm2': dummy_model_one
            },
            {
                'm1': dummy_model_three,
                'm2': dummy_model_three
            }
        ],
    'unequal_dims':
        [
            {
                'm1': dummy_model_one,
                'm2': dummy_model_three  
            }
        ]

}
class TestRecurse(unittest.TestCase):
    def test_unequal_dims(self):      
        for i in test_cases.get('unequal_dims'):
            self.assertRaises(ValueError, recurse, i.values())

    def test_equal_dims(self):       
        for i in test_cases.get('equal_dims'):
            self.assertIsInstance(recurse(i.values()), np.ndarray)

    def test_shape_array(self):
        test_cases_eq = test_cases.get('equal_dims')
        dims = [(2,2,2,5), (2,2,5), (2,2,6)]

        for (i,j) in zip(test_cases_eq, dims):
            self.assertTupleEqual(recurse(i.values()).shape, j)

class TestBioOpticalModel(unittest.TestCase):

    def test_check_dims_unequal(self):
        model = BioOpticalModel()
        for i in test_cases.get('unequal_dims'):
            self.assertRaises(ValueError, model._check_dims, None, **i)

    def test_check_dims_equal(self):
        model = BioOpticalModel()
        # no. of wavebands
        wb = [range(5), range(5), range(6)]
        ndims = [2, 1, 1]
        for (i,j,k) in zip(wb, test_cases.get('equal_dims'), ndims):
            self.assertEqual(model._check_dims(i, **j), k)



if __name__ == '__main__':
    unittest.main()