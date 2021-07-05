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

class TestRecurse(unittest.TestCase):
    def test_unequal_dims(self):
        test_cases = [
            {
                'm1': dummy_model_one,
                'm2': dummy_model_two
            },
            {
                'm1': dummy_model_one,
                'm2': dummy_model_three
            }
            ]
        
        for i in test_cases:
            self.assertRaises(ValueError, recurse, i)

    def test_equal_dims(self):
        test_cases = [
            # {
            #     'm1': dummy_model_four
            # },
            {
                'm1': dummy_model_one,
                'm2': dummy_model_one
            },
            {
                'm1': dummy_model_three,
                'm2': dummy_model_three
            }
            ]
        
        for i in test_cases:
            self.assertIsInstance(recurse(i), np.ndarray)

if __name__ == '__main__':
    unittest.main()