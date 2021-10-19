from hydropt.utils import recurse
import unittest
import numpy as np
from hydropt.hydropt import BioOpticalModel, check_iop_dims
from hydropt.bio_optics import clear_nat_water

def dummy_model_one(*args):
    return args[0]*np.ones([2,5])

def dummy_model_three(*args):
    return args[0]*np.ones([2,6])

def dummy_model_four(*args):
    def inner_one(x=args[0]):
        return x*np.ones([2,5])

    def inner_two(x=args[0]):
        return x*np.ones([2,5])

    return inner_one, inner_two

def dummy_model_full(*args):
    def inner_one(x=args[0]):
        return x*np.ones([2,63])

    def inner_two(x=args[0]):
        return x*np.ones([2,63])

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
            },
            {
                'water': clear_nat_water,
                'm2': dummy_model_full
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

class TestCheckIopDims(unittest.TestCase):

    def test_check_dims_unequal(self):
        for i in test_cases.get('unequal_dims'):
            self.assertRaises(ValueError, check_iop_dims, None, **i)

    def test_check_dims_equal(self):
        # no. of wavebands
        wb = [range(5), range(5), range(6)]
        ndims = [2, 1, 1]
        for (i,j,k) in zip(wb, test_cases.get('equal_dims'), ndims):
            self.assertEqual(check_iop_dims(i, **j), k)

class TestBioOpticalModel(unittest.TestCase):
    def setUp(self):
        self.bom = BioOpticalModel()
        self.wavebands = [i for i in range(5)]
    
    def test_iop_model(self):
        self.bom.set_iop(wavebands=self.wavebands, **test_cases.get('equal_dims')[1])
        self.assertListEqual(list(self.bom.iop_model.keys()), ['m1', 'm2'])
        # dictionary with gradient models should be empty
        self.assertListEqual(list(self.bom.gradient.keys()), [])

    def test_grad_model(self):
        self.bom.set_iop(wavebands=self.wavebands, **test_cases.get('equal_dims')[0])
        # dictionary w. iop models should be filled
        self.assertListEqual(list(self.bom.iop_model.keys()), ['m1', 'm2'])
        # dict w. gradient models should be filled
        self.assertListEqual(list(self.bom.gradient.keys()), ['m1', 'm2'])

    def test_get_iop(self):
        self.bom.set_iop(wavebands=self.wavebands, **test_cases.get('equal_dims')[0])
        iops = self.bom.get_iop(m1=2, m2=3)
        np.testing.assert_array_equal(iops, np.array([
            2*np.ones([2,5]),
            3*np.ones([2,5])]))

    def test_get_gradient(self):
        self.bom.set_iop(wavebands=self.wavebands, **test_cases.get('equal_dims')[0])
        iops = self.bom.get_gradient(m1=2, m2=3)
        np.testing.assert_array_equal(iops, np.array([
            2*np.ones([2,5]),
            3*np.ones([2,5])]))

    def test_sum_iop(self):
        self.bom.set_iop(wavebands=range(63), **test_cases.get('equal_dims')[3])
        np.testing.assert_array_equal(self.bom.sum_iop(), clear_nat_water(None)[0]())
        np.testing.assert_array_equal(self.bom.sum_iop(m2=2), clear_nat_water(None)[0]()+2)
        np.testing.assert_array_equal(self.bom.sum_iop(m2=2, incl_water=False), dummy_model_full(2)[0]())

if __name__ == '__main__':
    unittest.main()