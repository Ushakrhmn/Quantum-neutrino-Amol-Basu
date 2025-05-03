# unit tests for some of the methods in the module

import unittest
from helpers import extrapolate_to_zero
import numpy as np

class TestExtrapolation(unittest.TestCase):
    def test_exponential(self):
        meas = [1, np.exp(1), np.exp(2)]
        lambd = [0, 1, 2]
        result = extrapolate_to_zero(meas, lambd, method='exponential')
        expected = 1
        self.assertAlmostEqual(result, expected, places=8)
    def test_exponential_offset(self):
        meas = [np.exp(1), np.exp(2), np.exp(3)]
        lambd = [1, 2, 3]
        result = extrapolate_to_zero(meas, lambd, method='exponential')
        expected = 1
        self.assertAlmostEqual(result, expected, places=8)
    def test_linear(self):
        meas = [1, 2, 3]
        lambd = [0, 1, 2]
        result = extrapolate_to_zero(meas, lambd, method='linear')
        expected = 1
        self.assertAlmostEqual(result, expected, places=8)
    def test_linear_offset(self):
        meas = [2, 3, 4]
        lambd = [1, 2, 3]
        result = extrapolate_to_zero(meas, lambd, method='linear')
        expected = 1
        self.assertAlmostEqual(result, expected, places=8)
    def test_quadratic(self):
        meas = [1, 4, 9]
        lambd = [0, 1, 2]
        result = extrapolate_to_zero(meas, lambd, method='quadratic')
        expected = 1
        self.assertAlmostEqual(result, expected, places=8)
    def test_quadratic_offset(self):
        meas = [4, 9, 16]
        lambd = [1, 2, 3]
        result = extrapolate_to_zero(meas, lambd, method='quadratic')
        expected = 1
        self.assertAlmostEqual(result, expected, places=8)

if __name__ == '__main__':
    unittest.main()