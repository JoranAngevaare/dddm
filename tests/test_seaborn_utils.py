import dddm
import unittest
import numpy as np
from hypothesis import given, settings, strategies


class TestSeabornExtractor(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # @settings(max_examples=10)
    # @given(strategies.floats(1e-3, 1),
    #        strategies.floats(1.1,10),
    #        strategies.integers(1,100),
    #        strategies.floats(0.1, 0.9),
    #        strategies.integers(2, 10_000),
    #        )
    def test_get_xy(self, _min=0.01, _max=10, n=10, var=0.5, size=300):
        sigmas = np.linspace(_min, _max, n)
        errs = np.array([dddm.one_sigma_area(*self.get_xy(s, var=var, size=size))
                         for s in sigmas])
        # Very approximate, make sure we are less than a factor of 2
        # wrong for the 1 sigma calculation
        assert np.all(np.array(errs) / sigmas < 2)
        assert np.all(np.array(errs) / sigmas > 0.5)

    def get_xy(self, sigma, mean=(0, 2), var=0., size=300):
        """
        Get a simple gaussian smeared distribution based on a covariance matrix
        :param sigma: The amplitude of the blob
        :param var: off diagonal elements of the covariance matrix
        :return: Random samples of size <size>
        """

        cov = [(sigma / np.pi, var * sigma / np.pi),
               (var * sigma / np.pi, sigma / np.pi)]
        x, y = np.random.multivariate_normal(mean, cov, size=int(size)).T
        return x, y
