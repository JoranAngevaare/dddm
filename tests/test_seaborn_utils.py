import DirectDmTargets as dddm
import unittest
import os
import pymongo
import numpy as np


class TestMongoDownloader(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_xy(self):
        sigmas = np.linspace(0.0001, 10, 20)
        errs = np.array([dddm.one_sigma_area(*self.get_xy(s))[0] for s in sigmas])
        # Very approximate, make sure we are less than a factor of 2
        # wrong for the 1 sigma calculation
        assert np.all(np.array(errs) / sigmas < 2)
        assert np.all(np.array(errs) / sigmas > 0.5)

    @staticmethod
    def get_xy(sigma, mean=(0, 2), var=0., size=300):
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
