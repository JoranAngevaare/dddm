"""Some functions that will (soon) be removed"""
from unittest import TestCase
import dddm
from immutabledict import immutabledict


class TestDepricated(TestCase):
    def test_get_priors(self):
        for prior in ('Pato_2010 '
                      'Evans_2019 '
                      'migdal_wide '
                      'low_mass '
                      'migdal_extremely_wide '
                      'low_mass_fixed').split():
            with self.subTest(prior=prior):
                self.assertIsInstance(dddm.priors.get_priors(prior), immutabledict)
        with self.assertRaises(NotImplementedError):
            dddm.priors.get_priors('some typo')
