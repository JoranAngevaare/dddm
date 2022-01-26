import pandas as pd

import dddm
from unittest import TestCase


class TestExampleSpectra(TestCase):
    def test_simple_spectrum(self, spectrum_class=dddm.GenSpectrum):
        detector_class = dddm.examples.XenonSimple()
        detector_class._check_class()

        shm = dddm.SHM()

        spectrum_generator = spectrum_class(
            dark_matter_model=shm,
            experiment=detector_class)
        assert detector_class.e_min_kev == spectrum_generator.e_min_kev, "getattr of spectrum class is not working well"
        with self.assertRaises(AttributeError):
            _ = spectrum_generator.not_an_attribute

        test_spectrum = spectrum_generator.get_data(wimp_mass=10,  # gev/c2
                                                    cross_section=1e-45,
                                                    )
        self.assertTrue(len(test_spectrum))
        self.assertTrue(isinstance(test_spectrum, pd.DataFrame))

    def test_simple_detector_spectrum(self):
        self.test_simple_spectrum(spectrum_class=dddm.DetectorSpectrum)
