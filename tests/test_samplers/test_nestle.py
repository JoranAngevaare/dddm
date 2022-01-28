import os.path
import tempfile
from unittest import TestCase
import dddm
import matplotlib.pyplot as plt
import numpy as np


# def test_nested_simple_nestle_earth_shielding():
#     fit_class = dddm.NestedSamplerStatModel('Xe')
#     fit_class.config['sampler'] = 'nestle'
#     fit_class.config['tol'] = 0.1
#     fit_class.config['nlive'] = 30
#     fit_class.config['max_iter'] = 1
#     fit_class.config['earth_shielding'] = True
#     fit_class.config['save_intermediate'] = True
#     log.info(f"Fitting for parameters:\n{fit_class.config['fit_parameters']}")
#     fit_class.run_nestle()
#     fit_class.get_summary()


class NestleTest(TestCase):
    def setUp(self) -> None:
        self.ct = dddm.test_context()

    def test_nestle_shielded_full_astrophysics(self,):
        self.test_nestle(halo_name='shielded_shm')

    def test_nestle(self, halo_name='shm', fit_parameters=('log_mass', 'log_cross_section',)):
        mw = 50
        cross_section = 1e-45
        sampler = self.ct.get_sampler_for_detector(
            wimp_mass=mw,
            cross_section=cross_section,
            sampler_name='nestle',
            detector_name='Xe_simple',
            prior="Pato_2010",
            halo_name=halo_name,
            detector_kwargs=None,
            halo_kwargs=None,
            sampler_kwargs=dict(nlive=100, tol=0.1, verbose=0),
            fit_parameters=fit_parameters,
        )
        sampler.run()
        results = sampler.get_summary()

        for i, (thing, expected, avg) in enumerate(
                zip(
                    fit_parameters,
                    [getattr(sampler, f) for f in fit_parameters],
                    results['best_fit']
                )):
            std = np.sqrt(results['cov_matrix'][i][i])
            nsigma_off = np.abs(expected - avg) / std
            message = f'For {thing}: expected {expected:.2f} yielded different results {avg:.2f} +/- {std:.2f}. Off by {nsigma_off:.1f} sigma'
            self.assertTrue(nsigma_off < 4, message)
