import os.path
import tempfile
from unittest import TestCase
import dddm
import matplotlib.pyplot as plt
import numpy as np


class MCMCTests(TestCase):
    def setUp(self) -> None:
        self.ct = dddm.test_context()

    def test_emcee(self, fit_parameters=('log_mass', 'log_cross_section',)):
        mw = 50
        cross_section = 1e-45
        sampler = self.ct.get_sampler_for_detector(
            wimp_mass=mw,
            cross_section=cross_section,
            sampler_name='emcee',
            detector_name='Xe_simple',
            prior ="Pato_2010",
            halo_name='shm',
            detector_kwargs=None,
            halo_kwargs=None,
            sampler_kwargs= dict(nwalkers=30, nsteps=50, verbose=0),
            fit_parameters=fit_parameters,
        )
        with tempfile.TemporaryDirectory() as tmpdirname:
            sampler.run()
            sampler.show_corner()
            sampler.show_walkers()
            sampler.save_results(save_to_dir=tmpdirname)
            save_dir = sampler.config['save_dir']
            r = dddm.samplers.emcee.load_chain_emcee(os.path.split(save_dir)[0])
            dddm.samplers.emcee.emcee_plots(r)
            plt.clf()
            plt.close()
            samples = sampler._get_chain_flat_chain()
            mw_res = 10**samples[:, 0]
            sigma_res = 10**samples[:, 1]
            for thing, expected, values in zip(('mass', 'cross-section'),
                                                (mw, cross_section),
                                                (mw_res, sigma_res)):
                avg = np.mean(values)
                std = np.std(values)
                nsigma_off = (expected-avg)/std
                message = f'Expected {thing} of {expected} yielded different results {avg} +/- {std}. Off by {nsigma_off} sigma'
                self.assertTrue(nsigma_off < 3, message)

    def test_emcee_astrophysics_prior(self):
        self.test_emcee(fit_parameters=dddm.statistics.get_param_list())
