import tempfile
from unittest import TestCase
import dddm
import matplotlib.pyplot as plt


class MCMCTests(TestCase):
    def setUp(self) -> None:
        self.ct = dddm.test_context()

    def test_emcee(self):
        detector = self.ct.get_detector(n_energy_bins=10)

        fit_class = dddm.MCMCStatModel('Xe')
        fit_class.nwalkers = 10
        fit_class.nsteps = 20
        fit_class.verbose = 2

        with tempfile.TemporaryDirectory() as tmpdirname:
            fit_class.run_emcee()
            fit_class.show_corner()
            fit_class.show_walkers()
            fit_class.save_results(save_to_dir=tmpdirname)
            save_dir = fit_class.config['save_dir']
            r = dddm.samplers.emcee.load_chain_emcee(
                override_load_from=save_dir)
            dddm.samplers.emcee.emcee_plots(r)
            plt.clf()
            plt.close()


    def test_emcee_full_prior(self):
        fit_class = dddm.MCMCStatModel('Xe')
        fit_class.nwalkers = 10
        fit_class.nsteps = 20
        fit_class.verbose = 1

        with tempfile.TemporaryDirectory() as tmpdirname:
            fit_class.run_emcee()
            fit_class.show_corner()
            fit_class.show_walkers()
            fit_class.save_results(save_to_dir=tmpdirname)
            save_dir = fit_class.config['save_dir']
            r = dddm.samplers.emcee.load_chain_emcee(
                override_load_from=save_dir)
            dddm.samplers.emcee.emcee_plots(r, save=True, show=False)
            plt.clf()
            plt.close()


    def test_emcee_astrophysics_prior(self):
        fit_class = dddm.MCMCStatModel('Xe')
        fit_class.nwalkers = 10
        fit_class.nsteps = 20
        fit_class.set_fit_parameters(fit_class.known_parameters)

        with tempfile.TemporaryDirectory() as tmpdirname:
            fit_class.run_emcee()
            fit_class.show_corner()
            fit_class.show_walkers()
            fit_class.save_results(save_to_dir=tmpdirname)
            save_dir = fit_class.config['save_dir']
            r = dddm.samplers.emcee.load_chain_emcee(
                override_load_from=save_dir)
            dddm.samplers.emcee.emcee_plots(r, save=True, show=False)
            plt.clf()
            plt.close()
