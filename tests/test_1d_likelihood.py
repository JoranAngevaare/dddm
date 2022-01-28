"""
Test if the 1D likelihood returns a value that is close to the set benchmark value
"""
import dddm
import numpy as np
from hypothesis import given, settings, strategies
from unittest import skipIf, TestCase
from tqdm import tqdm

_known_detectors = dddm.test_context().detectors
_known_priors = 'Pato_2010 Evans_2019 migdal_wide low_mass migdal_extremely_wide low_mass_fixed'.split()


@skipIf(not dddm.utils.is_installed('pymultinest'), 'pymultinest is not installed')
@skipIf(dddm.utils.is_windows(), "Multinest only works on linux")
class TestLikelihoodMinimum(TestCase):
    def setUp(self) -> None:
        self.ct = dddm.test_context()

    @settings(deadline=None, max_examples=10)
    @given(strategies.floats(0.1, 50),
           strategies.integers(-47, -43),
           strategies.integers(0, len(_known_detectors) - 1),
           strategies.integers(0, len(_known_priors) - 1),
           strategies.booleans()
           )
    def test_likelihood_converges(self, mass, sigma, detector_i, prior_i, include_astrophysics):
        """Test that a 1D likelihood scan actually returns the maximum at the set value"""
        detector_name = _known_detectors[detector_i]
        prior_name = _known_priors[prior_i]
        if include_astrophysics:
            fit_params = ('log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density',)
        else:
            fit_params = ('log_mass', 'log_cross_section')
        sampler = self.ct.get_sampler_for_detector(
            wimp_mass=mass,
            cross_section=sigma,
            sampler_name='multinest_combined',
            detector_name=[detector_name],
            prior=prior_name,
            halo_name='shm',
            detector_kwargs=None,
            halo_kwargs=None,
            sampler_kwargs=dict(nlive=100, tol=0.1, verbose=0, detector_name='test_combined'),
            fit_parameters=fit_params,
        )
        sampler.set_benchmark(mass, sigma)
        sampler.set_prior(prior_name)
        sampler._fix_parameters()

        # Check that all the subconfigs are correctly set
        for c in sampler.sub_classes:
            assert c.log_cross_section == sigma
            assert c.log_mass == np.log10(mass)
            assert c.config['prior'] == dddm.get_priors(prior_name)
            assert c.benchmark_values is not None

        if benchmark_all_zero := not np.any(sampler.sub_classes[0].benchmark_values):
            print('If everything is zero, I don\'t have to check if we converge')
            return

        # Do the parameter scan
        likelihood_scan = []
        # Hardcoding the range of parameters to scan for reproducibility
        sigma_scan = np.linspace(sigma - 0.2, sigma + 0.2, 30)

        for s in tqdm(sigma_scan, desc='Cross-section scan'):
            ll = sampler.sub_classes[0]._log_probability_nested([np.log10(mass), s])
            likelihood_scan.append(ll)

        max_likelihood_index = np.argmax(likelihood_scan)
        max_is_close_to_true = np.isclose(
            sigma_scan[max_likelihood_index],
            sigma,
            # if the binning is course, the result will also be. Allow some tolerance
            atol=sigma_scan[1] - sigma_scan[0]
        )

        if not max_is_close_to_true:
            # This is a reason to break generally
            print(sigma_scan)
            print(likelihood_scan)

            # Check if the likelihood all has the same values, then we don't have to fail
            likelihood_is_flat = np.all(np.isclose(likelihood_scan[0], likelihood_scan))
            if not likelihood_is_flat:
                raise ValueError(f'{detector_name}-{prior_name}\tm:{mass}\ts:{sigma} failed')
