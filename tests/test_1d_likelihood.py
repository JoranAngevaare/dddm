"""
Test if the 1D likelihood returns a value that is close to the set benchmark value
"""
import dddm
import numpy as np
from hypothesis import given, settings, strategies
from unittest import TestCase, skipIf
from tqdm import tqdm

_known_detectors = dddm.test_context().detectors
_known_priors = 'Pato_2010 Evans_2019 migdal_wide low_mass migdal_extremely_wide low_mass_fixed'.split()


class TestLikelihoodMinimum(TestCase):
    def setUp(self) -> None:
        self.ct = dddm.test_context()

    @skipIf(*dddm.test_utils.skip_long_test())
    @settings(deadline=None, max_examples=3)
    @given(strategies.floats(0.1, 50),
           strategies.integers(-47, -43),
           strategies.integers(0, len(_known_priors) - 1),
           strategies.booleans()
           )
    def test_all_detectors(self, mass, sigma, prior_i, include_astrophysics):
        self._likelihood_converges_inner(mass=mass,
                                         sigma=sigma,
                                         prior_i=prior_i,
                                         include_astrophysics=include_astrophysics,
                                         detector_name=_known_detectors,
                                         nbins=10,
                                         )

    @settings(deadline=None, max_examples=3)
    @given(strategies.floats(0.1, 50),
           strategies.integers(-47, -43),
           strategies.integers(0, len(_known_priors) - 1),
           strategies.booleans()
           )
    def test_examples(self, mass, sigma, prior_i, include_astrophysics):
        self._likelihood_converges_inner(mass=mass,
                                         sigma=sigma,
                                         prior_i=prior_i,
                                         include_astrophysics=include_astrophysics,
                                         detector_name=['Xe_simple', 'Ar_simple', 'Ge_simple'],
                                         )

    @skipIf(*dddm.test_utils.skip_long_test())
    @settings(deadline=None, max_examples=1)
    @given(strategies.floats(0.1, 50),
           strategies.integers(-47, -43),
           strategies.integers(0, len(_known_priors) - 1),
           strategies.booleans()
           )
    def test_nr(self, mass, sigma, prior_i, include_astrophysics):
        self._likelihood_converges_inner(mass=mass,
                                         sigma=sigma,
                                         prior_i=prior_i,
                                         include_astrophysics=include_astrophysics,
                                         detector_name=['SuperCDMS_HV_Ge_NR', 'SuperCDMS_HV_Si_NR',
                                                        'SuperCDMS_iZIP_Ge_NR',
                                                        'SuperCDMS_iZIP_Si_NR', 'XENONnT_NR'],
                                         )

    @settings(deadline=None, max_examples=5)
    @given(strategies.floats(0.1, 50),
           strategies.integers(-40, -30),
           strategies.integers(0, len(_known_priors) - 1),
           strategies.booleans()
           )
    def test_examples_shielded(self, mass, sigma, prior_i, include_astrophysics):
        self._likelihood_converges_inner(mass=mass,
                                         sigma=sigma,
                                         prior_i=prior_i,
                                         include_astrophysics=include_astrophysics,
                                         detector_name=['Xe_simple'],
                                         halo_name='shielded_shm',
                                         halo_kwargs={'location': 'XENON'},
                                         nbins=30,
                                         )

    @skipIf(*dddm.test_utils.skip_long_test())
    @settings(deadline=None, max_examples=1)
    @given(strategies.floats(0.1, 50),
           strategies.integers(-47, -43),
           strategies.integers(0, len(_known_priors) - 1),
           strategies.booleans()
           )
    def test_migdal(self, mass, sigma, prior_i, include_astrophysics):
        self._likelihood_converges_inner(mass=mass,
                                         sigma=sigma,
                                         prior_i=prior_i,
                                         include_astrophysics=include_astrophysics,
                                         detector_name=['SuperCDMS_HV_Ge_Migdal',
                                                        'SuperCDMS_HV_Si_Migdal',
                                                        'SuperCDMS_iZIP_Ge_Migdal',
                                                        'SuperCDMS_iZIP_Si_Migdal',
                                                        'XENONnT_Migdal'],
                                         nbins=10
                                         )

    def _likelihood_converges_inner(self, mass, sigma, detector_name, prior_i, include_astrophysics,
                                    nbins=30, **kwargs):
        """Test that a 1D likelihood scan actually returns the maximum at the set value"""
        print(detector_name)
        prior_name = _known_priors[prior_i]
        if include_astrophysics:
            fit_params = ('log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density',)
        else:
            fit_params = ('log_mass', 'log_cross_section')
        sampler = self.ct.get_sampler_for_detector(
            **{**dict(
                wimp_mass=mass,
                cross_section=10 ** sigma,
                sampler_name='multinest_combined',
                detector_name=detector_name,
                prior=prior_name,
                halo_name='shm',
                detector_kwargs=None,
                halo_kwargs=None,
                sampler_kwargs=dict(nlive=100, tol=0.1, verbose=0, detector_name='test_combined'),
                fit_parameters=fit_params,
            ), **kwargs, }
        )
        sampler._fix_parameters()

        # Check that all the subconfigs are correctly set
        for c in sampler.sub_classes:
            assert c.log_cross_section == sigma, c.config
            assert c.log_mass == np.log10(mass)
            assert c.config['prior'] == dddm.get_priors(prior_name)
            assert c.benchmark_values is not None
        # sourcery skip use-named-expression
        benchmark_all_zero = not np.any(sampler.sub_classes[0].benchmark_values)
        if benchmark_all_zero:
            print('If everything is zero, I don\'t have to check if we converge')
            return

        # Do the parameter scan
        likelihood_scan = []
        # Hard-coding the range of parameters to scan for reproducibility
        sigma_scan = np.linspace(sigma - 0.2, sigma + 0.2, nbins)

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
