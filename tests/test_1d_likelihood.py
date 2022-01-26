"""
Test if the 1D likelihood returns a value that is close to the set benchmark value
"""
import dddm
import numpy as np
from hypothesis import given, settings, strategies
from tqdm import tqdm

known_detectors = [k for k in dddm.experiment_registry.keys() if ('migd' not in k and 'Combined' not in k)]
known_priors = 'Pato_2010 Evans_2019 migdal_wide low_mass migdal_extremely_wide low_mass_fixed'.split()


@settings(deadline=None, max_examples=10)
@given(strategies.floats(0.1, 50),
       strategies.integers(-47, -43),
       strategies.integers(0, len(known_detectors) - 1),
       strategies.integers(0, len(known_priors) - 1),
       )
def test_likelihood_converges(mass, sigma, detector_i, prior_i):
    """Test that a 1D likelihood scan actually returns the maximum at the set value"""
    detector_name = known_detectors[detector_i]
    prior_name = known_priors[prior_i]

    stats = dddm.CombinedInference(
        (detector_name,),
        'Combined',
    )

    stats.set_benchmark(mass, sigma)
    stats.set_prior(prior_name)
    stats._fix_parameters()

    # Check that all the subconfigs are correctly set
    for c in stats.sub_classes:
        assert c.log_cross_section == sigma
        assert c.log_mass == np.log10(mass)
        assert c.config['prior'] == dddm.get_priors(prior_name)
        assert c.benchmark_values is not None

    if benchmark_all_zero := not np.any(stats.sub_classes[0].benchmark_values):
        print('If everything is zero, I don\'t have to check if we converge')
        return

    # Do the parameter scan
    likelihood_scan = []
    # Hardcoding the range of parameters to scan for reproducibility
    sigma_scan = np.linspace(sigma - 0.2, sigma + 0.2, 30)

    for s in tqdm(sigma_scan, desc='Cross-section scan'):
        ll = stats.sub_classes[0]._log_probability_nested([np.log10(mass), s])
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
