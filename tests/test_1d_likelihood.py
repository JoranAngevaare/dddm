"""
Test if the 1D likelihood returns a value that is close to the set benchmark value
"""
import DirectDmTargets as dddm
import numpy as np
from hypothesis import given, settings, strategies
from tqdm import tqdm

known_detectors = [k for k in dddm.experiment.keys() if ('migd' not in k and 'Combined' not in k)]
known_priors = 'Pato_2010 Evans_2019 migdal_wide'.split()


@settings(deadline=None, max_examples=10)
@given(strategies.floats(0.1, 50),
       strategies.integers(-47, -43),
       strategies.integers(0, len(known_detectors)-1),
       strategies.integers(0, len(known_priors)-1),
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

    # Do the parameter scan
    likelihoods = []
    sigmas = np.linspace(sigma - 0.2, sigma + 0.2, 30)
    m = mass
    if not np.any(stats.sub_classes[0].benchmark_values):
        print('If everything is zero, I don\'t have to check if we converge')
        return
    for s in tqdm(sigmas, desc='Cross-section scan'):
        ll = stats.sub_classes[0]._log_probability_nested([np.log10(m), s])
        likelihoods.append(ll)

    max_ll = np.argmax(likelihoods)
    if not np.isclose(sigmas[max_ll],
                      sigma,
                      # if the binning is course, the result will also be. Allow some tolerance
                      atol=sigmas[1] - sigmas[0]
                      ):
        print(sigmas)
        print(likelihoods)
        if not np.all(np.isclose(likelihoods[0],
                                 likelihoods)

                      ):
            raise ValueError(f'{detector_name}-{prior_name}\tm:{mass}\ts:{sigma} failed')