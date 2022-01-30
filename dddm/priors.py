from immutabledict import immutabledict
import dddm
import numpy as np

export, __all__ = dddm.exporter()


@export
def get_priors(priors_from="Evans_2019"):
    """
    :return: dictionary of priors, type and values
    """
    if priors_from == "Pato_2010":
        priors = {'log_mass': {'range': [0.1, 3], 'prior_type': 'flat'},
                  'log_cross_section': {'range': [-46, -42], 'prior_type': 'flat'},
                  'density': {'range': [0.001, 0.9], 'prior_type': 'gauss', 'mean': 0.4,
                              'std': 0.1},
                  'v_0': {'range': [80, 380], 'prior_type': 'gauss', 'mean': 230, 'std': 30},
                  'v_esc': {'range': [379, 709], 'prior_type': 'gauss', 'mean': 544, 'std': 33},
                  'k': {'range': [0.5, 3.5], 'prior_type': 'flat'}}
    elif priors_from == "Evans_2019":
        # https://arxiv.org/abs/1901.02016
        priors = {'log_mass': {'range': [0.1, 3], 'prior_type': 'flat'},
                  'log_cross_section': {'range': [-46, -42], 'prior_type': 'flat'},
                  'density': {'range': [0.001, 0.9], 'prior_type': 'gauss', 'mean': 0.55,
                              'std': 0.17},
                  'v_0': {'range': [80, 380], 'prior_type': 'gauss', 'mean': 233, 'std': 3},
                  'v_esc': {'range': [379, 709], 'prior_type': 'gauss', 'mean': 528, 'std': 24.5}}
    elif priors_from == "migdal_wide":
        priors = {'log_mass': {'range': [-1.5, 1.5], 'prior_type': 'flat'},
                  'log_cross_section': {'range': [-48, -37], 'prior_type': 'flat'},
                  # see Evans_2019_constraint
                  'density': {'range': [0.001, 0.9], 'prior_type': 'gauss', 'mean': 0.55,
                              'std': 0.17},
                  'v_0': {'range': [80, 380], 'prior_type': 'gauss', 'mean': 233, 'std': 20},
                  'v_esc': {'range': [379, 709], 'prior_type': 'gauss', 'mean': 528, 'std': 24.5},
                  'k': {'range': [0.5, 3.5], 'prior_type': 'flat'}}
    elif priors_from == "low_mass":
        priors = {'log_mass': {'range': [-1.5, 1.5], 'prior_type': 'flat'},
                  'log_cross_section': {'range': [-48, -37], 'prior_type': 'flat'},
                  # see Evans_2019_constraint
                  'density': {'range': [0.0001, 1], 'prior_type': 'gauss', 'mean': 0.55,
                              'std': 0.17},
                  'v_0': {'range': [133, 333], 'prior_type': 'gauss', 'mean': 233, 'std': 20},
                  'v_esc': {'range': [405.5, 650.5], 'prior_type': 'gauss', 'mean': 528,
                            'std': 24.5}}
    elif priors_from == "low_mass_fixed":
        priors = {'log_mass': {'range': [-2, 2], 'prior_type': 'flat'},
                  'log_cross_section': {'range': [-53, -27], 'prior_type': 'flat'},
                  # see Evans_2019_constraint
                  'density': {'range': [0.0001, 1], 'prior_type': 'gauss', 'mean': 0.55,
                              'std': 0.17},
                  'v_0': {'range': [133, 333], 'prior_type': 'gauss', 'mean': 233, 'std': 20},
                  'v_esc': {'range': [405.5, 650.5], 'prior_type': 'gauss', 'mean': 528,
                            'std': 24.5}}
    elif priors_from == "migdal_extremely_wide":
        priors = {'log_mass': {'range': [-2, 3], 'prior_type': 'flat'},
                  'log_cross_section': {'range': [-50, -30], 'prior_type': 'flat'},
                  'density': {'range': [0.001, 0.9], 'prior_type': 'gauss', 'mean': 0.55,
                              'std': 0.5},
                  'v_0': {'range': [80, 380], 'prior_type': 'gauss', 'mean': 233, 'std': 90},
                  'v_esc': {'range': [379, 709], 'prior_type': 'gauss', 'mean': 528, 'std': 99},
                  'k': {'range': [0.5, 3.5], 'prior_type': 'flat'}
                  }
    else:
        raise NotImplementedError(
            f"Taking priors from {priors_from} is not implemented")

    for key in priors.keys():
        param = priors[key]
        if param['prior_type'] == 'flat':
            param['param'] = param['range']
            param['dist'] = flat_prior_distribution
        elif param['prior_type'] == 'gauss':
            param['param'] = param['mean'], param['std']
            param['dist'] = gauss_prior_distribution
    return immutabledict(priors)


def flat_prior_distribution(_range):
    return np.random.uniform(_range[0], _range[1])


def gauss_prior_distribution(_param):
    mu, sigma = _param
    return np.random.normal(mu, sigma)
