"""
Statistical model giving likelihoods for detecting a spectrum given a
benchmark to compare it with.
"""

import os
from sys import platform
import numericalunits as nu
import numpy as np
from immutabledict import immutabledict
from dddm import utils
from dddm.priors import get_priors
from dddm.recoil_rates import halo, halo_shielded, spectrum, detector_spectrum
from scipy.special import loggamma
import typing as ty
import dddm

export, __all__ = dddm.exporter()

# Set a lower bound to the log-likelihood (this becomes a problem due to
# machine precision). Set to same number as multinest.
LL_LOW_BOUND = 1e-90


def get_prior_list():
    return ['log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density']


def get_param_list():
    return ['log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density']


@export
class StatModel:
    # Keep these fit parameters in this order
    _parameter_order = ('log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density', 'k')
    allow_multiple_detectors = False
    known_parameters = tuple(get_param_list())
    benchmark_values = None

    def __init__(
            self,
            wimp_mass: ty.Union[float, int],
            cross_section: ty.Union[float, int],
            spectrum_class: ty.Union[detector_spectrum.DetectorSpectrum,
                                     spectrum.GenSpectrum],
            prior: dict,
            tmp_folder: str,
            fit_parameters=('log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density', 'k'),

            detector_name=None,
            verbose=False,
            notes='default',
    ):
        """
        Statistical model used for Bayesian interference of detection in multiple experiments.
        :param detector_name: name of the detector (e.g. Xe)
        """
        if detector_name is None:
            detector_name = spectrum_class.detector.detector_name
        if not issubclass(
                spectrum_class.__class__, detector_spectrum.GenSpectrum
        ) and (
                not isinstance(spectrum_class, (list, tuple))
                or not issubclass(
            spectrum_class[0].__class__, detector_spectrum.GenSpectrum
        )
        ):
            raise ValueError(f'{spectrum_class}, {spectrum_class.__class__} is not the right class')

        self.spectrum_class = spectrum_class
        if isinstance(prior, str):
            prior = get_priors(prior)
        self.config = dict(
            detector=detector_name,
            notes=notes,
            start=utils.now(),
            prior=prior,

            _wimp_mass=wimp_mass,
            _cross_section=cross_section,
            # _spectrum_class=spectrum_class,
        )
        self.log = self.get_logger(tmp_folder, verbose)
        self.log.info(f"initialized for {detector_name} detector.")
        self.set_fit_parameters(fit_parameters)

    def __str__(self):
        return (f"StatModel::for {self.config['detector']} detector. "
                f"For info see the config file:\n{self.config}")

    def get_logger(self, tmp_folder, verbosity):
        if verbosity > 1:
            level = 'DEBUG'
        elif verbosity:
            level = 'INFO'
        else:
            level = 'WARNING'

        if 'win' not in platform:
            log_path = os.path.join(tmp_folder,
                                    f"log_{utils.now()}.log")
            self.config['logging'] = log_path
        else:
            log_path = None

        return utils.get_logger(self.__class__.__name__,
                                level,
                                path=log_path)

    def set_benchmark(self):
        """
        Set up the benchmark used in this statistical model. Likelihood
        of other models can be evaluated for this 'truth'
        """
        self.config['log_mass'] = np.log10(self.config['_wimp_mass'])
        self.config['log_cross_section'] = np.log10(self.config['_cross_section'])

    def set_fit_parameters(self, params):
        """Write the fit parameters to the config"""
        self.log.info(f'NestedSamplersetting fit parameters to {params}')
        if not isinstance(params, (list, tuple)):
            raise TypeError("Set the parameter names in a list of strings")
        known_params = self.known_parameters[:len(params)]
        if tuple(params) != tuple(known_params):
            err_message = f"The parameters are not input in the correct order. Please" \
                          f"insert {known_params} rather than {params}."
            raise NameError(err_message)
        self.config['fit_parameters'] = params

    def set_models(self):
        """
        Update the dm model with with the required settings from the prior
        """
        dm_model = self.spectrum_class.dm_model
        if self._earth_shielding:
            dm_model.log_mass = self.log_mass
            dm_model.log_cross_section = self.log_cross_section
            dm_model.location = self.spectrum_class.location
        dm_model.v_0 = self.v_0 * nu.km / nu.s
        dm_model.rho_dm = self.density * nu.GeV / nu.c0 ** 2 / nu.cm ** 3
        dm_model.v_esc = self.v_esc * nu.km / nu.s
        assert self.spectrum_class.dm_model.v_0 == self.v_0 * nu.km / nu.s

    def _fix_parameters(self, _do_evaluate_benchmark=True):
        """
        This is a very important function as it makes sure all the
        classes are setup in the right order
        :param _do_evaluate_benchmark: Evaluate the benchmark
        :return: None
        """
        if no_prior_has_been_set := self.config['prior'] is None:
            raise ValueError
        if no_wimp_mass_set := self.config.get('log_mass') is None:
            self.set_benchmark()
        elif self.config['log_cross_section'] is None:
            raise ValueError('Someone forgot to set sigma?!')

        # Very important that this comes AFTER the prior setting as we depend on it
        self.set_models()

        # Finally, set the benchmark
        if not _do_evaluate_benchmark:
            # Only do this for the combined experiments!
            self.log.info('Skipping evaluating the benchmark!')
        else:
            self.log.info('evaluate benchmark\tall ready to go!')
            self.eval_benchmark()
        # No more changes allowed
        self.config = immutabledict(self.config)

    def check_spectrum(self):
        """Lazy alias for eval_spectrum"""
        parameter_names = self._parameter_order[:2]
        parameter_values = [self.config['log_mass'], self.config['log_cross_section']]
        return self.eval_spectrum(parameter_values, parameter_names)

    def eval_benchmark(self):
        self.log.info('preparing for running, setting the benchmark')
        if self.bench_is_set:
            raise RuntimeError(self.bench_is_set)
        self.benchmark_values = self.check_spectrum()
        # Save a copy of the benchmark in the config file
        self.config['benchmark_values'] = list(self.benchmark_values)

    def total_log_prior(self, parameter_vals, parameter_names):
        """
        For each of the parameter names, read the prior

        :param parameter_vals: the values of the model/benchmark considered as the truth
        :param parameter_names: the names of the parameter_values
        :return:
        """
        # single parameter to fit
        if isinstance(parameter_names, str):
            lp = self.log_prior(parameter_vals, parameter_names)

        # check the input and compute the prior
        elif len(parameter_names) > 1:
            if len(parameter_vals) != len(parameter_names):
                raise ValueError(
                    f"provide enough names {parameter_names}) "
                    f"for parameters (len{len(parameter_vals)})")
            lp = np.sum([self.log_prior(*_x) for _x in
                         zip(parameter_vals, parameter_names)])
        else:
            raise TypeError(
                f"incorrect format provided. Theta should be array-like for "
                f"single value of parameter_names or Theta should be "
                f"matrix-like for array-like parameter_names. Theta, "
                f"parameter_names (provided) = "
                f"{parameter_vals, parameter_names}")
        return lp

    def log_probability(self, parameter_vals, parameter_names):
        """

        :param parameter_vals: the values of the model/benchmark considered as the truth
        :param parameter_names: the names of the parameter_values
        :return:
        """
        self.log.debug('evaluate log probability')

        lp = self.total_log_prior(parameter_vals, parameter_names)

        if not np.isfinite(lp):
            return -LL_LOW_BOUND

        self.log.info('loading rate for given parameters')
        evaluated_rate = self.eval_spectrum(parameter_vals, parameter_names)

        # Compute the likelihood
        ll = log_likelihood(self.benchmark_values, evaluated_rate)
        if np.isnan(lp + ll):
            raise ValueError(
                f"Returned NaN from likelihood. lp = {lp}, ll = {ll}")
        self.log.info('likelihood evaluated')
        return lp + ll

    def log_prior(self, value, variable_name):
        """
        Compute the prior of variable_name for a given value
        :param value: value of variable name
        :param variable_name: name of the 'value'. This name should be in the
        config of the class under the priors with a similar content as the
        priors as specified in the get_prior function.
        :return: prior of value
        """
        # For each of the priors read from the config file how the prior looks
        # like. Get the boundaries (and mean (m) and width (s) for gaussian
        # distributions).
        self.log.info(f'evaluating priors for {variable_name}')
        if self.config['prior'][variable_name]['prior_type'] == 'flat':
            a, b = self.config['prior'][variable_name]['param']
            return log_flat(a, b, value)
        elif self.config['prior'][variable_name]['prior_type'] == 'gauss':
            a, b = self.config['prior'][variable_name]['range']
            m, s = self.config['prior'][variable_name]['param']
            return log_gauss(a, b, m, s, value)
        else:
            raise TypeError(
                f"unknown prior type '{self.config['prior'][variable_name]['prior_type']}',"
                f" choose either gauss or flat")

    @property
    def _earth_shielding(self):
        return str(self.spectrum_class.dm_model) == 'shielded_shm'

    def eval_spectrum(self,
                      values: ty.Union[list, tuple, np.ndarray],
                      parameter_names: ty.Union[ty.List[str], ty.Tuple[str]]
                      ):
        """
        For given values and parameter names, return the spectrum one would have
        with these parameters. The values and parameter names should be array
        like objects of the same length. Usually, one fits either two
        ('log_mass', 'log_cross_section') or five parameters ('log_mass',
        'log_cross_section', 'v_0', 'v_esc', 'density').
        :param values: array like object of
        :param parameter_names: names of parameters
        :return: a spectrum as specified by the parameter_names
        """
        self.log.debug(f'evaluate spectrum for {len(values)} parameters')
        if len(values) != len(parameter_names):
            raise ValueError(f'{len(values)} != len({parameter_names})')
        if isinstance(parameter_names, str):
            raise NotImplementedError(f"Got single param {parameter_names}?")
        if len(parameter_names) not in [2, 5]:
            raise NotImplementedError('Use either 2 or 5 parameters to fit')
        checked_values = check_shape(values)
        log_mass = checked_values[0]
        log_cross_section = checked_values[1]

        # Update dark matter parameters in place
        dm_class = self.spectrum_class.dm_model
        if self._earth_shielding and len(parameter_names) >= 2:
            dm_class.log_cross_section = log_cross_section
            dm_class.log_mass = log_mass
            assert self.spectrum_class.dm_model.log_mass == log_mass

        elif len(parameter_names) == 5:
            if tuple(parameter_names) != tuple(self._parameter_order[:len(parameter_names)]):
                raise NameError(
                    f"The parameters are not in correct order. Please insert"
                    f"{self._parameter_order[:len(parameter_names)]} rather than "
                    f"{parameter_names}.")

            v_0 = checked_values[2] * nu.km / nu.s
            v_esc = checked_values[3] * nu.km / nu.s
            rho_dm = checked_values[4] * nu.GeV / nu.c0 ** 2 / nu.cm ** 3
            dm_class.v_0 = v_0
            dm_class.v_esc = v_esc
            dm_class.rho_dm = rho_dm
            assert self.spectrum_class.dm_model.rho_dm == rho_dm

        spectrum = self.spectrum_class.get_counts(
            wimp_mass=10. ** log_mass,
            cross_section=10. ** log_cross_section,
            poisson=False)

        self.log.debug('returning results')
        return spectrum

    def read_priors_mean(self, prior_name) -> ty.Union[int, float]:
        self.log.debug(f'reading {prior_name}')
        if self.config['prior'] is None:
            raise ValueError('Prior not set!')
        return self.config['prior'][prior_name]['mean']

    @property
    def v_0(self) -> ty.Union[int, float]:
        return self.read_priors_mean('v_0')

    @property
    def v_esc(self) -> ty.Union[int, float]:
        return self.read_priors_mean('v_esc')

    @property
    def density(self) -> ty.Union[int, float]:
        return self.read_priors_mean('density')

    @property
    def log_mass(self):
        return self.config['log_mass']

    @property
    def log_cross_section(self):
        return self.config['log_cross_section']

    @property
    def bench_is_set(self):
        return self.benchmark_values is not None


def log_likelihood_function(nb, nr):
    """
    return the ln(likelihood) for Nb expected events and Nr observed events

    #     :param nb: expected events
    #     :param nr: observed events
    #     :return: ln(likelihood)
    """
    if nr == 0:
        # For i~0, machine precision sets nr to 0. However, this becomes a
        # little problematic since the Poisson log likelihood for 0 is not
        # defined. Hence we cap it off by setting nr to 10^-100.
        nr = LL_LOW_BOUND
    return np.log(nr) * nb - loggamma(nb + 1) - nr


def log_likelihood(model, y):
    """
    :param model: pandas dataframe containing the number of counts in bin i
    :param y: the number of counts in bin i
    :return: sum of the log-likelihoods of the bins
    """

    if len(y) != len(model):
        raise ValueError(f"Data and model should be of same dimensions (now "
                         f"{len(y)} and {len(model)})")

    res = 0
    # pylint: disable=consider-using-enumerate
    for i in range(len(y)):
        Nr = y[i]
        Nb = model[i]
        res_bin = log_likelihood_function(Nb, Nr)
        if np.isnan(res_bin):
            raise ValueError(
                f"Returned NaN in bin {i}. Below follows data dump.\n"
                f"log_likelihood: {log_likelihood_function(Nb, Nr)}\n"
                f"i = {i}, Nb, Nr =" + " %.2g %.2g \n" % (Nb, Nr) + "")
        if not np.isfinite(res_bin):
            return -np.inf
        res += res_bin
    return res


def check_shape(xs):
    """
    :param xs: values
    :return: flat array of values
    """
    if len(xs) <= 0:
        raise TypeError(
            f"Provided incorrect type of {xs}. Takes either np.array or list")
    if not isinstance(xs, np.ndarray):
        xs = np.array(xs)
    for i, x in enumerate(xs):
        if np.shape(x) == (1,):
            xs[i] = x[0]
    return xs


def log_flat(a, b, x):
    """
    Return a flat prior as function of x in log space
    :param a: lower bound
    :param b: upper bound
    :param x: value
    :return: 0 for x in bound, -np.inf otherwise
    """
    try:
        return 0 if a < x < b else -np.inf
    except ValueError:
        result = np.zeros(len(x))
        mask = (x > a) & (x < b)
        result[~mask] = -np.inf
        return result


def log_gauss(a, b, mu, sigma, x):
    """
    Return a gaussian prior as function of x in log space
    :param a: lower bound
    :param b: upper bound
    :param mu: mean of gauss
    :param sigma: std of gauss
    :param x: value to evaluate
    :return: log prior of x evaluated for gaussian (given by mu and sigma) if in
    between the bounds
    """
    try:
        # for single values of x
        if a < x < b:
            return -0.5 * np.sum(
                (x - mu) ** 2 / (sigma ** 2) + np.log(sigma ** 2))
        return -np.inf
    except ValueError:
        # for array like objects return as follows
        result = np.zeros(len(x))
        mask = (x > a) & (x < b)
        result[~mask] = -np.inf
        result[mask] = -0.5 * np.sum(
            (x - mu) ** 2 / (sigma ** 2) + np.log(sigma ** 2))
        return result
