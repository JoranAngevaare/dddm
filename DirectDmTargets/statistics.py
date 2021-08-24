"""Statistical model giving likelihoods for detecting a spectrum given a benchmark to compare it with."""

import logging
import os
import time
from sys import platform
import numericalunits as nu
import numpy as np
import pandas as pd
from DirectDmTargets import context, detector, halo, utils
from scipy.special import loggamma
import typing as ty

# Set a lower bound to the log-likelihood (this becomes a problem due to
# machine precision). Set to same number as multinest.
LL_LOW_BOUND = 1e-90


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
    elif priors_from == "migdal_extremely_wide":
        priors = {'log_mass': {'range': [-2, 3], 'prior_type': 'flat'},
                  'log_cross_section': {'range': [-50, -30], 'prior_type': 'flat'},
                  'density': {'range': [0.001, 0.9], 'prior_type': 'gauss', 'mean': 0.55,
                              'std': 0.5},
                  'v_0': {'range': [80, 380], 'prior_type': 'gauss', 'mean': 233, 'std': 90},
                  'v_esc': {'range': [379, 709], 'prior_type': 'gauss', 'mean': 528, 'std': 99},
                  'k': {'range': [0.5, 3.5], 'prior_type': 'flat'}}
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
    return priors


def get_prior_list():
    return ['mw', 'sigma', 'v_0', 'v_esc', 'density']


def get_param_list():
    return ['log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density']


class StatModel:
    # Keep these fit parameters in this order
    _parameter_order = ('log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density', 'k')
    benchmark_values = None

    def __init__(
            self,
            detector_name,
            verbose=False,
            detector_config=None,
            do_init=True):
        """
        Statistical model used for Bayesian interference of detection in multiple experiments.
        :param detector_name: name of the detector (e.g. Xe)
        """
        if detector_name not in detector.experiment and detector_config is None:
            raise ValueError('Please provide detector that is '
                             'preconfigured or provide new one with detector_dict')
        if detector_config is None:
            detector_config = detector.experiment[detector_name]

        self.config = dict(detector=detector_name,
                           detector_config=detector_config,
                           # poisson=False,
                           n_energy_bins=detector_config.get('n_energy_bins', 10),
                           earth_shielding=False,
                           save_intermediate=False,
                           E_max=detector_config.get('E_max', 100)
                           )

        self.log = self.get_logger(verbose)
        self.set_prior("Pato_2010")
        self.log.info(
            f"initialized for {detector_name} detector. See  print(stat_model) for default settings")
        if do_init:
            self.set_default()

    @property
    def bench_is_set(self):
        return self.benchmark_values is None

    def get_logger(self, verbosity):
        if verbosity > 1:
            level = 'DEBUG'
        elif verbosity:
            level = 'INFO'
        else:
            level = 'WARNING'

        log = utils.get_logger(self.__class__.__name__, level)

        if 'win' not in platform:
            self.config['logging'] = os.path.join(
                context.context['tmp_folder'],
                f"log_{utils.now()}.log")
            print(f'StatModel::\tSave log to {self.config["logging"]}')
            log.addHandler(logging.StreamHandler())

        return log

    def __str__(self):
        return f"StatModel::for {self.config['detector']} detector. For info see the config file:\n{self.config}"

    def read_priors_mean(self, prior_name) -> ty.Union[int, float]:
        self.log.debug(f'reading {prior_name}')
        return self.config['prior'][prior_name]['mean']
    
    @property
    def v_0(self) -> ty.Union[int, float]:
        return self.read_priors_mean('v_0')

    @property
    def v_esc(self) -> ty.Union[int, float]:
        return self.read_priors_mean('v_0')

    @property
    def density(self) -> ty.Union[int, float]:
        return self.read_priors_mean('v_0')

    @property
    def log_mass(self):
        return self.config['mw']

    @property
    def log_cross_section(self):
        return self.config['sigma']

    def insert_prior_manually(self, input_priors):
        self.log.warning(
            f'Inserting {input_priors} as priors. For the right format check '
            f'DirectDmTargets/statistics.py. I assume your format is right.')
        self.config['prior'] = input_priors

    def set_prior(self, priors_from):
        self.log.info(f'set_prior')
        self.config['prior'] = get_priors(priors_from)

    def set_benchmark(self, mw=50, sigma=-45):
        """
        Set up the benchmark used in this statistical model. Likelihood of
        other models can be evaluated for this 'truth'

        :param mw: mass of benchmark wimp in GeV. log10(mass) will be saved to config
        :param sigma: cross-section of wimp in cm^2. log10(sigma) will be saved to config
        """
        self.log.info(f'taking log10 of mass of {mw}')
        self.config['mw'] = np.log10(mw)
        self.config['sigma'] = sigma
        if not ((mw == 50) and (sigma == -45)):
            self.log.warning(f'taking log10 of mass of {mw}')
            self.eval_benchmark()

    def set_models(self, halo_model='default', spec='default'):
        """
        Update the config with the required settings
        :param halo_model: The halo model used
        :param spec: class used to generate the response of the spectrum in the
        detector
        """

        if self.config['earth_shielding']:
            self.log.info(
                f'StatModel::\tsetting model to VERNE model. Using:'
                f"\nlog_mass={self.config['mw']},"
                f"\nlog_cross_section={self.config['sigma']},"
                f"\nlocation={self.config['detector_config']['location']},"
                f'\nv_0={self.config["v_0"]} * nu.km / nu.s,'
                f'\nv_esc={self.config["v_esc"]} * nu.km / nu.s,'
                f'\nrho_dm={self.config["density"]} * nu.GeV / nu.c0 ** 2 / nu.cm ** 3')
            model = halo.VerneSHM(
                log_mass=self.config['mw'],
                log_cross_section=self.config['sigma'],
                location=self.config['detector_config']['location'],
                v_0=self.v_0 * nu.km / nu.s,
                v_esc=self.v_esc * nu.km / nu.s,
                rho_dm=self.density * nu.GeV / nu.c0 ** 2 / nu.cm ** 3)

            self.config['halo_model'] = halo_model if halo_model != 'default' else model
            self.log.info(
                f'StatModel::\tmodel is set to: {self.config["halo_model"]}')
        else:
            self.log.info(
                f'StatModel::\tSetting model to SHM. Using:'
                f'\nv_0={self.config["v_0"]} * nu.km / nu.s,'
                f'\nv_esc={self.config["v_esc"]} * nu.km / nu.s,'
                f'\nrho_dm={self.config["density"]} * nu.GeV / nu.c0 ** 2 / nu.cm ** 3')
            self.config['halo_model'] = halo_model if halo_model != 'default' else halo.SHM(
                v_0=self.v_0 * nu.km / nu.s,
                v_esc=self.v_esc * nu.km / nu.s,
                rho_dm=self.density * nu.GeV / nu.c0 ** 2 / nu.cm ** 3)
        if self.config['earth_shielding']:
            self.config['save_intermediate'] = True
        else:
            self.config['save_intermediate'] = False
        self.log.info(
            f'StatModel::\tsave_intermediate:\n\t\t{self.config["save_intermediate"]}')

        self.config['spectrum_class'] = spec if spec != 'default' else detector.DetectorSpectrum

        if halo_model != 'default' or spec != 'default':
            self.log.warning(f"StatModel::\tre-evaluate benchmark")
            self.eval_benchmark()

    def set_det_params(self):
        self.log.info(f'StatModel::\treading detector parameters')
        # This is a legacy statement
        self.config['det_params'] = self.config['detector_config']

    def set_fit_parameters(self, params):
        self.log.info(f'NestedSamplerStatModel::\tsetting fit'
                      f' parameters to {params}')
        if not isinstance(params, (list, tuple)):
            raise TypeError("Set the parameter names in a list of strings")
        for param in params:
            if param not in self.known_parameters:
                err_message = f"{param} does not match any of the known parameters try " \
                              f"any of {self.known_parameters}"
                raise NotImplementedError(err_message)
        if params != self.known_parameters[:len(params)]:
            err_message = f"The parameters are not input in the correct order. Please" \
                          f" insert {self.known_parameters[:len(params)]} rather than {params}."
            raise NameError(err_message)
        self.config['fit_parameters'] = params

    def set_default(self):
        self.log.info(f'StatModel::\tinitializing')
        self.set_benchmark()
        self.log.info(f'StatModel::\tset_benchmark\tdone')
        self.set_models()
        self.log.info(f'StatModel::\tset_models\tdone')
        self.set_det_params()
        self.log.info(f'StatModel::\tset_det_params\tdone')
        self.eval_benchmark()
        self.log.info(
            f'StatModel::\tevaluate benchmark\tdone\n\tall ready to go!')

    def find_intermediate_result(
            self,
            nbin=None,
            model=None,
            mw=None,
            sigma=None,
            rho=None,
            v_0=None,
            v_esc=None,
            poisson=None,
            det_conf=None):
        """
        :return: exists (bool), the name, and if exists the spectrum
        """
        self.log.warning(
            'Saving intermediate results. Computational gain may be limited')
        if 'DetectorSpectrum' not in str(self.config['spectrum_class']):
            raise ValueError("Input detector spectrum")
        # Name the file according to the main parameters. Note that for each of
        # the main parameters
        file_name = os.path.join(
            context.context['spectra_files'],
            'nbin-%i' %
            (self.config['n_energy_bins'] if nbin is None else nbin),
            'model-%s' %
            (str(
                self.config['halo_model']) if model is None else str(model)),
            'mw-%.2f' %
            (10. ** self.config['mw'] if mw is None else 10. ** mw),
            'log_s-%.2f' %
            (self.config['sigma'] if sigma is None else sigma),
            'rho-%.2f' %
            (self.density if rho is None else rho),
            'v_0-%.1f' %
            (self.v_0 if v_0 is None else v_0),
            'v_esc-%i' %
            (self.v_esc if v_esc is None else v_esc),
            'poisson_%i' %
            (int(
                self.config['poisson'] if poisson is None else poisson)),
            'spectrum')
        print(file_name)
        # Add all other parameters that are in the detector config
        if det_conf is None:
            det_conf = self.config['detector_config']
        for key in det_conf.keys():
            if callable(self.config['detector_config'][key]):
                continue
            file_name = file_name + '_' + \
                str(self.config['detector_config'][key])
        file_name = file_name.replace(' ', '_')
        file_name = file_name + '.csv'
        data_at_path, file_path = utils.add_pid_to_csv_filename(file_name)

        # There have been some issues with mixed results for these two
        # densities. Remove those files.
        if rho == 0.55 or rho == 0.4:
            if data_at_path:
                write_time = os.path.getmtime(file_path)
                feb17_2020 = 1581932824.5842493
                if write_time < feb17_2020:
                    self.log.error(
                        f'StatModel::\tWARNING REMOVING {file_path}')
                    os.remove(file_path)
                    data_at_path, file_path = utils.add_pid_to_csv_filename(
                        file_name)
                    self.log.warning(
                        f'StatModel::\tRe-evatulate, now we have {file_path}. Is there data: {data_at_path}')

        if data_at_path:
            try:
                binned_spectrum = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                self.log.error(
                    "StatModel::\tdataframe empty, have to remake the data!")
                os.remove(file_path)
                binned_spectrum = None
                data_at_path = False
        else:
            self.log.warning(
                "StatModel::\tNo data at path. Will have to make it.")
            binned_spectrum = None
            utils.check_folder_for_file(file_path)

        self.log.info(f"StatModel::\tdata at {file_path} = {data_at_path}")
        return data_at_path, file_path, binned_spectrum

    def save_intermediate_result(self, binned_spectrum, spectrum_file):
        """
        Save evaluated binned spectrum according to naming convention
        :param binned_spectrum: evaluated spectrum
        :param spectrum_file: name where to save the evaluated spectrum
        :return:
        """
        self.log.info(f"StatModel::\tsaving spectrum at {spectrum_file}")
        if os.path.exists(spectrum_file):
            # Do not try overwriting existing files.
            return
        try:
            # rename the file to also reflect the hosts name such that we don't make two
            # copies at the same place with from two different hosts
            if context.host not in spectrum_file:
                spectrum_file = spectrum_file.replace(
                    '.csv', context.host + '.csv')
            try:
                binned_spectrum.to_csv(spectrum_file, index=False)
            except Exception as e:
                self.log.error(
                    f"Error while saving {spectrum_file}. Sleep 5 sec and "
                    f"retry. The error was:\n{e} ignoring that for now.")
                time.sleep(5)
                if os.path.exists(spectrum_file):
                    os.remove(spectrum_file)
                    time.sleep(5)
                    binned_spectrum.to_csv(spectrum_file, index=False)
        except PermissionError as e:
            self.log.warning(f'{e} occurred, ignoring that for now.')
            # While computing the spectrum another instance has saved a file
            # with the same name

    def check_spectrum(self):
        parameter_names = self._parameter_order[:2]
        parameter_values = [self.config['mw'], self.config['sigma'], ]
        return self.eval_spectrum(parameter_values, parameter_names)

        self.log.info(
            f"StatModel::\tevaluating\n\t\t{self.config['spectrum_class']}"
            f"\n\tfor mw = {10. ** self.config['mw']}, "
            f"\n\tsig = {10. ** self.config['sigma']}, "
            f"\n\thalo model = \n\t\t{self.config['halo_model']} and "
            f"\n\tdetector = \n\t\t{self.config['detector_config']}")
        if self.config['save_intermediate']:
            self.log.info(f"StatModel::\tlooking for intermediate results")
            interm_exists, interm_file, interm_spec = self.find_intermediate_result()
            if interm_exists:
                return interm_spec

        # Initialize the spectrum class if:
        # A) we are not saving intermediate results
        # B) we haven't yet computed the desired intermediate spectrum
        spectrum = self.config['spectrum_class'](
            10. ** self.config['mw'],
            10. ** self.config['sigma'],
            self.config['halo_model'],
            self.config['detector_config'])
        spectrum = self.config_to_spectrum(spectrum)

        for e_min_max in 'E_min E_max'.split():
            if e_min_max in self.config:
                self.log.info(f'set {e_min_max} to {self.config[e_min_max]}')
                spectrum.set_config(dict(e_min_max=self.config[e_min_max]))

        binned_spectrum = spectrum.get_data(
            poisson=self.config['poisson'] if poisson is None else poisson
        )
        self.log.warning(f'Check spectrum, {spectrum.E_max} keV E_max')

        if self.config['save_intermediate']:
            self.save_intermediate_result(binned_spectrum, interm_file)
        return binned_spectrum

    def eval_benchmark(self):
        self.log.info(
            f'StatModel::\tpreparing for running, setting the benchmark')
        df = self.check_spectrum()
        print(df)
        time.sleep(10)
        self.benchmark_values = df['counts']
        self.bench_is_set = True
        # Save a copy of the benchmark in the config file
        self.config['benchmark_values'] = list(self.benchmark_values)

    def check_bench_set(self):
        if not self.bench_is_set:
            self.log.info(f'StatModel::\tbenchmark not set->doing so now')
            self.eval_benchmark()

    def log_probability(self, parameter_vals, parameter_names):
        """

        :param parameter_vals: the values of the model/benchmark considered as the truth
        :param parameter_names: the names of the parameter_values
        :return:
        """
        self.log.info(
            f'StatModel::\tEngines running full! Lets get some probabilities')
        self.check_bench_set()

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
        if not np.isfinite(lp):
            return -np.inf
        self.log.info(f'StatModel::\tloading rate for given parameters')
        evaluated_rate = self.eval_spectrum(
            parameter_vals, parameter_names)['counts']

        # Compute the likelihood
        ll = log_likelihood(self.benchmark_values, evaluated_rate)
        if np.isnan(lp + ll):
            raise ValueError(
                f"Returned NaN from likelihood. lp = {lp}, ll = {ll}")
        self.log.info(f'StatModel::\tlikelihood evaluated')
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
        self.log.info(f'StatModel::\tevaluating priors for {variable_name}')
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
            raise ValueError(f'trying to fit {len(values)} parameters but '
                             f'{parameter_names} are given.')

        if isinstance(parameter_names, str):
            raise NotImplementedError(
                f"Trying to fit a single parameter ({parameter_names}), such a "
                f"feature is not implemented.")

        checked_values = check_shape(values)

        if self.config['save_intermediate']:
            self.log.info(f"load results from intermediate file")

            spec_class = self.config['halo_model']

            if self.config['earth_shielding']:
                if str(spec_class) != str(halo.VerneSHM()):
                    raise ValueError('Not running with shielding!')

            interm_exists, interm_file, interm_spec = self.find_intermediate_result(
                nbin=self.config['n_energy_bins'],
                model=str(spec_class),
                mw=checked_values[0],
                sigma=checked_values[1],
                v_0=checked_values[2] if len(checked_values) > 2 else self.v_0,
                v_esc=checked_values[3] if len(checked_values) > 3 else self.v_esc,
                rho=checked_values[4] if len(checked_values) > 4 else self.density,
                poisson=False,
                det_conf=self.config['detector_config']
            )
            if interm_exists:
                return interm_spec
            self.log.info(f"No file found, save intermediate result later")

        if len(parameter_names) == 2:
            if parameter_names[0] == 'log_mass' and parameter_names[1] == 'log_cross_section':
                # This is the right order
                pass
            else:
                raise NotImplementedError(
                    f"Trying to fit two parameters ({parameter_names}), this is not implemented.")
            self.log.debug(
                f"evaluating {self.config['spectrum_class']} for mw = {10. ** checked_values[0]}, "
                f"sig = {10. ** checked_values[1]}, halo model = {self.config['halo_model']} and "
                f"detector = {self.config['detector_config']}")
            if self.config['earth_shielding']:
                self.log.debug(f"Setting spectrum to Verne in likelihood code")
                fit_shm = halo.VerneSHM(
                    log_mass=checked_values[0],  # self.config['mw'],
                    log_cross_section=checked_values[1],  # self.config['sigma'],
                    location=self.config['detector_config']['location'],
                    v_0=self.v_0 * nu.km / nu.s,
                    v_esc=self.v_esc * nu.km / nu.s,
                    rho_dm=self.density * nu.GeV / nu.c0 ** 2 / nu.cm ** 3)
            else:
                fit_shm = self.config['halo_model']

        elif len(parameter_names) == 5:
            if parameter_names != self._parameter_order[:len(parameter_names)]:
                raise NameError(
                    f"The parameters are not in correct order. Please insert"
                    f"{self._parameter_order[:len(parameter_names)]} rather than "
                    f"{parameter_names}.")

            if self.config['earth_shielding']:
                self.log.debug(
                    f"Setting spectrum to Verne in likelihood code")
                fit_shm = halo.VerneSHM(
                    log_mass=checked_values[0],  # 'mw
                    log_cross_section=checked_values[1],  # 'sigma'
                    location=self.config['detector_config']['location'],
                    v_0=checked_values[2] * nu.km / nu.s,  # 'v_0'
                    v_esc=checked_values[3] * nu.km / nu.s,  # 'v_esc'
                    rho_dm=checked_values[
                        4] * nu.GeV / nu.c0 ** 2 / nu.cm ** 3)  # 'density'
            else:
                self.log.debug(f"Using SHM in likelihood code")
                fit_shm = halo.SHM(
                    v_0=checked_values[2] * nu.km / nu.s,
                    v_esc=checked_values[3] * nu.km / nu.s,
                    rho_dm=checked_values[4] * nu.GeV / nu.c0 ** 2 / nu.cm ** 3)

        elif len(parameter_names) > 2 and not len(parameter_names) == 5:
            raise NotImplementedError(
                f"Not so quickly cowboy, before you code fitting "
                f"{len(parameter_names)} parameters or more, first code it! "
                f"You are now trying to fit {parameter_names}. Make sure that "
                f"you are not using forcing a string in this part of the code)")
        else:
            raise NotImplementedError(
                f"Something strange went wrong here. Trying to fit for the"
                f"parameter_names = {parameter_names}")

        spectrum = self.config['spectrum_class'](
            10. ** checked_values[0],
            10. ** checked_values[1],
            fit_shm,
            self.config['detector_config'])

        spectrum = self.config_to_spectrum(spectrum)
        binned_spectrum = spectrum.get_data(poisson=False)
        self.log.debug(f"we have results!")

        if np.any(binned_spectrum['counts'] < 0):
            error_message = (
                f"Finding negative rates. Presumably v_esc is too small. "
                f"Or one or more priors are not constrained correctly. "
                f"dump of parameters:\n" f"{parameter_names} = {values}."
            )
            if 'migd' in self.config['detector']:
                binned_spectrum = spectrum.set_negative_to_zero(binned_spectrum)
                self.log.error(error_message)
            else:
                raise ValueError(error_message)
        if self.config['save_intermediate']:
            self.log.debug(f"saving results")
            self.save_intermediate_result(binned_spectrum, interm_file)
        self.log.debug(f"returning results")
        return binned_spectrum

    def config_to_spectrum(self, spectrum,
                           copy_fields=('E_min',
                                        'E_max',
                                        'n_energy_bins',
                                        )
                           ):
        """Set the config of the spectrum to have the same value as we do"""
        for to_copy in copy_fields:
            if to_copy in self.config:
                self.log.info(f'set {to_copy} to {self.config[to_copy]}')
                spectrum.set_config({to_copy: self.config[to_copy]})
        return spectrum


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


def flat_prior_distribution(_range):
    return np.random.uniform(_range[0], _range[1])


def gauss_prior_distribution(_param):
    mu, sigma = _param
    return np.random.normal(mu, sigma)


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
        if a < x < b:
            return 0
        return -np.inf
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
