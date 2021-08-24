"""
Do a likelihood fit. The class NestedSamplerStatModel is used for fitting
applying the bayesian algorithm nestle/multinest
"""

from __future__ import absolute_import, unicode_literals
import datetime
import json
import logging
import os
import shutil
import tempfile
from warnings import warn
import corner
import matplotlib.pyplot as plt
import numpy as np
from DirectDmTargets import context, detector, statistics, utils
from scipy import special as spsp

log = logging.getLogger()


class NestedSamplerStatModel(statistics.StatModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config.update(
            {'tol': 0.1,  # Tolerance for sampling
             'nlive': 1024,  # number of live points
             'sampler': 'multinest',
             })
        self.log_dict = {
            'did_run': False,
            'saved_in': None,
            'tmp_dir': None,
            'garbage_bin': []}

        self.result = False
        self.set_fit_parameters(('log_mass', 'log_cross_section'))

    def check_did_run(self):
        if not self.log_dict['did_run']:
            self.log.info(f'did not run yet, lets fire it up!')
            if self.config['sampler'] == 'nestle':
                self.run_nestle()
            elif self.config['sampler'] == 'multinest':
                self.run_multinest()
            else:
                raise NotImplementedError(
                    f'No such sampler as {self.config["sampler"]}, perhaps '
                    f'try nestle or multinest')
        else:
            self.log.info(f'did run')

    def check_did_save(self):
        self.log.info(f'did not save yet, we don\'t want to lose our results so better do it now')
        if self.log_dict['saved_in'] is None:
            self.save_results()

    def log_probability_nested(self, parameter_vals, parameter_names):
        """

        :param parameter_vals: the values of the model/benchmark considered as the truth
        # :param parameter_values: the values of the parameters that are being varied
        :param parameter_names: the names of the parameter_values
        :return:
        """
        self.log.debug(f'there we go! Find that log probability')
        evaluated_rate = self.eval_spectrum(parameter_vals, parameter_names)[
            'counts']

        ll = statistics.log_likelihood(self.benchmark_values, evaluated_rate)
        if np.isnan(ll):
            raise ValueError(f"Returned NaN from likelihood. ll = {ll}")
        self.log.debug(f'found it! returning the log likelihood')
        return ll

    def log_prior_transform_nested(self, x, x_name):
        self.log.debug(f'doing some transformations for nestle/multinest to read the priors')
        this_prior = self.config['prior'][x_name]
        prior_type = this_prior['prior_type']

        if prior_type == 'flat':
            a, b = this_prior['param']
            # Prior transform of a flat prior is a simple line.
            return x * (b - a) + a
        if prior_type == 'gauss':
            # Get the range from the config file
            a, b = this_prior['range']
            m, s = this_prior['param']

            # Here the prior transform is being constructed and shifted. This may not seem trivial
            # and one is advised to request a notebook where this is explained
            # from the developer(s).
            aprime = spsp.ndtr((a - m) / s)
            bprime = spsp.ndtr((b - m) / s)
            xprime = x * (bprime - aprime) + aprime
            res = m + s * spsp.ndtri(xprime)
            return res
        raise ValueError(f"unknown prior type '{prior_type}'")

    def _log_probability_nested(self, theta):
        """warp log_prior_transform_nested"""
        ndim = len(theta)
        result = self.log_probability_nested(
            theta, self.known_parameters[:ndim])
        return result

    def _log_prior_transform_nested(self, theta):
        result = [
            self.log_prior_transform_nested(val, self.known_parameters[i])
            for i, val in enumerate(theta)]
        return np.array(result)

    def run_nestle(self):
        self._fix_parameters()
        self._print_before_run()
        if self.config['sampler'] != 'nestle':
            raise RuntimeError(f'Trying to run nestle but initialization '
                               f'requires {self.config["sampler"]}')

        # Do the import of nestle inside the class such that the package can be
        # loaded without nestle
        try:
            import nestle
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'package nestle not found. See README for installation')

        self.log.debug(f'We made it to my core function, lets do that optimization')
        method = 'multi'  # use MutliNest algorithm
        ndim = len(self.config['fit_parameters'])
        tol = self.config['tol']  # the stopping criterion

        assert_str = f"Unknown configuration of fit pars: {self.config['fit_parameters']}"
        assert self.config["fit_parameters"] == self.known_parameters[:ndim], assert_str

        self.log.warning(f'run_nestle::\tstart_fit for {ndim} parameters')

        start = datetime.datetime.now()
        try:
            self.result = nestle.sample(
                self._log_probability_nested,
                self._log_prior_transform_nested,
                ndim,
                method=method,
                npoints=self.config['nlive'],
                maxiter=self.config.get('max_iter'),
                dlogz=tol)
        except ValueError as e:
            self.config['fit_time'] = -1
            self.log.error(
                f'Nestle did not finish due to a ValueError. Was running with'
                f'{self.config["fit_parameters"]}')
            raise e

        end = datetime.datetime.now()
        dt = (end - start).total_seconds()
        self.log.info(f'fit_done in {dt} s ({dt / 3600} h)')
        self.config['fit_time'] = dt
        self.log_dict['did_run'] = True
        self.log.info(f'Finished with running optimizer!')

    def _print_before_run(self):
        self.log.warning(
            f"""
            --------------------------------------------------
            {utils.now()}\n\tFinal print of all of the set options:
            self.config['tol'] = {self.config['tol']}
            self.config['nlive'] = {self.config['nlive']}
            self.config["sampler"] = {self.config["sampler"]}
            self.log = {self.log}
            self.result = {self.result}
            self.config["fit_parameters"] = {self.config["fit_parameters"]}
            halo_model = {self.config['halo_model']} with:
                v_0 = {self.config['halo_model'].parameter_dict()}
            self.benchmark_values = {np.array(self.benchmark_values)}
            self.config = {self.config}
            --------------------------------------------------
            """
        )

    def run_multinest(self):
        self._fix_parameters()
        self._print_before_run()
        if self.config["sampler"] != 'multinest':
            raise ValueError(f'Wrong sampler {self.config["sampler"]}!')
        # Do the import of multinest inside the class such that the package can be
        # loaded without multinest
        try:
            from pymultinest.solve import run, Analyzer, solve
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'package pymultinest not found. See README for installation')

        n_dims = len(self.config["fit_parameters"])
        tol = self.config['tol']  # the stopping criterion
        save_at = self.get_save_dir()
        assert_str = f'Unknown configuration of fit pars: {self.config["fit_parameters"]}'
        assert self.config["fit_parameters"] == self.known_parameters[:n_dims], assert_str

        self.log.warning(f'start_fit for %i parameters' % n_dims)

        start = datetime.datetime.now()

        # Multinest saves output to a folder. First write to the tmp folder,
        # move it to the results folder later
        _tmp_folder = self.get_save_dir()
        save_at_temp = os.path.join(_tmp_folder, 'multinest')

        solve_multinest(
            LogLikelihood=self._log_probability_nested,  # SafeLoglikelihood,
            Prior=self._log_prior_transform_nested,  # SafePrior,
            n_live_points=self.config['nlive'],
            n_dims=n_dims,
            outputfiles_basename=save_at_temp,
            verbose=True,
            evidence_tolerance=tol,
            # null_log_evidence=statistics.LL_LOW_BOUND,
            max_iter=self.config.get('max_iter', 0),

        )
        self.result_file = save_at_temp

        # Open a save-folder after successful running multinest. Move the
        # multinest results there.
        utils.check_folder_for_file(save_at)
        end = datetime.datetime.now()
        dt = (end - start).total_seconds()
        self.log.info(f'fit_done in {dt} s ({dt / 3600} h)')
        self.log_dict['did_run'] = True
        self.config['fit_time'] = dt

        self.log.info(f'Finished with running Multinest!')

    def get_summary(self):
        self.log.info(f"getting the summary (or at least trying) let\'s first see if I did run")
        self.check_did_run()
        # keep a dictionary of all the results
        resdict = {}

        if self.config["sampler"] == 'multinest':
            # Do the import of multinest inside the class such that the package can be
            # loaded without multinest
            try:
                from pymultinest.solve import run, Analyzer, solve
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'package pymultinest not found. See README for installation')
            self.log.info('start analyzer of results')
            analyzer = Analyzer(len(self.config['fit_parameters']),
                                outputfiles_basename=self.result_file)
            # Taken from multinest.solve
            self.result = analyzer.get_stats()
            samples = analyzer.get_equal_weighted_posterior()[:, :-1]

            self.log.info('parameter values:')
            for name, col in zip(self.config['fit_parameters'],
                                 samples.transpose()):
                self.log.info(
                    '%15s : %.3f +- %.3f' %
                    (name, col.mean(), col.std()))
                resdict[name + '_fit_res'] = (
                    '{0:5.2f} +/- {1:5.2f}'.format(col.mean(), col.std()))
                if 'log_' in name:
                    resdict[name[4:] + '_fit_res'] = '%.3g +/- %.2g' % (
                        10. ** col.mean(), 10. ** (col.mean()) * np.log(10.) * col.std())
                    self.log.info(f'\t {name[4:]},'
                                  f' {resdict[name[4:] + "_fit_res"]}')
            resdict['n_samples'] = len(samples.transpose()[0])
            # Pass the samples to the self.result to be saved.
            self.result['samples'] = samples
        elif self.config["sampler"] == 'nestle':
            # Do the import of nestle inside the class such that the package can be
            # loaded without nestle
            try:
                import nestle
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'package nestle not found. See README for installation')
            # taken from mattpitkin.github.io/samplers-demo/pages/samplers-samplers-everywhere/#Nestle  # noqa
            # estimate of the statistical uncertainty on logZ
            logZerrnestle = np.sqrt(self.result.h / self.config['nlive'])
            # re-scale weights to have a maximum of one
            nweights = self.result.weights / np.max(self.result.weights)
            # get the probability of keeping a sample from the weights
            keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
            # get the posterior samples
            samples_nestle = self.result.samples[keepidx, :]
            # estimate of the statistcal uncertainty on logZ
            resdict['nestle_nposterior'] = len(samples_nestle)
            resdict['nestle_time'] = self.config['fit_time']  # run time
            # log marginalised likelihood
            resdict['nestle_logZ'] = self.result.logz
            resdict['nestle_logZerr'] = logZerrnestle  # uncertainty on log(Z)
            resdict['summary'] = self.result.summary()
            p, cov = nestle.mean_and_cov(
                self.result.samples, self.result.weights)
            for i, key in enumerate(self.config['fit_parameters']):
                resdict[key + '_fit_res'] = (
                    '{0:5.2f} +/- {1:5.2f}'.format(p[i], np.sqrt(cov[i, i])))
                self.log.info(f'\t, {key}, {resdict[key + "_fit_res"]}')
                if 'log_' in key:
                    resdict[key[4:] + '_fit_res'] = '%.3g +/- %.2g' % (
                        10. ** p[i], 10. ** (p[i]) * np.log(10) * np.sqrt(cov[i, i]))
                    self.log.info(
                        f'\t, {key[4:]}, {resdict[key[4:] + "_fit_res"]}')
        self.log.info(f"Alright we got all the info we need")
        return resdict

    def get_save_dir(self, force_index=False, _hash=None) -> str:
        saved_in = self.log_dict['saved_in']
        saved_ok = isinstance(saved_in, str) and os.path.exists(saved_in)
        if saved_ok and not force_index:
            return saved_in
        target_save = utils.open_save_dir(f'nes_{self.config["sampler"][:2]}',
                                          force_index=force_index,
                                          _hash=_hash)
        self.log_dict['saved_in'] = target_save
        self.log.info(f'get_save_dir\tsave_dir = {target_save}')
        return target_save

    def get_tmp_dir(self, force_index=False, _hash=None):
        if (not self.log_dict['tmp_dir']) or force_index:
            self.log_dict['tmp_dir'] = utils.open_save_dir(
                f'{self.config["sampler"]}',
                base_dir=context.context['tmp_folder'],
                force_index=force_index,
                _hash=_hash)
        self.log.info(
            f'get_tmp_dir\ttmp_dir = {self.log_dict["tmp_dir"]}')
        return self.log_dict['tmp_dir']

    def save_results(self, force_index=False):
        self.log.info('Saving results after checking we did run')
        # save fit parameters to config
        self.check_did_run()
        save_dir = self.get_save_dir(force_index=force_index)
        fit_summary = self.get_summary()
        self.log.info(f'storing in {save_dir}')
        # save the config, chain and flattened chain
        pid_id = 'pid' + str(os.getpid()) + '_'
        with open(os.path.join(save_dir, f'{pid_id}config.json'), 'w') as file:
            json.dump(convert_dic_to_savable(self.config), file, indent=4)
        with open(os.path.join(save_dir, f'{pid_id}res_dict.json'), 'w') as file:
            json.dump(convert_dic_to_savable(fit_summary), file, indent=4)
        np.save(
            os.path.join(save_dir, f'{pid_id}config.npy'),
            convert_dic_to_savable(self.config))
        np.save(os.path.join(save_dir, f'{pid_id}res_dict.npy'),
                convert_dic_to_savable(fit_summary))

        for col in self.result.keys():
            if col == 'samples' or not isinstance(col, dict):
                if self.config["sampler"] == 'multinest' and col == 'samples':
                    # in contrast to nestle, multinest returns the weighted
                    # samples.
                    store_at = os.path.join(save_dir,
                                            f'{pid_id}weighted_samples.npy')
                    np.save(store_at, self.result[col])
                else:
                    store_at = os.path.join(
                        save_dir,
                        pid_id + col + '.npy')
                    np.save(store_at, self.result[col])
            else:
                np.save(os.path.join(save_dir, pid_id + col + '.npy'),
                        convert_dic_to_savable(self.result[col]))
        if 'logging' in self.config:
            store_at = os.path.join(save_dir,
                                    self.config['logging'].split('/')[-1])
            shutil.copy(self.config['logging'], store_at)
        self.log.info(f'save_results::\tdone_saving')

    def show_corner(self):
        self.check_did_save()
        save_dir = self.log_dict['saved_in']

        if self.config['sampler'] == 'multinest':
            combined_results = load_multinest_samples_from_file(save_dir)
            multinest_corner(combined_results, save_dir)
        elif self.config['sampler'] == 'nestle':
            combined_results = load_nestle_samples_from_file(save_dir)
            nestle_corner(combined_results, save_dir)
        else:
            # This cannot happen
            raise ValueError(f"Impossible, sampler was {self.config['sampler']}")
        self.log.info(f'Enjoy the plot. Maybe you do want to save it too?')


class CombinedInference(NestedSamplerStatModel):
    def __init__(self, targets, *args, **kwargs):
        NestedSamplerStatModel.__init__(self, *args, **kwargs)

        if not np.all([t in detector.experiment for t in targets]):
            raise NotImplementedError(
                f'Insert tuple of sub-experiments. {targets} are incorrect format')
        if len(targets) < 2:
            self.log.warning(
                "Don't use this class for single experiments! Use NestedSamplerStatModel instead")
        self.log.debug(f'Register {targets}')
        self.sub_detectors = targets
        self.config['sub_sets'] = targets
        self.sub_classes = [
            NestedSamplerStatModel(det)
            for det in self.sub_detectors
        ]
        self.log.debug(f'Sub detectors are set: {self.sub_classes}')

    def _fix_parameters(self):
        """Fix the parameters of the sub classes"""
        for c in self.sub_classes:
            self.log.debug(f'Fixing parameters for{c}')
            c._fix_parameters()

    def _log_probability_nested(self, theta):
        return np.sum([c._log_probability_nested(theta)
                       for c in self.sub_classes])

    def copy_config(self, keys):
        for k in keys:
            if k not in self.config:
                raise ValueError(
                    f'One or more of keys not in config: '
                    f'{np.setdiff1d(keys, list(self.config.keys()))}')
        copy_of_config = {k: self.config[k] for k in keys}
        self.log.info(f'update config with {copy_of_config}')
        for c in self.sub_classes:
            self.log.debug(f'{c} with config {c.config}')
            c._fix_parameters()
            c._print_before_run()

    def save_sub_configs(self, force_index=False):
        save_dir = self.get_save_dir(force_index=force_index)
        self.log.info(
            f'CombinedInference::\tSave configs of sub_experiments to {save_dir}')
        # save the config
        save_dir = os.path.join(save_dir, 'sub_exp_configs')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for c in self.sub_classes:
            save_as = os.path.join(f'{save_dir}', f'{c.config["detector"]}_')
            with open(save_as + 'config.json', 'w') as file:
                json.dump(convert_dic_to_savable(c.config), file, indent=4)
            np.save(save_as + 'config.npy', convert_dic_to_savable(c.config))
            shutil.copy(c.config['logging'], save_as +
                        c.config['logging'].split('/')[-1])
            self.log.info(f'save_sub_configs::\tdone_saving')


def convert_dic_to_savable(config):
    result = config.copy()
    for key in result.keys():
        if utils.is_savable_type(result[key]):
            pass
        elif isinstance(result[key], dict):
            result[key] = convert_dic_to_savable(result[key])
        else:
            result[key] = str(result[key])
    return result


def load_nestle_samples(
        load_from='nested',
        base=utils.get_result_folder(),
        item='latest'):
    save = load_from
    files = os.listdir(base)
    if item == 'latest':
        _selected_files = [int(f.split(save)[-1]) for f in files if save in f]
        if not _selected_files:
            raise FileNotFoundError(
                f'No results in {base}. That only has {files}')
        item = max(_selected_files)

    load_dir = os.path.join(base, save + str(item) + '/')
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f'Cannot find {load_dir} specified by arg: '
                                f'{item}')
    return load_nestle_samples_from_file(load_dir)


def load_nestle_samples_from_file(load_dir):
    log.info(f'load_nestle_samples::\tloading {load_dir}')
    keys = ['config', 'res_dict', 'h', 'logl', 'logvol', 'logz', 'logzerr',
            'ncall', 'niter', 'samples', 'weights']
    result = {}
    files_in_dir = os.listdir(load_dir)
    for key in keys:
        for file in files_in_dir:
            if key + '.npy' in file:
                result[key] = np.load(
                    os.path.join(load_dir, file),
                    allow_pickle=True)
                break
        else:
            raise FileNotFoundError(f'No {key} in {load_dir} only:\n{files_in_dir}')
        if key == 'config' or key == 'res_dict':
            result[key] = result[key].item()
    log.info(
        f"load_nestle_samples::\tdone loading\naccess result with:\n{keys}")
    return result


def load_multinest_samples_from_file(load_dir):
    keys = os.listdir(load_dir)
    keys = [key for key in keys if os.path.isfile(os.path.join(load_dir, key))]
    result = {}
    for key in keys:
        if '.npy' in key:
            naked_key = key.split('.npy')[0]
            naked_key = do_strip_from_pid(naked_key)
            tmp_res = np.load(os.path.join(load_dir, key), allow_pickle=True)
            if naked_key == 'config' or naked_key == 'res_dict':
                result[naked_key] = tmp_res.item()
            else:
                result[naked_key] = tmp_res
    return result


def do_strip_from_pid(string):
    """
    remove PID identifier from a string
    """
    if 'pid' not in string:
        return string

    new_key = string.split("_")
    new_key = "_".join(new_key[1:])
    return new_key


def load_multinest_samples(load_from='nested', item='latest'):
    base = utils.get_result_folder()
    save = load_from
    files = os.listdir(base)
    if item == 'latest':
        item = max([int(f.split(save)[-1]) for f in files if save[:3] in f])

    load_dir = os.path.join(base, save + str(item))
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Cannot find {load_dir} specified by arg: "
                                f"{item}")
    return load_multinest_samples_from_file(load_dir)


def _get_info(result, _result_key):
    info = r"$M_\chi}$=%.2f" % 10. ** np.float(result['config']['mw'])
    for prior_key in result['config']['prior'].keys():
        if (prior_key in result['config']['prior'] and
                'mean' in result['config']['prior'][prior_key]):
            mean = result['config']['prior'][prior_key]['mean']
            info += f"\n{prior_key} = {mean}"
    nposterior, ndim = np.shape(result[_result_key])
    info += "\nnposterior = %s" % nposterior
    for str_inf in ['detector', 'notes', 'start', 'fit_time', 'poisson',
                    'n_energy_bins']:
        if str_inf in result['config']:
            info += f"\n{str_inf} = %s" % result['config'][str_inf]
            if str_inf == 'start':
                info = info[:-7]
            if str_inf == 'fit_time':
                info += 's (%.1f h)' % (float(result['config'][str_inf]) / 3600.)
    return info, ndim


def multinest_corner(
        result,
        save=False,
        _result_key='weighted_samples',
        _weights=False):
    info, ndim = _get_info(result, _result_key)
    labels = statistics.get_param_list()[:ndim]
    truths = []
    for prior_name in statistics.get_prior_list()[:ndim]:
        if prior_name != "rho_0":
            truths.append(result['config'][prior_name])
        else:
            truths.append(result['config']['density'])
    weight_kwargs = dict(weights=result['weights']) if _weights else {}
    fig = corner.corner(
        result[_result_key],
        **weight_kwargs,
        labels=labels,
        range=[0.99999, 0.99999, 0.99999, 0.99999, 0.99999][:ndim],
        truths=truths,
        show_titles=True)
    fig.axes[1].set_title(f"Fit title", loc='left')
    fig.axes[1].text(0, 1, info, verticalalignment='top')
    if save:
        plt.savefig(f"{save}corner.png", dpi=200)


def nestle_corner(result, save=False):
    multinest_corner(result, save, _result_key='samples', _weights=True)


def solve_multinest(LogLikelihood, Prior, n_dims, **kwargs):
    """
    See PyMultinest Solve() for documentation
    """
    from pymultinest.solve import run, Analyzer
    kwargs['n_dims'] = n_dims
    files_temporary = False
    if 'outputfiles_basename' not in kwargs:
        files_temporary = True
        tempdir = tempfile.mkdtemp('pymultinest')
        kwargs['outputfiles_basename'] = tempdir + '/'
    outputfiles_basename = kwargs['outputfiles_basename']

    def SafePrior(cube, ndim, nparams):
        a = np.array([cube[i] for i in range(n_dims)])
        b = Prior(a)
        for i in range(n_dims):
            cube[i] = b[i]

    def SafeLoglikelihood(cube, ndim, nparams, lnew):
        a = np.array([cube[i] for i in range(n_dims)])
        likelihood = float(LogLikelihood(a))
        if not np.isfinite(likelihood):
            warn('WARNING: loglikelihood not finite: %f\n' % likelihood)
            warn('         for parameters: %s\n' % a)
            warn('         returned very low value instead\n')
            return -statistics.LL_LOW_BOUND
        return likelihood

    kwargs['LogLikelihood'] = SafeLoglikelihood
    kwargs['Prior'] = SafePrior
    run(**kwargs)

    analyzer = Analyzer(
        n_dims, outputfiles_basename=outputfiles_basename)
    try:
        stats = analyzer.get_stats()
    except ValueError as e:
        # This can happen during testing if we limit the number of iterations
        warn(f'Cannot load output file: {e}')
        stats = {'nested sampling global log-evidence': -1,
                 'nested sampling global log-evidence error': -1
                 }
    samples = analyzer.get_equal_weighted_posterior()[:, :-1]

    return dict(logZ=stats['nested sampling global log-evidence'],
                logZerr=stats['nested sampling global log-evidence error'],
                samples=samples,
                )
