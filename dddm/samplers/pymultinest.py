"""
Do a likelihood fit. The class NestedSamplerStatModel is used for fitting
applying the bayesian algorithm nestle/multinest
"""

from __future__ import absolute_import, unicode_literals
import datetime
import json
import os
import shutil
import tempfile
from warnings import warn
import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy import special as spsp
import dddm
import typing as ty
from immutabledict import immutabledict

export, __all__ = dddm.exporter()


@export
class MultiNestSampler(dddm.StatModel):
    def __init__(self,
                 wimp_mass: ty.Union[float, int],
                 cross_section: ty.Union[float, int],
                 spectrum_class: ty.Union[dddm.DetectorSpectrum,
                                          dddm.GenSpectrum],
                 prior: dict,
                 tmp_folder: str,
                 results_dir: str = None,
                 fit_parameters=('log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density', 'k'),
                 detector_name=None,
                 verbose=False,
                 notes='default',
                 nlive=1024,
                 tol=0.1,
                 ):
        super().__init__(wimp_mass=wimp_mass,
                         cross_section=cross_section,
                         spectrum_class=spectrum_class,
                         prior=prior,
                         tmp_folder=tmp_folder,
                         fit_parameters=fit_parameters,
                         detector_name=detector_name,
                         verbose=verbose,
                         notes=notes,
                         )
        self.results_dir = results_dir
        self.config.update(
            {'tol': tol,  # Tolerance for sampling
             'nlive': nlive,  # number of live points
             })
        self.log_dict = {
            'did_run': False,
            'saved_in': None,
            'tmp_dir': tmp_folder,
        }

        self.result = False

    def check_did_run(self):
        if not self.log_dict['did_run']:
            self.log.info('did not run yet, lets fire it up!')
            self.run()
        else:
            self.log.info('did run')

    def check_did_save(self):
        self.log.info(
            "did not save yet, we don't want to lose our results so better do it now"
        )

        if self.log_dict['saved_in'] is None:
            self.save_results()

    def log_probability_nested(self, parameter_vals, parameter_names):
        """

        :param parameter_vals: the values of the model/benchmark considered as the truth
        # :param parameter_values: the values of the parameters that are being varied
        :param parameter_names: the names of the parameter_values
        :return:
        """
        self.log.debug('there we go! Find that log probability')
        evaluated_rate = self.eval_spectrum(parameter_vals, parameter_names)

        ll = dddm.statistics.log_likelihood(self.benchmark_values, evaluated_rate)
        if np.isnan(ll):
            raise ValueError(f"Returned NaN from likelihood. ll = {ll}")
        self.log.debug('found it! returning the log likelihood')
        return ll

    def log_prior_transform_nested(self, x, x_name):
        self.log.debug(
            'doing some transformations for nestle/multinest to read the priors'
        )

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
            return m + s * spsp.ndtri(xprime)
        raise ValueError(f"unknown prior type '{prior_type}'")

    def _log_probability_nested(self, theta):
        """warp log_prior_transform_nested"""
        ndim = len(theta)
        return self.log_probability_nested(
            theta, self.known_parameters[:ndim])

    def _log_prior_transform_nested(self, theta):
        result = [
            self.log_prior_transform_nested(val, self.known_parameters[i])
            for i, val in enumerate(theta)]
        return np.array(result)

    def _print_before_run(self):
        self.log.warning(
            f"""
            --------------------------------------------------
            {dddm.utils.now()}\n\tFinal print of all of the set options:
            self.log = {self.log}
            self.result = {self.result}
            self.benchmark_values = {np.array(self.benchmark_values)}
            self.config = {self.config}
            --------------------------------------------------
            """
        )

    def run(self):
        self._fix_parameters()
        self._print_before_run()
        try:
            from pymultinest.solve import run, Analyzer, solve
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'package pymultinest not found. See README') from e

        n_dims = len(self.config["fit_parameters"])
        tol = self.config['tol']  # the stopping criterion
        save_at = self.get_save_dir()

        self.log.warning(f'start_fit for {n_dims} parameters')

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
            # null_log_evidence=dddm.statistics.LL_LOW_BOUND,
            max_iter=self.config.get('max_iter', 0),

        )
        self.result_file = save_at_temp

        # Open a save-folder after successful running multinest. Move the
        # multinest results there.
        dddm.utils.check_folder_for_file(save_at)
        end = datetime.datetime.now()
        dt = (end - start).total_seconds()
        self.log.info(f'fit_done in {dt} s ({dt / 3600} h)')
        self.log_dict['did_run'] = True
        # release the config
        self.config = dddm.utils._immutable_to_dict(self.config)
        self.config['fit_time'] = dt

        self.log.info('Finished with running Multinest!')

    def get_summary(self):
        self.log.info(
            "getting the summary (or at least trying) let's first see if I did run"
        )

        self.check_did_run()
        # keep a dictionary of all the results
        resdict = {}

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
        resdict['best_fit'] = np.mean(samples.transpose(), axis=1)
        print(resdict['best_fit'])
        resdict['cov_matrix'] = np.cov(samples.transpose())
        print(resdict['cov_matrix'])
        resdict['n_samples'] = len(samples.transpose()[0])
        # Pass the samples to the self.result to be saved.
        self.result['samples'] = samples

        self.log.info('Alright we got all the info we need')
        return resdict

    def get_save_dir(self, force_index=False, _hash=None) -> str:
        saved_in = self.log_dict['saved_in']
        saved_ok = isinstance(saved_in, str) and os.path.exists(saved_in)
        if saved_ok and not force_index:
            return saved_in
        target_save = dddm.context.open_save_dir(
            f'nes_{self.__class__.__name__[:3]}',
            base_dir=self.results_dir,
            force_index=force_index,
            _hash=_hash)
        self.log_dict['saved_in'] = target_save
        self.log.info(f'get_save_dir\tsave_dir = {target_save}')
        return target_save

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
                if col == 'samples':
                    # in contrast to nestle, multinest returns the weighted
                    # samples.
                    store_at = os.path.join(save_dir,
                                            f'{pid_id}weighted_samples.npy')
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
        self.log.info('save_results::\tdone_saving')

    def show_corner(self):
        self.check_did_save()
        save_dir = self.log_dict['saved_in']
        combined_results = load_multinest_samples_from_file(save_dir)
        multinest_corner(combined_results, save_dir)
        self.log.info('Enjoy the plot. Maybe you do want to save it too?')


def convert_dic_to_savable(config):
    result = config.copy()
    if isinstance(config, immutabledict):
        result = dict(config.items())
    for key, value in result.items():
        if dddm.utils.is_savable_type(value):
            continue
        if isinstance(value, (dict, immutabledict)):
            result[key] = convert_dic_to_savable(result[key])
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
        else:
            result[key] = str(result[key])
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
            if naked_key in ['config', 'res_dict']:
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


def _get_info(result, _result_key):
    info = r"$M_\chi}$=%.2f" % 10. ** np.float64(result['config']['log_mass'])
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
    labels = dddm.statistics.get_param_list()[:ndim]
    truths = []
    for prior_name in dddm.statistics.get_prior_list()[:ndim]:
        if prior_name == "rho_0":
            prior_name = 'density'
        if prior_name in result['config']:
            truths.append(result['config'][prior_name])
        else:
            truths.append(result['config']['prior'][prior_name]['mean'])

    weight_kwargs = dict(weights=result['weights']) if _weights else {}
    fig = corner.corner(
        result[_result_key],
        **weight_kwargs,
        labels=labels,
        range=[0.99999, 0.99999, 0.99999, 0.99999, 0.99999][:ndim],
        truths=truths,
        show_titles=True)
    fig.axes[1].set_title('Fit title', loc='left')
    fig.axes[1].text(0, 1, info, verticalalignment='top')
    if save:
        plt.savefig(f"{save}corner.png", dpi=200)


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
            warn(f'WARNING: loglikelihood not finite: {likelihood}\n'
                 f'for parameters {a}, returned very low value instead')
            return -dddm.statistics.LL_LOW_BOUND
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
