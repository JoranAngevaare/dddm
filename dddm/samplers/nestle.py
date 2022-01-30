from __future__ import absolute_import, unicode_literals
import datetime
import json
import os
import shutil
import numpy as np
import dddm
from .pymultinest import MultiNestSampler, multinest_corner, convert_dic_to_savable

log = dddm.utils.log
export, __all__ = dddm.exporter()


@export
class NestleSampler(MultiNestSampler):
    def run(self):
        self._fix_parameters()
        self._print_before_run()

        # Do the import of nestle inside the class such that the package can be
        # loaded without nestle
        try:
            import nestle
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'package nestle not found. See README for installation')

        self.log.debug('We made it to my core function, lets do that optimization')
        method = 'multi'  # use MutliNest algorithm
        ndim = len(self.config['fit_parameters'])
        tol = self.config['tol']  # the stopping criterion

        assert_str = f"Unknown configuration of fit pars: {self.config['fit_parameters']}"
        assert tuple(self.config["fit_parameters"]) == tuple(
            self.known_parameters[:ndim]), assert_str

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
        self.config = dddm.utils._immutable_to_dict(self.config)
        self.config['fit_time'] = dt
        self.log_dict['did_run'] = True
        self.log.info('Finished with running optimizer!')

    def get_summary(self):
        self.log.info(
            "getting the summary (or at least trying) let's first see if I did run"
        )

        self.check_did_run()
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
        resdict = {
            'nestle_nposterior': len(samples_nestle),
            'nestle_time': self.config['fit_time'],
            'nestle_logZ': self.result.logz,
            'nestle_logZerr': logZerrnestle,
            'summary': self.result.summary(),
        }

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
        resdict['best_fit'] = p
        resdict['cov_matrix'] = cov
        resdict['weighted_samples'] = samples_nestle
        self.log.info('Alright we got all the info we need')
        return resdict

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
        np.save(os.path.join(save_dir, f'{pid_id}config.npy'),
                convert_dic_to_savable(self.config))
        np.save(os.path.join(save_dir, f'{pid_id}weighted_samples.npy'),
                fit_summary.get('weighted_samples'))
        np.save(os.path.join(save_dir, f'{pid_id}res_dict.npy'),
                convert_dic_to_savable(fit_summary))

        for col in self.result.keys():
            if col == 'samples' or not isinstance(col, dict):
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
        combined_results = load_nestle_samples_from_file(save_dir)
        nestle_corner(combined_results, save_dir)
        self.log.info('Enjoy the plot. Maybe you do want to save it too?')


def load_nestle_samples_from_file(load_dir):
    log.info(f'load_nestle_samples::\tloading {load_dir}')
    keys = ['config', 'res_dict', 'h', 'logl', 'logvol', 'logz', 'logzerr',
            'ncall', 'niter', 'samples', 'weights', 'weighted_samples']
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
        if key in ['config', 'res_dict']:
            result[key] = result[key].item()
    log.info(
        f"load_nestle_samples::\tdone loading\naccess result with:\n{keys}")
    return result


def nestle_corner(result, save=False):
    multinest_corner(result, save)
