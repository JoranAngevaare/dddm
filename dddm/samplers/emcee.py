"""
Do a likelihood fit. The class MCMCStatModel is used for fitting applying
the MCMC algorithm emcee.

MCMC is:
    slower than the nestle package; and
    harder to use since one has to choose the 'right' initial parameters

Nevertheless, the walkers give great insight in how the likelihood-function is
felt by the steps that the walkers make
"""
from warnings import warn
import datetime
import json
import os
import corner
import matplotlib.pyplot as plt
import numpy as np
from dddm import statistics, utils
import dddm
import typing as ty

export, __all__ = dddm.exporter()
log = dddm.utils.log


def default_emcee_save_dir():
    """The name of folders where to save results from the MCMCStatModel"""
    return 'emcee'


@export
class MCMCStatModel(statistics.StatModel):

    def __init__(
            self,
            wimp_mass: ty.Union[float, int],
            cross_section: ty.Union[float, int],
            spectrum_class: ty.Union[dddm.DetectorSpectrum,
                                     dddm.GenSpectrum],
            prior: dict,
            tmp_folder: str,
            fit_parameters=('log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density', 'k'),

            detector_name=None,
            verbose=False,
            notes='default',
            nwalkers=50,
            nsteps=100,
            remove_frac=0.2,
            emcee_thin=15

    ):
        super().__init__(wimp_mass=wimp_mass,
                         cross_section=cross_section,
                         spectrum_class=spectrum_class,
                         prior=prior,
                         tmp_folder=tmp_folder,
                         fit_parameters=fit_parameters,
                         detector_name=detector_name,
                         verbose=verbose,
                         notes=notes)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        # self.config['fit_parameters'] = ['log_mass', 'log_cross_section']
        self.sampler = None
        self.pos = None
        self.log_dict = {'sampler': False, 'did_run': False, 'pos': False}
        self.remove_frac = remove_frac
        self.emcee_thin = emcee_thin

    def _set_pos(self, use_pos=None):
        """Set the starting position of the walkers"""
        self.log_dict['pos'] = True
        if use_pos is not None:
            self.log.info("using specified start position")
            self.pos = use_pos
            return
        nparameters = len(self.config['fit_parameters'])
        keys = statistics.get_prior_list()[:nparameters]

        ranges = [self.config['prior'][self.config['fit_parameters'][i]]['range']
                  for i in range(nparameters)]
        pos = []

        for i, key in enumerate(keys):
            val = getattr(self, key)
            self.log.warning(f'{key} is {val}')
            a, b = ranges[i]
            start_at = val + 0.005 * val * np.random.randn(self.nwalkers, 1)

            start_at = np.clip(start_at, a, b)
            pos.append(start_at)
        pos = np.hstack(pos)
        self.pos = pos

    def set_sampler(self, mult=True):
        """init the MCMC sampler"""
        # Do the import of emcee inside the class such that the package can be
        # loaded without emcee
        try:
            import emcee
        except ModuleNotFoundError:
            raise ModuleNotFoundError('package emcee not found. See README')

        ndim = len(self.config['fit_parameters'])
        self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim,
                                             self.log_probability,
                                             args=([self.config['fit_parameters']]),
                                             )
        self.log_dict['sampler'] = True

    def run(self):
        self._fix_parameters()

        if not self.log_dict['sampler']:
            self.set_sampler()
        if not self.log_dict['pos']:
            self._set_pos()
        start = datetime.datetime.now()
        try:
            self.sampler.run_mcmc(self.pos, self.nsteps, progress=False)
        except ValueError as e:
            raise ValueError(
                f"MCMC did not finish due to a ValueError. Was running with\n"
                f"pos={self.pos.shape} nsteps = {self.nsteps}, walkers = "
                f"{self.nwalkers}, ndim = "
                f"{len(self.config['fit_parameters'])} for fit parameters "
                f"{self.config['fit_parameters']}") from e
        end = datetime.datetime.now()
        self.log_dict['did_run'] = True
        dt = (end - start).total_seconds()
        self.log.info(f"fit_done in {dt} s ({dt / 3600} h)")
        # Release config for writing!
        self.config = utils._immutable_to_dict(self.config)
        self.config['fit_time'] = dt

    def show_walkers(self):
        if not self.log_dict['did_run']:
            self.run()
        labels = self.config['fit_parameters']
        fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        for i, label_i in enumerate(labels):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(label_i)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")

    def show_corner(self):
        if not self.log_dict['did_run']:
            self.run()
        self.log.info(
            f"Removing a fraction of {self.remove_frac} of the samples, total"
            f"number of removed samples = {self.nsteps * self.remove_frac}")
        flat_samples = self._get_chain_flat_chain()
        truths = [getattr(self, prior_name) for prior_name in
                  statistics.get_prior_list()[:len(self.config['fit_parameters'])]]

        corner.corner(flat_samples, labels=self.config['fit_parameters'], truths=truths)

    def _get_chain_flat_chain(self):
        return self.sampler.get_chain(
            discard=int(self.nsteps * self.remove_frac),
            thin=self.emcee_thin, flat=True
        )

    def save_results(
            self,
            save_to_dir=default_emcee_save_dir(),
            force_index=False):
        # save fit parameters to config
        self.config['fit_parameters'] = self.config['fit_parameters']
        if not self.log_dict['did_run']:
            self.run()
        # open a folder where to save to results
        save_dir = dddm.context.open_save_dir(
            default_emcee_save_dir(),
            base_dir=save_to_dir,
            force_index=force_index)
        # save the config, chain and flattened chain
        with open(os.path.join(save_dir, 'config.json'), 'w') as fp:
            json.dump(utils.convert_dic_to_savable(self.config), fp, indent=4)
        np.save(os.path.join(save_dir, 'config.npy'),
                utils.convert_dic_to_savable(self.config))

        save_at = os.path.join(save_dir, 'full_chain.npy')
        np.save(save_at, self.sampler.get_chain())

        save_at = os.path.join(save_dir, 'flat_chain.npy')
        flat_chain = self._get_chain_flat_chain()
        np.save(save_at, flat_chain)
        self.config['save_dir'] = save_dir
        self.log.info("save_results::\tdone_saving")


def load_chain_emcee(load_from,
                     item='latest'):
    files = os.listdir(load_from)
    if item == 'latest':
        try:
            item = files[-1]
        except ValueError:
            log.warning(files)
            item = 0
    result = {}
    load_dir = os.path.join(load_from, str(item))
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Cannot find {load_dir} specified by arg: "
                                f"{item}")
    log.info(f"loading {load_dir}")

    keys = ['config', 'full_chain', 'flat_chain']

    for key in keys:
        result[key] = np.load(
            os.path.join(
                load_dir,
                key + '.npy'),
            allow_pickle=True)
        if key == 'config':
            result[key] = result[key].item()
    log.info(f"done loading\naccess result with:\n{keys}")
    return result


def emcee_plots(result, save=False, plot_walkers=True, show=False):
    if not isinstance(save, bool):
        assert os.path.exists(save), f"invalid path '{save}'"
    info = r"$M_\chi}$=%.2f" % 10 ** np.float64(result['config']['log_mass'])
    for prior_key in result['config']['prior'].keys():
        try:
            mean = result['config']['prior'][prior_key]['mean']
            info += f"\n{prior_key} = {mean}"
        except KeyError:
            pass
    nsteps, nwalkers, ndim = np.shape(result['full_chain'])

    for str_inf in ['notes', 'start', 'fit_time', 'poisson', 'nwalkers', 'nsteps', 'n_energy_bins']:
        try:
            info += f"\n{str_inf} = %s" % result['config'][str_inf]
            if str_inf == 'start':
                info = info[:-7]
            if str_inf == 'fit_time':
                info += 's (%.1f h)' % (float(result['config'][str_inf]) / 3600.)
        except KeyError:
            pass
    info += "\nnwalkers = %s" % nwalkers
    info += "\nnsteps = %s" % nsteps
    labels = statistics.get_param_list()[:ndim]
    truths = [result['config'][prior_name]
              if prior_name in result['config']
              else result['config']['prior'][prior_name]['mean']
              for prior_name in
              statistics.get_prior_list()[:ndim]]
    fig = corner.corner(
        result['flat_chain'],
        labels=labels,
        range=[0.99999, 0.99999, 0.99999, 0.99999, 0.99999][:ndim],
        truths=truths,
        show_titles=True)
    fig.axes[1].set_title(f"{result['config']['detector']}", loc='left')
    fig.axes[1].text(0, 1, info, verticalalignment='top')

    if plot_walkers:
        _plot_walkers(result, truths, labels, save, show)
    _plt_cleanup(f"{save}corner.png", save, show)


def _plot_walkers(result, truths, labels, save, show):
    fig, axes = plt.subplots(len(labels), figsize=(10, 5), sharex=True)
    for i, label_i in enumerate(labels):
        ax = axes[i]
        ax.plot(result['full_chain'][:, :, i], "k", alpha=0.3)
        ax.axhline(truths[i])
        ax.set_xlim(0, len(result['full_chain']))
        ax.set_ylabel(label_i)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    _plt_cleanup(f"{save}flat_chain.png", save, show)


def _plt_cleanup(name, save, show):
    if save:
        plt.savefig(name, dpi=200)
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()
