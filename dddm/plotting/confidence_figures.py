"""Utility for opening and displaying results from multinest optimization"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import dddm
from glob import glob
from tqdm import tqdm
import traceback

export, __all__ = dddm.exporter()


@export
class DDDMResult:
    """Parse results from fitting from nested sampling"""
    result: dict = None

    def __init__(self, path, sampler='multinest'):
        """
        Open a class for organizing the results from running an optimization
        :param path: Path to the base dir of the results to open
        """
        assert os.path.exists(path)
        self.path = path
        self.log = dddm.utils.get_logger(self.__class__.__name__)
        self.sampler = sampler
        self.setup()

    def setup(self):
        if self.sampler == 'multinest':
            self.result = dddm.samplers.pymultinest.load_multinest_samples_from_file(self.path)
        elif self.sampler == 'nestle':
            self.result = dddm.samplers.nestle.load_nestle_samples_from_file(self.path)
        else:
            raise RuntimeError(f'{self.sampler} is invalid')

    def __repr__(self):
        # Can we avoid duplication with config summary or make a property factory?
        get_props = ('detector', 'mass', 'sigma', 'nlive',
                     'halo_model', 'notes', 'n_parameters',
                     )
        to_print = [f'{p}: {getattr(self, p)}' for p in get_props]
        return ', '.join(to_print)

    def config_summary(self,
                       get_props=(
                               'detector',
                               'mass',
                               'sigma',
                               'nlive',
                               'halo_model',
                               'notes',
                               'n_parameters',
                       )
                       ) -> pd.DataFrame:
        df = {k: [getattr(self, k)] for k in get_props}
        return pd.DataFrame(df)

    def result_summary(self) -> pd.DataFrame:
        df = {k: [v] for k, v in self.result.get('res_dict', {}).items()}
        return pd.DataFrame(df)

    def summary(self) -> pd.DataFrame:
        return pd.concat([self.config_summary(),
                          self.result_summary()],
                         axis=1)

    def get_from_config(self, to_get: str, if_not_available=None):
        return self.result.get('config', {}).get(to_get, if_not_available)

    def get_samples(self):
        return self.result.get('weighted_samples').T[:2]

    @property
    def detector(self):
        return str(self.get_from_config('detector'))

    @property
    def nlive(self):
        return int(self.get_from_config('nlive', -1))

    @property
    def sigma(self):
        sigma = float(self.get_from_config('sigma', 0))
        if sigma == 0:
            sigma = float(self.get_from_config('log_cross_section', 0))
        return sigma

    @property
    def mass(self):
        log_mass = self.get_from_config('log_mass', None)
        if log_mass is None:
            return -1
        return round(np.power(10, log_mass), 3)

    @property
    def halo_model(self):
        return str(self.get_from_config('halo_model'))

    @property
    def notes(self):
        return str(self.get_from_config('notes'))

    @property
    def n_parameters(self):
        param = self.get_from_config('fit_parameters')
        if param is None:
            return None

        return len(param)


@export
class SeabornPlot:
    def __init__(self, result: DDDMResult):
        self.result = result
        self.log = dddm.utils.get_logger(self.__class__.__name__)

    def __repr__(self):
        return f'{self.__class__.__name__}:: {self.result.__repr__()}'

    def plot_samples(self, **kwargs) -> None:
        kwargs.setdefault('s', 1)
        kwargs.setdefault('facecolor', 'gray')
        kwargs.setdefault('alpha', 0.2)
        self.log.debug(f'setting kwargs to {kwargs}')
        plt.scatter(*self.samples, **kwargs)

    def plot_bench(self, c='cyan', **kwargs):
        plt.scatter(self.result.get_from_config('log_mass'),
                    self.result.sigma,
                    s=10 ** 2,
                    edgecolors='black',
                    c=c, marker='X', zorder=1000, **kwargs)

    @property
    def samples(self) -> np.ndarray:
        return self.result.get_samples()

    def best_fit(self) -> tuple:
        best = np.mean(self.samples, axis=1)
        std = np.std(self.samples, axis=1)
        return best, std

    def plot_best_fit(self, **kwargs) -> None:
        kwargs.setdefault('capsize', 5)
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linewidth', 2)
        kwargs.setdefault('zorder', 300)
        (x, y), (x_err, y_err) = self.best_fit()
        plt.errorbar(x, y, xerr=x_err, yerr=y_err, **kwargs)

    def samples_to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {'mass': self.samples[0],
             'cross_section': self.samples[1]}
        )

    def plot_sigma_contours(self, nsigma=2, **kwargs):
        kwargs.setdefault('bw_adjust', 0.25)
        levels = sorted((1 - np.array([0.6827, 0.9545, 0.9973][:nsigma])))
        kwargs.setdefault('levels', levels)

        df = self.samples_to_df()
        sns.kdeplot(data=df, x='mass', y='cross_section', **kwargs)

    def plot_kde(self, **kwargs):
        kwargs.setdefault('levels', 100)
        kwargs.setdefault('fill', True)
        self.plot_sigma_contours(**kwargs)


@export
class ResultsManager:
    result_cache: list = None
    result_df: pd.DataFrame = None

    def __init__(self, pattern=None, sampler='multinest'):
        self.sampler = sampler
        self.log = dddm.utils.get_logger(self.__class__.__name__)
        if pattern is not None:
            self.register_pattern(pattern)

    def __repr__(self):
        return f'info for {len(self.result_cache)}'

    def _add_result(self, path: str, tolerant=False):
        if self.result_cache is None:
            self.result_cache = []
        try:
            result = DDDMResult(path, sampler=self.sampler)
        except KeyboardInterrupt as interrupt:
            raise interrupt
        except Exception as e:
            if not tolerant:
                raise e
            self.log.debug(e)
            self.log.warning(f'loading {path} lead to {e}, {traceback.format_exc()}')
            return
        self.result_cache.append(result)

    def add_result(self, path: str):
        self._add_result(path)
        self.build_df()

    def register_pattern(self, pattern, show_tqdm=True):
        matches = glob(pattern)
        if len(matches) == 0:
            raise ValueError(f'No matches for {pattern}')
        self.log.info(f'Opening {len(matches)} matches')
        for path in tqdm(matches, disable=not show_tqdm):
            self.log.debug(f'open {path}')
            self._add_result(path, tolerant=True)
        self.log.info('Opening done, build df')
        self.build_df()

    def apply_mask(self, mask):
        assert len(mask) == len(self.result_cache)
        self.log.debug(f'Removing {np.sum(mask)}/'
                       f'{len(self.result_cache)}')
        new_list = [self.result_cache[m_i] for m_i, m in enumerate(mask) if m]
        del self.result_cache
        self.result_cache = new_list
        del new_list

        assert len(self.result_cache) == np.sum(mask)
        self.build_df()

    def build_df(self):
        dfs = [r.summary() for r in self.result_cache]
        if not len(dfs):
            raise ValueError('No files!')
        self.result_df = pd.concat(dfs)

    @property
    def df(self):
        """Lazy alias"""
        return self.result_df


def _pow10(x):
    return 10 ** x


def set_xticks_top(show_lines=False,
                   rotation=0,
                   x_ticks = (0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000, 10_000),
                   x_label=r"$M_{\chi}$ $[\mathrm{GeV}/\mathrm{c}^{2}]$"):
    ax = plt.gca()
    bin_range = ax.get_xlim()
    secax = ax.secondary_xaxis('top', functions=(_pow10, np.log10))


    x_ticks = [t for t in x_ticks if t > 10 ** bin_range[0] and t < 10 ** bin_range[1]]
    if show_lines:
        for x_tick in x_ticks:
            ax.axvline(np.log10(x_tick), alpha=0.1)

    def str_fmt(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [str_fmt(x) for x in x]
        if x <= 0.1:
            return f'${x:.2f}$'
        if x <= 1:
            return f'${x:.1f}$'
        return f'${int(x)}$'

    secax.set_ticks(x_ticks, labels=str_fmt(x_ticks))
    secax.xaxis.set_tick_params(rotation=rotation)
    secax.set_xlabel(x_label)
    return secax


def x_label():
    plt.xlabel(r"$\log_{10}(M_{\chi}$/$[\mathrm{GeV}/\mathrm{c}^{2}]$)")


def y_label():
    plt.ylabel(r"$\log_{10}(\sigma_{S.I.}$/$[\mathrm{cm}^{2}]$)")
