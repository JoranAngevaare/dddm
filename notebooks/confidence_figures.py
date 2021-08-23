from IPython.core.pylabtools import figsize
import shutil
import scipy.optimize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sys
import os
import warnings
import pandas as pd
from tqdm import tqdm
import colorsys
from matplotlib.colors import LogNorm
import numpy as np
import DirectDmTargets as dddm
from matplotlib import cm
import scipy

from sklearn.neighbors import KernelDensity


class DDDMResult:
    result: dict = None

    def __init__(self, path):
        assert os.path.exists(path)
        self.path = path
        self.setup()

    def setup(self):
        self.result = dddm.load_multinest_samples_from_file(self.path)

    def __repr__(self):
        to_print = f'det: {self.detector}, '
        to_print += f'mw: {self.mass}, '
        to_print += f'sigma: {self.sigma}, '
        to_print += f'nlive: {self.nlive}, '
        to_print += f'halo_model: {self.halo_model}, '
        to_print += f'notes: {self.notes}, '
        return to_print

    def config_summary(self,
                       keys_from_conf=(
                               'detector',
                               'mass',
                               'sigma',
                               'nlive',
                               'halo_model',
                               'notes',
                       )
                       ) -> pd.DataFrame:
        df = {
            k:
                [self.get_from_config(k)] if k != 'mass' else [self.mass]
            for k in keys_from_conf
        }
        return pd.DataFrame(df)

    def result_summary(self) -> pd.DataFrame:
        df = {k: [v] for k, v in self.result.get('res_dict', {}).items()}
        return pd.DataFrame(df)

    def summary(self) -> pd.DataFrame:
        return pd.concat([self.config_summary(), self.result_summary()], axis=1)

    def get_from_config(self, to_get: str):
        return self.result.get('config', {}).get(to_get)

    @property
    def detector(self):
        return self.get_from_config('detector')

    @property
    def nlive(self):
        return self.get_from_config('nlive')

    @property
    def sigma(self):
        return self.get_from_config('sigma')

    @property
    def mass(self):
        return round(np.power(10, self.get_from_config('mw')), 3)

    @property
    def halo_model(self):
        return self.get_from_config('halo_model')

    @property
    def notes(self):
        return self.get_from_config('notes')

    def get_samples(self):
        return self.result.get('weighted_samples').T[:2]


class ResultPlot:
    def __init__(self, result: DDDMResult):
        self.result = result

    def __repr__(self):
        return f'{self.__class__.__name__} ::{self.result.__repr__()}'

    def plot_samples(self, **kwargs):
        kwargs.setdefault('s', 1)
        kwargs.setdefault('facecolor', 'gray')
        kwargs.setdefault('alpha', 0.2)
        plt.scatter(*self.samples, s=1, facecolor='gray', alpha=0.1)

    @property
    def samples(self):
        return self.result.get_samples()

    def _prior_to_kwargs(self, kwargs):
        if 'range' not in kwargs:
            prior = self.result.get_from_config('prior')
            r = prior['log_mass']['range']
            r += prior['log_cross_section']['range']
            kwargs.setdefault('range', r)
        return kwargs

    def best_fit(self):
        best = np.mean(self.samples, axis=1)
        std = np.std(self.samples, axis=1)
        return best, std

    def plot_best_fit(self, **kwargs):
        kwargs.setdefault('capsize', 5)
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linewidth', 2)
        kwargs.setdefault('zorder', 300)
        (x, y), (x_err, y_err) = self.best_fit()
        plt.errorbar(x, y, xerr=x_err, yerr=y_err, **kwargs)

    def samples_to_df(self):
        return pd.DataFrame(
            {'mass': self.samples[0],
             'cross_section': self.samples[1]}
        )


class SeabornPlot(ResultPlot):
    def plot_sigma_contours(self, nsigma=2, **kwargs):
        kwargs.setdefault('bw_adjust', 0.25)
        levels = (1 - np.array([0.6827, 0.9545, 0.9973][:nsigma]))
        levels.sort()
        kwargs.setdefault('levels', levels)

        df = self.samples_to_df()
        sns.kdeplot(data=df, x='mass', y='cross_section', **kwargs)

    def plot_kde(self, **kwargs):
        kwargs.setdefault('levels', 100)
        kwargs.setdefault('fill', True)
        self.plot_sigma_contours(**kwargs)


class PlotResultScipyKDE(ResultPlot):
    """Deprecated"""
    kde_res: tuple = None

    def set_kde(self, **kwargs):
        kwargs.setdefault('bandwidth', 0.05)
        kwargs = self._prior_to_kwargs(kwargs)

        X, Y = self.samples
        self.kde_res = kde2D(X, Y, **kwargs)

    def plot_kde(self, **kwargs):
        if self.kde_res is None:
            self.set_kde()

        xx, yy, zz = self.kde_res
        plt.pcolormesh(xx, yy, zz, **kwargs)
        plt.colorbar()

    def plot_sigma_contours(self, nsigma=2, **kwargs):
        if self.kde_res is None:
            self.set_kde()
        xx, yy, zz = self.kde_res

        kwargs.setdefault('linewidths', 1)
        kwargs.setdefault('colors', ['black', 'red', 'white'][:nsigma])
        plt.contour(xx, yy, zz,
                    levels=determine_levels(zz, nsigma),
                    **kwargs,
                    )


def pow10(x):
    return 10 ** x


def set_xticks_top(only_lines=False):
    xlim = plt.xlim()

    ax = plt.gca()
    x_ticks = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 1000]
    x_ticks = [x for x in x_ticks if (np.log10(x) <= xlim[1] and np.log10(x) >= xlim[0])]
    for x_tick in x_ticks:
        ax.axvline(np.log10(x_tick), alpha=0.1)
    if only_lines:
        return

    secax = ax.secondary_xaxis('top', functions=(pow10, np.log10))
    secax.set_ticks(x_ticks)
    secax.set_xticklabels([str(x) for x in x_ticks])
    secax.xaxis.set_tick_params(rotation=45)
    secax.set_xlabel(r"$M_{\chi}$ $[GeV/c^{2}]$")


def x_label():
    plt.xlabel(r"$\log_{10}(M_{\chi}$ $[GeV/c^{2}]$)")


def y_label():
    plt.ylabel(r"$\log_{10}(\sigma_{S.I.}$ $[cm^{2}]$)")


def kde2D(x, y, bandwidth,
          xbins=100j,
          ybins=100j,
          range=None,
          **kwargs):
    """Build 2D kernel density estimate (KDE)."""
    # Thanks https://stackoverflow.com/a/41639690!

    if range is None:
        # create grid of sample locations (default: 100x100)
        xx, yy = np.mgrid[x.min():x.max():xbins,
                 y.min():y.max():ybins]
    else:
        xx, yy = np.mgrid[range[0]:range[1]:xbins,
                 range[2]:range[3]:ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)
    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


def determine_levels(histogram_2d: np.array,
                     nsigma=2,
                     ):
    norm = histogram_2d.sum()
    contours = [0.6827, 0.9545, 0.9973][:nsigma]
    levels = []

    # Take histogram bin membership as proportional to Likelihood
    # This is true when data comes from a Markovian process
    def objective(limit, target):
        w = np.where(histogram_2d > limit)
        count = histogram_2d[w]
        return count.sum() - target

    for contour_i in contours:
        target = norm * contour_i
        level_i = scipy.optimize.bisect(objective,
                                        histogram_2d.min(),
                                        histogram_2d.max(),
                                        args=(target,))
        levels.append(level_i)

    levels.reverse()
    levels.append(histogram_2d.max())
    return levels
