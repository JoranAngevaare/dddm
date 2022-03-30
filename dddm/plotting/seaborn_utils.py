""""
Small script to extract the results from seaborn to calculate confidence intervals

I'm sorry for this script, I wanted to have something robust but I couldn't find it anyware.
Seaborn is doing a great job, so let's use it's functionality.

This work is mostly based on:
https://github.com/mwaskom/seaborn/blob/ff0fc76b4b65c7bcc1d2be2244e4ca1a92e4e740/seaborn/distributions.py

"""
from seaborn.distributions import _DistributionPlotter, KDE
from seaborn._decorators import _deprecate_positional_args

from numbers import Number
import math

import numpy as np
import matplotlib.pyplot as plt
import dddm
import warnings

export, __all__ = dddm.exporter()


@_deprecate_positional_args
def kdeplot(x=None, *, y=None, shade=None, vertical=False, kernel=None, bw=None, gridsize=200,
            cut=3, clip=None, legend=True, cumulative=False, shade_lowest=None, cbar=False,
            cbar_ax=None, cbar_kws=None, ax=None, weights=None, hue=None, palette=None,
            hue_order=None, hue_norm=None, multiple="layer", common_norm=True, common_grid=False,
            levels=10, thresh=.05, bw_method="scott", bw_adjust=1, log_scale=None, color=None,
            fill=None, data=None, data2=None, warn_singular=True, **kwargs, ):
    levels = kwargs.pop("n_levels", levels)
    p = _DistributionPlotter(
        data=data,
        variables=_DistributionPlotter.get_semantics(locals()),
    )
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    if ax is None:
        ax = plt.gca()

    p._attach(ax, allowed_types=["numeric", "datetime"], log_scale=log_scale)

    method = ax.fill_between if fill else ax.plot
    color = _default_color(method, hue, color, kwargs)

    # Pack the kwargs for statistics.KDE
    estimate_kws = dict(bw_method=bw_method, bw_adjust=bw_adjust, gridsize=gridsize, cut=cut,
                        clip=clip, cumulative=cumulative, )

    p.plot_bivariate_density(common_norm=common_norm, fill=fill, levels=levels, thresh=thresh,
                             legend=legend, color=color, warn_singular=warn_singular, cbar=cbar,
                             cbar_ax=cbar_ax, cbar_kws=cbar_kws, estimate_kws=estimate_kws,
                             **kwargs, )
    kwargs = dict(common_norm=common_norm, fill=fill, levels=levels, thresh=thresh, legend=legend,
                  color=color, warn_singular=warn_singular, cbar=cbar, cbar_ax=cbar_ax,
                  cbar_kws=cbar_kws, estimate_kws=estimate_kws, **kwargs, )
    return p, kwargs


def get_bivariate(self, common_norm, fill, levels, thresh, color, warn_singular,
                  estimate_kws, **contour_kws, ):
    estimator = KDE(**estimate_kws)

    if not set(self.variables) - {"x", "y"}:
        common_norm = False

    all_data = self.plot_data.dropna()

    # Loop through the subsets and estimate the KDEs
    densities, supports = {}, {}

    for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):

        # Extract the data points from this sub set and remove nulls
        sub_data = sub_data.dropna()
        observations = sub_data[["x", "y"]]

        # Extract the weights for this subset of observations
        weights = sub_data["weights"] if "weights" in self.variables else None
        # Check that KDE will not error out
        variance = observations[["x", "y"]].var()
        if any(math.isclose(x, 0) for x in variance) or variance.isna().any():
            raise ValueError(
                "Dataset has 0 variance; skipping density estimate. "
            )

        # Estimate the density of observations at this level
        observations = observations["x"], observations["y"]
        density, support = estimator(*observations, weights=weights)

        # Transform the support grid back to the original scale
        xx, yy = support
        if self._log_scaled("x"):
            xx = np.power(10, xx)
        if self._log_scaled("y"):
            yy = np.power(10, yy)
        support = xx, yy

        # Apply a scaling factor so that the integral over all subsets is 1
        if common_norm:
            density *= len(sub_data) / len(all_data)

        key = tuple(sub_vars.items())
        densities[key] = density
        supports[key] = support

    # Define a grid of iso-proportion levels
    if thresh is None:
        thresh = 0
    assert isinstance(levels, Number)
    levels = np.linspace(thresh, 1, levels)

    # Transform from iso-proportions to iso-densities
    draw_levels = {
        k: self._quantile_to_level(d, levels)
        for k, d in densities.items()
    }

    # Loop through the subsets again and plot the data
    for sub_vars, _ in self.iter_data("hue"):
        key = tuple(sub_vars.items())
        if key not in densities:
            continue
        density = densities[key]
        xx, yy = supports[key]
        return xx, yy, density, draw_levels, key


def _default_color(*args, **kwargs):
    return 'k'


def _extract_data(x, y, **kwargs):
    p, intermediate_kwargs = kdeplot(x=x, y=y, levels=3, **kwargs)
    x, y, H, levels, levels_keys = get_bivariate(p, **intermediate_kwargs)
    return x, y, H, levels, levels_keys


@export
def one_sigma_area(x, y, clf=True, **kwargs):
    x, y, H, levels, levels_keys = _extract_data(x, y, **kwargs)
    if clf:
        plt.clf()
    bin_area = np.diff(x[:2]) * np.diff(y[:2])
    return (bin_area * np.sum(H > list(levels.values())[0][1]))[0]
