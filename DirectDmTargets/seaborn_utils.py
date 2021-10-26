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
import warnings

import numpy as np
import matplotlib.pyplot as plt


def _default_color(*args, **kwargs):
    return 'k'


@_deprecate_positional_args
def kdeplot(
        x=None,  # Allow positional x, because behavior will not change with reorg
        *,
        y=None,
        shade=None,  # Note "soft" deprecation, explained below
        vertical=False,  # Deprecated
        kernel=None,  # Deprecated
        bw=None,  # Deprecated
        gridsize=200,  # TODO maybe depend on uni/bivariate?
        cut=3, clip=None, legend=True, cumulative=False,
        shade_lowest=None,  # Deprecated, controlled with levels now
        cbar=False, cbar_ax=None, cbar_kws=None,
        ax=None,

        # New params
        weights=None,  # TODO note that weights is grouped with semantics
        hue=None,
        palette=None,
        hue_order=None,
        hue_norm=None,
        multiple="layer",
        common_norm=True,
        common_grid=False,
        levels=10,
        thresh=.05,
        bw_method="scott",
        bw_adjust=1,
        log_scale=None,
        color=None,
        fill=None,

        # Renamed params
        data=None,
        data2=None,

        # New in v0.12
        warn_singular=True,

        **kwargs,
):
    # Handle deprecation of `data2` as name for y variable
    if data2 is not None:

        y = data2

        # If `data2` is present, we need to check for the `data` kwarg being
        # used to pass a vector for `x`. We'll reassign the vectors and warn.
        # We need this check because just passing a vector to `data` is now
        # technically valid.

        x_passed_as_data = (
            x is None
            and data is not None
            and np.ndim(data) == 1
        )

        if x_passed_as_data:
            msg = "Use `x` and `y` rather than `data` `and `data2`"
            x = data
        else:
            msg = "The `data2` param is now named `y`; please update your code"

        warnings.warn(msg, FutureWarning)

    # Handle deprecation of `vertical`
    if vertical:
        msg = (
            "The `vertical` parameter is deprecated and will be removed in a "
            "future version. Assign the data to the `y` variable instead."
        )
        warnings.warn(msg, FutureWarning)
        x, y = y, x

    # Handle deprecation of `bw`
    if bw is not None:
        msg = (
            "The `bw` parameter is deprecated in favor of `bw_method` and "
            f"`bw_adjust`. Using {bw} for `bw_method`, but please "
            "see the docs for the new parameters and update your code."
        )
        warnings.warn(msg, FutureWarning)
        bw_method = bw

    # Handle deprecation of `kernel`
    if kernel is not None:
        msg = (
            "Support for alternate kernels has been removed. "
            "Using Gaussian kernel."
        )
        warnings.warn(msg, UserWarning)

    # Handle deprecation of shade_lowest
    if shade_lowest is not None:
        if shade_lowest:
            thresh = 0
        msg = (
            "`shade_lowest` is now deprecated in favor of `thresh`. "
            f"Setting `thresh={thresh}`, but please update your code."
        )
        warnings.warn(msg, UserWarning)

    # Handle `n_levels`
    # This was never in the formal API but it was processed, and appeared in an
    # example. We can treat as an alias for `levels` now and deprecate later.
    levels = kwargs.pop("n_levels", levels)

    # Handle "soft" deprecation of shade `shade` is not really the right
    # terminology here, but unlike some of the other deprecated parameters it
    # is probably very commonly used and much hard to remove. This is therefore
    # going to be a longer process where, first, `fill` will be introduced and
    # be used throughout the documentation. In 0.12, when kwarg-only
    # enforcement hits, we can remove the shade/shade_lowest out of the
    # function signature all together and pull them out of the kwargs. Then we
    # can actually fire a FutureWarning, and eventually remove.
    if shade is not None:
        fill = shade

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

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

    if not p.has_xy_data:
        return ax

    # Pack the kwargs for statistics.KDE
    estimate_kws = dict(
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
        cumulative=cumulative,
    )

    if p.univariate:
        raise NotImplementedError
        plot_kws = kwargs.copy()

        p.plot_univariate_density(
            multiple=multiple,
            common_norm=common_norm,
            common_grid=common_grid,
            fill=fill,
            color=color,
            legend=legend,
            warn_singular=warn_singular,
            estimate_kws=estimate_kws,
            **plot_kws,
        )

    else:

        p.plot_bivariate_density(
            common_norm=common_norm,
            fill=fill,
            levels=levels,
            thresh=thresh,
            legend=legend,
            color=color,
            warn_singular=warn_singular,
            cbar=cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            estimate_kws=estimate_kws,
            **kwargs,
        )
    kwargs = dict(
        common_norm=common_norm,
        fill=fill,
        levels=levels,
        thresh=thresh,
        legend=legend,
        color=color,
        warn_singular=warn_singular,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        estimate_kws=estimate_kws,
        **kwargs,
    )
    return p, kwargs


def get_bivariate(self,
                  common_norm,
                  fill,
                  levels,
                  thresh,
                  color,
                  legend,
                  cbar,
                  warn_singular,
                  cbar_ax,
                  cbar_kws,
                  estimate_kws,
                  **contour_kws, ):
    contour_kws = contour_kws.copy()

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
        if "weights" in self.variables:
            weights = sub_data["weights"]
        else:
            weights = None

        # Check that KDE will not error out
        variance = observations[["x", "y"]].var()
        if any(math.isclose(x, 0) for x in variance) or variance.isna().any():
            msg = (
                "Dataset has 0 variance; skipping density estimate. "
                "Pass `warn_singular=False` to disable this warning."
            )
            if warn_singular:
                warnings.warn(msg, UserWarning)
            continue

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
    if isinstance(levels, Number):
        levels = np.linspace(thresh, 1, levels)
    else:
        if min(levels) < 0 or max(levels) > 1:
            raise ValueError("levels must be in [0, 1]")

    # Transform from iso-proportions to iso-densities
    if common_norm:
        common_levels = self._quantile_to_level(
            list(densities.values()), levels,
        )
        draw_levels = {k: common_levels for k in densities}
    else:
        draw_levels = {
            k: self._quantile_to_level(d, levels)
            for k, d in densities.items()
        }

    # Get a default single color from the attribute cycle
    if self.ax is None:
        default_color = "C0" if color is None else color
    else:
        scout, = self.ax.plot([], color=color)
        default_color = scout.get_color()
        scout.remove()

    # Define the coloring of the contours
    if "hue" in self.variables:
        for param in ["cmap", "colors"]:
            if param in contour_kws:
                msg = f"{param} parameter ignored when using hue mapping."
                warnings.warn(msg, UserWarning)
                contour_kws.pop(param)
    else:

        # Work out a default coloring of the contours
        coloring_given = set(contour_kws) & {"cmap", "colors"}
        if fill and not coloring_given:
            cmap = self._cmap_from_color(default_color)
            contour_kws["cmap"] = cmap
        if not fill and not coloring_given:
            contour_kws["colors"] = [default_color]

        # Use our internal colormap lookup
        cmap = contour_kws.pop("cmap", None)
        if isinstance(cmap, str):
            cmap = color_palette(cmap, as_cmap=True)
        if cmap is not None:
            contour_kws["cmap"] = cmap

    # Loop through the subsets again and plot the data
    for sub_vars, _ in self.iter_data("hue"):

        if "hue" in sub_vars:
            color = self._hue_map(sub_vars["hue"])
            if fill:
                contour_kws["cmap"] = self._cmap_from_color(color)
            else:
                contour_kws["colors"] = [color]

        ax = self._get_axes(sub_vars)

        # Choose the function to plot with
        # TODO could add a pcolormesh based option as well
        # Which would look something like element="raster"
        if fill:
            contour_func = ax.contourf
        else:
            contour_func = ax.contour

        key = tuple(sub_vars.items())
        if key not in densities:
            continue
        density = densities[key]
        xx, yy = supports[key]

        label = contour_kws.pop("label", None)

        cset = contour_func(
            xx, yy, density,
            levels=draw_levels[key],
            **contour_kws,
        )
        return xx, yy, density, draw_levels, key


def extract_data(x, y, **kwargs):
    p, intermediate_kwargs = kdeplot(x=x, y=y, levels=3, **kwargs)
    x, y, H, levels, levels_keys = get_bivariate(p, **intermediate_kwargs)
    return x, y, H, levels, levels_keys


def one_sigma_area(x, y, **kwargs):
    x, y, H, levels, levels_keys = extract_data(x, y, **kwargs)
    plt.imshow(H > list(levels.values())[0][1], extent=[x[0], x[-1], y[0], y[-1]])
    bin_area = np.diff(x[:2]) * np.diff(y[:2])
    return bin_area * np.sum(H > list(levels.values())[0][1])
