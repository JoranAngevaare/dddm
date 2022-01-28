"""Some basic functions for plotting et cetera. Used to for instance to
check that the likelihood function is well behaved"""
# disable bandit
import pickle
import colorsys
import os

import matplotlib.pyplot as plt
import numpy as np
import dddm  # from dddm import statistics, halo, detector, utils
from tqdm import tqdm


def error_bar_hist(ax, data, data_range=None, nbins=50, **kwargs):
    x, y, yerr = hist_data(data, data_range, nbins)
    ax.errorbar(x, y, yerr=yerr, capsize=3, marker='o', **kwargs)


def hist_data(data, data_range=None, nbins=50):
    if data_range is None:
        data_range = [np.min(data), np.max(data)]
    else:
        assert_str = "make sure data_range is of fmt [x_min, x_max]"
        assert isinstance(data_range, (list, tuple)) and len(
            data_range) == 2, assert_str

    counts, _bin_edges = np.histogram(data, range=data_range, bins=nbins)
    bin_centers = (_bin_edges[:-1] + _bin_edges[1:]) / 2
    x, y, yerr = bin_centers, counts, np.sqrt(counts)
    return x, y, yerr


def simple_hist(y: np.ndarray):
    plt.figure(figsize=(19, 6))
    ax = plt.gca()
    data_all = hist_data(y)
    ax.plot(data_all[0], data_all[1], drawstyle='steps-mid',
            label="Pass through")
    error_bar_hist(ax, y)


def ll_element_wise(x, y, clip_val=-1e4):
    rows = len(x)
    cols = len(x[0])
    r = np.zeros((rows, cols))
    for i in tqdm(range(rows)):
        for j in range(cols):
            r[i][j] = dddm.statistics.log_likelihood_function(x[i][j], y[i][j])
    return np.clip(r, clip_val, 0)


def show_ll_function(npoints=1e4, clip_val=-1e4, min_val=0.1):
    from pylab import meshgrid, cm, imshow, colorbar, title
    from matplotlib.colors import LogNorm
    x = np.arange(0, npoints, 1)
    y = np.arange(0, npoints, 1)
    X, Y = meshgrid(x, y)  # grid of point
    Z = -ll_element_wise(X, Y, clip_val)
    im = imshow(Z, cmap=cm.RdBu,
                norm=LogNorm(min_val, -clip_val))  # drawing the function
    colorbar(im, label=r'$-\mathcal{L}$')  # adding the colorbar on the right
    title(r'$-\mathcal{L}$ clipped at %i' % clip_val)
    plt.xlabel("Nb")
    plt.ylabel("Nr")


def plt_ll_sigma_mass(spec_clas, vary, det_class=dddm.examples.XenonSimple, bins=10, m=50,
                      sig=1e-45):
    assert vary in ['mass', 'sig'], "use sig or mass"
    use_SHM = dddm.SHM()
    det = det_class(n_energy_bins=bins)
    events = spec_clas(dark_matter_model=use_SHM, experiment=det)
    data = events.get_data(m, sig, poisson=False)

    if vary == 'sig':
        plt.xlabel(r'$\sigma$ $[cm^2]$')
        plt.axvline(sig, alpha=0.5, color='red', label='truth')
        var = np.linspace(0.1 * sig, 10 * sig, 30)

        def model(x):
            return events.get_counts(m, x, poisson=False)

    elif vary == 'mass':
        plt.xlabel('mass [GeV/$c^2$]')
        plt.axvline(m, alpha=0.5, color='red', label='truth')
        var = np.linspace(m / 10, m * 10, 50)

        def model(x):
            return events.get_counts(x, sig, poisson=False)
    else:
        raise ValueError(f'Can not vary {vary}')
    plr = [dddm.statistics.log_likelihood(data['counts'], model(x)) for x in
           tqdm(var)]

    plt.xlim(var[0], var[-1])
    var, plr = dddm.utils.remove_nan(var, plr), dddm.utils.remove_nan(plr, var)
    plt.plot(var, plr, drawstyle='steps-mid')
    plt.ylim(np.min(plr), np.max(plr))


def plt_ll_sigma_spec(det_class=dddm.examples.XenonSimple, bins=10, m=50, sig=1e-45):
    spec = dddm.GenSpectrum
    plt_ll_sigma_mass(spec, 'sig', det_class=det_class, bins=bins, m=m, sig=sig)


def plt_ll_mass_spec(det_class=dddm.examples.XenonSimple, bins=10, m=50, sig=1e-45):
    spec = dddm.GenSpectrum
    plt_ll_sigma_mass(spec, 'mass', det_class=det_class, bins=bins, m=m, sig=sig)


def plt_ll_sigma_det(det_class=dddm.examples.XenonSimple, bins=10, m=50, sig=1e-45):
    spec = dddm.DetectorSpectrum
    plt_ll_sigma_mass(spec, 'sig', det_class=det_class, bins=bins, m=m, sig=sig)


def plt_ll_mass_det(det_class=dddm.examples.XenonSimple, bins=10, m=50, sig=1e-45):
    spec = dddm.DetectorSpectrum
    plt_ll_sigma_mass(spec, 'mass', det_class=det_class, bins=bins, m=m, sig=sig)


def plt_priors(itot=100):
    priors = dddm.get_priors()
    for key in priors.keys():
        par = priors[key]['param']
        dist = priors[key]['dist']
        data = [dist(par) for _ in range(itot)]
        if priors[key]['prior_type'] == 'gauss':
            plt.axvline(priors[key]['mean'], c='r')
            plt.axvline(priors[key]['mean'] - priors[key]['std'], c='b')
            plt.axvline(priors[key]['mean'] + priors[key]['std'], c='b')
        plt.hist(data, bins=100)
        plt.title(key)


def plot_spectrum(data, color='blue', label='label', linestyle='none',
                  plot_error=True):
    plt.errorbar(data['bin_centers'], data['counts'],
                 xerr=(data['bin_left'] - data['bin_right']) / 2,
                 yerr=np.sqrt(data['counts']) if plot_error else np.zeros(
                     len(data['counts'])),
                 color=color,
                 linestyle=linestyle,
                 capsize=2,
                 marker='o',
                 label=label,
                 markersize=2
                 )


def get_color_from_range(val, _range=(0, 1), it=0):
    if not np.iterable(_range):
        _range = [0, _range]
    red_to_green = (val - _range[0]) / np.diff(_range)
    assert 0 <= red_to_green <= 1, f'{val} vs {_range} does not work'
    assert it <= 2
    # in HSV, red is 0 deg and green is 120 deg (out of 360);
    # divide red_to_green with 3 to map [0, 1] to [0, 1./3.]
    hue = red_to_green / 3.0
    hue += it / 3
    res = colorsys.hsv_to_rgb(hue, 1, 1)
    # return [int(255 * float(r)) for r in res]
    return [float(r) for r in res]


def save_canvas(name,
                save_dir='./figures',
                dpi=200,
                tight_layout=False,
                pickle_dump=True):
    """Wrapper for saving current figure"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/.')
    for sub_folder in 'pdf pkl svg'.split():
        sub_dir = os.path.join(save_dir, sub_folder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    if tight_layout:
        plt.tight_layout()
    if pickle_dump:
        pickle_dump_figure(os.path.join(save_dir, 'pkl', f'{name}.pkl'))
    if os.path.exists(save_dir):
        plt.savefig(f"{save_dir}/{name}.png", dpi=dpi, bbox_inches="tight")
        for extension in 'pdf svg'.split():
            plt.savefig(
                os.path.join(
                    save_dir,
                    extension,
                    f'{name}.{extension}'),
                dpi=dpi,
                bbox_inches="tight")
    else:
        raise FileExistsError(f'{save_dir} does not exist or does not have /pdf')


def pickle_dump_figure(name):
    fig = plt.gcf()
    pickle.dump(fig, open(name, 'wb'))


def open_pickle_figure(name):
    # disable bandit
    return pickle.load(open(name, 'rb'))
