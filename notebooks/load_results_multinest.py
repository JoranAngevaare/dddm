"""Py-file to minimize fundtions in loading notebook."""

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
from common_init import *
print("load_results_multinest.py\tstart")


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 100)


figsize(8, 6)
print("load_results_multinest.py\tdone packages, get results")

##
# Results
##

# loading normal results.
results = {}
all_res = dddm.context.context['results_dir']
res_dirs = os.listdir(all_res)
res_dirs.reverse()
no_result = []
load_errors = []
for i, resdir in enumerate(tqdm(res_dirs)):
    try:
        result = dddm.load_multinest_samples_from_file(
            all_res + '/' + resdir + '/')
    except Exception as e:
        print(f'Error {e} in loading {resdir}')
        load_errors.append([i, all_res + '/' + resdir, e])
        continue
    if result.keys():
        result['dir'] = all_res + '/' + resdir + '/'
        results[i] = result
    else:
        no_result.append(all_res + '/' + resdir + '/')

print("load_results_multinest.py\tdone, convert to results dataframe")


# TODO
# This is not such a nice function to convert the results that were
# loading into a pd-DataFrame
def results_to_df(res):
    """Takes res and converts it to a pd.DataFrame"""
    df = pd.DataFrame()
    items = sorted(list(res.keys()))
    df['item'] = items
    for key in tqdm(res[np.min(list(res.keys()))].keys()):
        if key in ['samples', 'weights', 'weightedsamples']:
            continue
        if key == 'config' or key == 'res_dict':
            for sub_key in res[items[0]][key].keys():
                if sub_key == 'prior':
                    for sub_sub_key in res[items[0]][key][sub_key].keys():
                        if isinstance(res[items[0]][key]
                                      [sub_key][sub_sub_key], dict):
                            for sub_sub_sub_key in res[items[0]
                                                       ][key][sub_key][sub_sub_key].keys():
                                try:
                                    df[key + '_' + sub_key + '_' + sub_sub_key + '_' + sub_sub_sub_key] = [
                                        res[it][key][sub_key][sub_sub_key][sub_sub_sub_key] for it in items]
                                except KeyError:
                                    pass
                        else:
                            df[key + '_' + sub_key + '_' + sub_sub_key] = [res[it]
                                                                           [key][sub_key][sub_sub_key] for it in items]
                else:
                    try:
                        df[key + '_' + sub_key] = [res[it][key][sub_key]
                                                   for it in items]
                    except KeyError:
                        pass

        else:
            try:
                df[key] = [res[it][key] for it in items]
            except KeyError:
                pass

    for essential in ['tol', 'nlive']:
        _res = []
        if not 'config_' + essential in df.keys():
            for it in items:
                try:
                    _res.append(res[it]['config'][essential])
                except (KeyError, IndexError) as e:
                    print(f'{e} for {it}')
                    _res.append(None)
            df['config_' + essential] = _res
    df['mw'] = 10 ** df['config_mw']
    df['n_fit_parameters'] = [len(pars)
                              for pars in df['config_fit_parameters']]
    return df


df = results_to_df(results)
df['exp'] = [det[:2] for det in df['config_detector']]
print("load_results_multinest.py\tdone, open with 'df'")


###
# Helper functions
###
def delete_empty(paths, delete=False, only_old=True):
    """Delete data from a given set of paths. Checks if they are empty and older than
    100 h if only_old == True"""
    for path in tqdm(paths):
        if not delete:
            print(f'delete {path}')
        if os.path.exists(path) and len(os.listdir(path)) == 0 and delete:
            if only_old:
                t_create = datetime.datetime.fromtimestamp(
                    os.path.getmtime(path))
                dt = datetime.datetime.now() - t_create
                if dt > datetime.timedelta(hours=100):
                    shutil.rmtree(path)
            else:
                shutil.rmtree(path)


def delete_with_note(df, note, delete=False):
    """
    :param df: pd.dataframe
    :param note: str, name of config_notes that are to be deleted
    :param delete: if True delete all with this note
    """
    mask = df.config_notes == note
    paths = df[mask]['dir']
    for path in paths:
        cmd = f"rm -rf {path}"
        print(cmd)
        if delete:
            shutil.rmtree(path)


def delete_with_mask(df, mask, delete=False):
    paths = df[mask]['dir']
    for path in tqdm(paths):
        cmd = f"rm -rf {path}"
        print(cmd)
        if delete:
            shutil.rmtree(path)


print(
    "load_results_multinest.py\tIntroduced helperfunctions.\n\tSee delete_empty, "
    "delete_with_note and delete_with_mask")


###
# Plotting
###
# Preferred color maps
colormaps = [
    cm.inferno_r,
    cm.cubehelix,
    cm.ocean,
    cm.plasma_r,
    cm.gnuplot2_r,
    cm.viridis,
    cm.cividis_r,
    cm.brg]


def bin_center(xedges, yedges):
    return 0.5 * (xedges[0:-1] + xedges[1:]), 0.5 * (yedges[0:-1] + yedges[1:])


def get_p_i(i):
    m, sig = results[i]['weighted_samples'].T[:2]
    return np.array([m, sig])


def pow10(x):
    return 10 ** x


def get_info(i):
    """
    Get info of results i to display with plot
    :param i:
    :return:
    """
    info = ""
    for str_inf in [
        'detector',
        'notes',
        'start',
        'fit_time',
        'save_intermediate',
        'earth_shielding',
        'nlive',
        'tol',
            'poisson']:
        try:
            info += f"\n{str_inf} = %s" % results[i]['config'][str_inf]
            if str_inf == 'start':
                info = info[:-7]
            if str_inf == 'fit_time':
                info += 's (%.1f h)' % (results[i]['config'][str_inf] / 3600.)
        except KeyError:
            # We were trying to load something that wasn't saved in the config
            # file, ignore it for now.
            pass
    info += '\n\n--prior--'
    for it, val in results[i]['config']['prior'].items():
        if it == 'k':
            continue
        if val['prior_type'] == 'gauss':
            info += f'\n{it} = {val["mean"]} +/- {val["std"]}'
            info += f' | range = {val["range"]}'
        else:
            info += f'\n{it}_range = {val["range"]}'
    info += '\n\n--fit--'
    for it, val in results[i]['res_dict'].items():
        info += f'\n{it} = {val}'
    if '\n' == info[:1]:
        info = info[1:]
    return info


def get_savename_title(i, save_label=''):
    """
    Wrapper for getting name and title
    :param i:
    :param save_label:
    :return:
    """
    this_df = df[df['item'] == i]
    print(this_df[['item',
                   'mw',
                   'config_sigma',
                   'n_fit_parameters',
                   'config_halo_model',
                   'config_nlive']])
    title = f'$m_w$={this_df["mw"].values[0]}' + ' $Gev/c^{2}$'
    title += rf', $\log(\sigma)$ = {this_df["config_sigma"].values[0]}\n'
    title += f'model ={this_df["config_halo_model"].values[0]}'
    title += f', nfit={this_df["n_fit_parameters"].values[0]}'
    name = str(save_label)
    name += f'mw-{this_df["mw"].values[0]}_s-{this_df["config_sigma"].values[0]}_'
    name += f'{this_df["config_halo_model"].values[0]}_n-{this_df["n_fit_parameters"].values[0]}'
    name += f'_{i}'
    return name, title


def get_binrange(i):
    """
    Get bin range of results i
    :param i:
    :return:
    """
    return results[i]['config']['prior']['log_mass']['range'], results[i]['config']['prior']['log_cross_section'][
        'range']


def one_confidence_plot(
        i,
        save_label='',
        save_dir='figures/',
        corner=False,
        **kwargs):
    """

    :param i:
    :param save_label:
    :param save_dir:
    :param corner:
    :return:
    """
    det = results[i]['config']['detector']
    name, title = get_savename_title(i, save_label)
    res, _ = combined_confidence_plot(i, text_box=str(det), combine=False,
                                      **kwargs
                                      )
    plt.title(f'{title}')
    info = get_info(i)
    ax = plt.gca()
    plt.text(1.3, 1, info, transform=ax.transAxes, fontsize=12, bbox=dict(
        facecolor="white", boxstyle="round"), verticalalignment='top')
    if os.path.exists(save_dir):
        plt.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_dir}/{name}.pdf", dpi=300, bbox_inches="tight")
        if corner:
            dddm.multinest_corner(results[i], save_dir + name)
    else:
        warnings.warn(f"Warning! No {save_dir}")
    return res[0]


def _confidence_plot(item, X, Y, H, bin_range, text_box=False, nsigma=3,
                     make_xticks=True,
                     cbar_note="", cmap=cm.inferno_r, alpha=1,
                     ):
    xmean, xerr = weighted_avg_and_std(X, np.mean(H, axis=0))
    ymean, yerr = weighted_avg_and_std(Y, np.mean(H, axis=1))
    print(f'X mean, std {xmean, xerr}')
    print(f'Y mean, std {ymean, yerr}')
    norm = H.sum()  # Find the norm of the sum
    # Set contour levels
    contour3 = 0.99
    contour2 = 0.95
    contour1 = 0.68

    # Take histogram bin membership as proportional to Likelihood
    # This is true when data comes from a Markovian process
    def objective(limit, target):
        w = np.where(H > limit)
        count = H[w]
        return count.sum() - target

    target1 = norm * contour1
    level1 = scipy.optimize.bisect(
        objective, H.min(), H.max(), args=(target1,))
    levels = [level1]
    # Find levels by summing histogram to objective
    if nsigma > 1:
        target2 = norm * contour2
        level2 = scipy.optimize.bisect(
            objective, H.min(), H.max(), args=(target2,))
        levels.append(level2)
        if nsigma > 2:
            target3 = norm * contour3
            level3 = scipy.optimize.bisect(
                objective, H.min(), H.max(), args=(target3,))
            levels.append(level3)
        if nsigma > 3:
            print('Nsigma too big')

    levels.reverse()
    levels.append(H.max())

    if levels[0] == levels[1]:
        warnings.warn(
            f"Two levels are at the same level. This means you don't have sufficient samples. The levels: {levels}")
        levels[0] /= 1.01
        levels = np.unique(levels)

    contours = plt.contourf(X, Y, H, (0,),
                            levels=levels,
                            alpha=alpha,
                            cmap=cm.get_cmap(cmap, len(levels) - 1),
                            norm=LogNorm()
                            )
    ax = plt.gca()
    cset2 = ax.contour(X, Y, H, contours.levels, colors='k')

    # We don't really need dashed contour lines to indicate negative
    # regions, so let's turn them off.
    for c in cset2.collections:
        c.set_linestyle('solid')

    cbar = ax.figure.colorbar(contours)
    col_labels = [r'$3\sigma$', r'$2\sigma$', r'$1\sigma$'][3 - nsigma:]
    cbar.set_ticklabels(col_labels)
    cbar.set_label("Posterior probability" + cbar_note)

    if make_xticks:
        secax = ax.secondary_xaxis('top', functions=(pow10, np.log10))

        if 'migd' in results[item]['config']['detector']:
            x_ticks = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
        else:
            x_ticks = [15, 25, 50, 100, 250, 500, 1000]
        for x_tick in x_ticks:
            ax.axvline(np.log10(x_tick), alpha=0.1)
        secax.set_ticks(x_ticks)
        secax.set_xticklabels([str(x) for x in x_ticks])
        secax.xaxis.set_tick_params(rotation=45)
        plt.xlim(*bin_range[0])
        plt.ylim(*bin_range[1])
        plt.xlabel(r"$\log_{10}(M_{\chi}$ $[GeV/c^{2}]$)")
        secax.set_xlabel(r"$M_{\chi}$ $[GeV/c^{2}]$")
        plt.ylabel(r"$\log_{10}(\sigma_{S.I.}$ $[cm^{2}]$)")

    if text_box:
        plt.text(0.05, 0.95, text_box, transform=ax.transAxes, alpha=0.5,
                 bbox=dict(facecolor="white", boxstyle="round"))
    return xmean, xerr, ymean, yerr


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


def match_other_item(i, verbose=False, diff_det_type=False,
                     anti_match_for=['config_detector'],
                     ommit_matching=[]):
    # Setup
    this_df = df[df.item == i]
    if len(this_df) == 0:
        warnings.warn(f"WARNING: NO item {i}")
        return None, None
    assert len(this_df) < 2, f'found >1 entries in df for {i}'
    sub_df = df.copy()

    # Remove some keys, they are not interesting for the matching
    match_keys = df.keys()
    if isinstance(ommit_matching, str):
        ommit_matching = [ommit_matching]
    for _k in [
        'item',
        'config_det_params',
        'config_start',
        'config_notes',
        'config_fit_time',
        'config_fit_parameters',
        'global evidence error',
        'global evidence',
        'marginals',
        'modes',
        'nested importance sampling global log-evidence error',
        'nested importance sampling global log-evidence',
        'nested sampling global log-evidence error',
        'nested sampling global log-evidence',
        'res_dict_log_mass_fit_res',
        'res_dict_mass_fit_res',
        'res_dict_log_cross_section_fit_res',
        'res_dict_cross_section_fit_res',
        'res_dict_n_samples',
        'weighted_samples',
        'dir',
            'mw']:
        if _k not in ommit_matching:
            ommit_matching.append(_k)
    if ommit_matching:
        keys_to_unmatch = []
        for key in ommit_matching:
            if key.endswith('*'):
                for _skey in [m for m in match_keys if key.strip('*') in m]:
                    print(f'Starred expr. {key}: unmathcing for {_skey}')
                    keys_to_unmatch.append(_skey)
            else:

                keys_to_unmatch.append(key)

        match_keys = [m for m in match_keys if m not in keys_to_unmatch]

    # look for keys that are different from in the result from the input
    if isinstance(anti_match_for, str):
        anti_match_for = [anti_match_for]
    if anti_match_for:
        keys_to_match = []
        for key in anti_match_for:
            if key.endswith('*'):
                for _skey in [m for m in match_keys if key.strip('*') in m]:
                    print(f'Starred expr. {key}: antimatching for {_skey}')
                    keys_to_match.append(_skey)
            else:
                keys_to_match.append(key)
        anti_match_for = keys_to_match

    # Now let's match, remove items where the key is not the same
    for key in match_keys:
        val = this_df[key].values[0]
        if 'dist' in key:
            continue
        if key in anti_match_for:
            continue
        if verbose:
            print(f'looking for {key}:{val}\t\ttotlen:{len(sub_df)}')
        if np.iterable(val) and len(val) == 2:
            mask = [((_val[0] == val[0]) and (_val[1] == val[1]))
                    for _val in sub_df[key]]
        else:
            mask = sub_df[key] == val
        sub_df = sub_df[mask]
        if len(sub_df) == 0:
            # None left, stop
            break

    # Now let's antimatch, remove items where the key is the same
    if anti_match_for:
        for key in anti_match_for:
            if key not in match_keys:
                if verbose:
                    print(f'looked for {key} in {match_keys}. No such key')
                raise ValueError(
                    f"No such key {key} to match to any of match_keys")
            try:
                val = this_df[key].values[0]
            except TypeError as e:
                print(val, this_df.keys())
                raise e
            if verbose:
                print(f'looking for {key}: not {val}\t\ttotlen:{len(sub_df)}')
            if np.iterable(val) and len(val) == 2:
                mask = np.array([((_val[0] == val[0]) and (_val[1] == val[1])) for _val
                                 in sub_df[key]])
            else:
                mask = sub_df[key] == val
            sub_df = sub_df[~mask]
            if len(sub_df) == 0:
                # None left, stop
                break
    if len(sub_df) == 0:
        warnings.warn("WARNING: NO MATCH")
        return None, None

    # We assume you want a different detector
    for _det in ['Ge', 'Xe']:
        if _det in this_df['config_detector'].values[0]:
            mask = np.array([_det in det for det in sub_df['config_detector']])
            if not diff_det_type:
                sub_df = sub_df[mask]
            else:
                try:
                    sub_df = sub_df[~mask]
                except TypeError:
                    print(sub_df, mask)
                    return sub_df, mask
    if len(sub_df) == 1:
        return sub_df['item'].values, sub_df
    if len(sub_df) == 0:
        warnings.warn("WARNING: NO MATCH")
        return None, None
    warnings.warn(
        "WARNING: MULTIPLE MATHCES\nreturning: items, result DataFrame")
    return sub_df.item.values, sub_df


def find_single_other(i, **kwargs):
    _res, _df = match_other_item(i, **kwargs)
    if np.iterable(_res) and len(_res) == 0:
        return _res[0]
    elif np.iterable(_res):
        warnings.warn(f'Multiple res for it {i}. See its: {_res}')
        return _res[0]
    else:
        warnings.warn(f'No res for it {i}. See its: {_res}')
        return None


def find_other_ge(
        i,
        det_order=(
            'Ge_migd_HV_Si_bg',
            'Ge_migd_HV_bg',
            'Ge_migd_iZIP_Si_bg',
            'Ge_migd_iZIP_bg'),
        **kwargs):
    assert np.sum(df[df['item'] == i]['config_detector'] == det_order[0])
    # First match
    _, _df = match_other_item(i, **kwargs)
    if not np.iterable(_df) and not _df:
        return np.full(3, np.nan)
    # Remove xenon
    _df = _df[_df['exp'] == 'Ge']
    # Extract one item per sub-detector type
    _its = []
    for exp in np.unique(det_order[1:]):
        _mask = _df['config_detector'] == exp
        if not np.sum(_mask):
            _its.append(None)
        else:
            _is = _df[_mask].item.values
            if len(_is) > 1:
                warnings.warn(f'Multiple res for {exp}. See its: {_is}')
            _its.append(_is[0])
    return _its


def combined_confidence_plot(items,
                             combine=True,
                             bin_range=None,
                             nsigma=2,
                             smoothing=None,
                             cbar_notes=None,
                             text_box=False,
                             cmap=None,
                             show_both=False,
                             alpha=0.5,
                             nbins=50):
    if not np.iterable(items):
        items = [items]
    if bin_range is None:
        bin_range = [results[items[0]]['config']['prior']['log_mass']['range'],
                     results[items[0]]['config']['prior']['log_cross_section']['range']]
    # Note that we need one note for the combined plot as well
    if cbar_notes is None:
        cbar_notes = np.repeat('', len(items) + int(combine))
    assert len(cbar_notes) == len(
        items) + int(combine), f"lengths do no match {len(cbar_notes) , len(items) + int(combine)}"

    hists = []
    _results = []
    for k, item in enumerate(items):
        if item is None:
            warnings.warn(f'No item for {cbar_notes[k]}')
            continue
        x, y = get_p_i(item)
        # Make a 2d normed histogram
        _hist, xedges, yedges = np.histogram2d(
            x, y, bins=nbins, range=bin_range, normed=True)
        X, Y = bin_center(xedges, yedges)
        _hist = _hist.T
        if smoothing:
            _hist = scipy.ndimage.filters.gaussian_filter(
                _hist,
                [np.sqrt(nbins) / 10, np.sqrt(nbins) / 10],
                mode='constant')

        # Single plot
        if not combine or len(items) == 1 or show_both:
            res = _confidence_plot(
                item,
                X,
                Y,
                _hist,
                bin_range,
                text_box=text_box,
                nsigma=nsigma,
                cbar_note=cbar_notes[k],
                cmap=cmap[k] if cmap else colormaps[k],
                make_xticks=False,
                alpha=alpha /
                len(items) if combine else alpha)
            _results.append(res)
        hists.append(_hist)
        combined_hist = None if len(items) > 1 else (X, Y, _hist)
    # Combined plot
    if combine and len(items) >= 2:
        H = 1
        for _h in hists:
            H *= _h
        H = H / np.sum(H)
        # pylint: disable=undefined-loop-variable
        res = _confidence_plot(item, X, Y, H, bin_range,
                               text_box=text_box,
                               nsigma=nsigma,
                               cbar_note=cbar_notes[k + 1],
                               cmap=cmap if cmap else colormaps[k + 1],
                               alpha=0.8)
        _results.append(res)
        combined_hist = (X, Y, H)
    return _results, combined_hist


def save_canvas(name, save_dir='./figures', tight_layout=False):
    dddm.check_folder_for_file(save_dir + '/.')
    dddm.check_folder_for_file(save_dir + '/pdf/.')
    if tight_layout:
        plt.tight_layout()
    if os.path.exists(save_dir) and os.path.exists(save_dir + '/pdf'):
        plt.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_dir}/pdf/{name}.pdf", dpi=300, bbox_inches="tight")
    else:
        raise FileExistsError(
            f'{save_dir} does not exist or does not have /pdf')


def plot_prior_range(res, it, sigma_dist=[5, 10]):
    assert len(sigma_dist) == 2
    xmean, xerr, ymean, yerr = res
    br = get_binrange(it)
    for _br in br[0]:
        plt.axvspan(_br, 2 * _br - xmean, alpha=0.3, color='r')
    for _br in br[1]:
        plt.axhspan(_br, 2 * _br - ymean,
                    xmin=br[0][0],
                    xmax=br[0][1],
                    alpha=0.3, color='r')
    lims = [
        np.clip(
            xmean + sigma_dist[0] * np.array([-xerr, xerr]),
            *(br[0] + np.array([-0.25, 0.25]))
        ),
        np.clip(
            ymean + sigma_dist[1] * np.array([-yerr, yerr]),
            *(br[1] + np.array([-0.5, 0.5])))
    ]

    plt.xlim(*lims[0])
    plt.ylim(*lims[1])


def plot_fit_res(res, c='r', label='best fit'):
    xmean, xerr, ymean, yerr = res
    plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr,
                 c=c, capsize=3, label=label, marker='o',
                 linestyle='None', zorder=900)


def plot_bench(it, c='cyan', label='benchmark value'):
    plt.scatter(results[it]['config']['mw'],
                results[it]['config']['sigma'],
                s=10**2,
                edgecolors='black',
                c=c, marker='X', label=label, zorder=1000)


def exec_show(show=True):
    if show:
        plt.show()
    else:
        plt.clf()


def def_show_single(it, this_name, name_base, show=True, **kwargs):
    kwargs['alpha'] = 1
    res = one_confidence_plot(
        it,
        corner=False,
        save_dir='figures/misc/',
        **kwargs)
    plot_prior_range(res, it)
    plot_fit_res(res)
    plot_bench(it)

    plt.grid(axis='y')
    plt.legend(loc='upper right')
    name = this_name + name_base
    save_canvas(name, save_dir=f'figures/{name_base}/')
    exec_show(show)

    name = this_name + 'corner' + name_base
    dddm.multinest_corner(results[it])
    save_canvas(name, save_dir=f'figures/{name_base}/')


def get_color(val, _range=[0, 1], it=0):
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


def pop_from_list(l, idx, default="placeholder"):
    """
    Get an item from a list in a save way. It it fails to extract index, return default
    :param l: list
    :param idx: index (or list of indexes if list is nested
    :param default: string that is returned if no item can be found on requested place in
    the list
    :return: list[idx] or list[idx[0]][idx[1][...]
    """
    if np.iterable(idx) and len(idx) > 1:
        try:
            return pop_from_list(l[idx[0]], idx[1:])
        except (IndexError, TypeError):
            return default
    else:
        if np.iterable(idx) and len(idx) == 1:
            idx = idx[0]
        try:
            return l[idx]
        except (IndexError, TypeError):
            return default


def combine_sets(items,
                 level=0,
                 name_base='test',
                 verbose=False,
                 show=True,
                 notes=None,  # should be of as in the docstring
                 **combined_kwargs):
    """

    :param items: set of sets, e.g.
        [
        [[Xe_0, Xe_1],[Ge_0, Ge_1]],
        ...
        ]
    :param level:
        0: Overlay [Xe_0, Xe_1, Ge_0, Ge_1]
        1: Combine to [Xe_avg, Ge_avg]
        2: Combine deep to [Xe&Ge_avg]
    :param notes:  list of notes for each of the levels. E.g.
        notes = [
        [['Xe_0', 'Xe_1'],['Ge_0', 'Ge_1']],  # level 0
        [['Xe'], ['Ge']],  # level 1
        ['Xe', 'Ge', 'Xe + Ge' ],  # level 2
        ]
    :param name_base: The category of plots to be produced
    :param verbose: Print intermediate messages
    :param show: show figures (otherwise only saved if False)
    :param combined_kwargs: arguments for the plots to be made. See combined_confidence_plot
    :return:
    """
    mode = {0: 'overlay',
            1: 'combine',
            2: 'deep_combine'}[level]
    if notes is None:
        notes = [
            [[None], [None]],  # level 0
            [],  # Level 1
            []  # level 2
        ]

    def f_print(string):
        if verbose:
            print('combine_sets::\t' + string)

    f_print(f'Combine items. Mode {mode}')
    if os.path.exists(f'figures/{name_base}'):
        f_print(f'remove old figures/{name_base}')
        shutil.rmtree(f'figures/{name_base}')
    assert len(notes) >= level + \
        1, f'level {level} requires >= {level} cbar-notes'
    for set_number, sets in enumerate(tqdm(items)):
        f_print(f'Starting with set {set_number}')
        f_print(f'at 0/{level}')
        combined_hists = []
        # with sets = [[Xe_0, Xe_1],[Ge_0, Ge_1]]
        for sub_number, sub_set in enumerate(sets):
            f_print(f'doing {sub_set}')
            # with sub_set = [Xe_0, Xe_1]
            for number_i, it in enumerate(sub_set):
                f_print(f'doing {it}')
                if pd.isnull(it):
                    continue
                # with it = [Xe_0]
                _name = f'set{set_number}_level0_sub{sub_number}.{number_i}'
                f_print(f'Plotting {_name}')
                kwargs = combined_kwargs.copy()
                kwargs['cbar_notes'] = [pop_from_list(
                    notes, [0, sub_number, number_i])]
                def_show_single(it, _name, name_base, **kwargs)
                exec_show(show)

        if level > 0:
            f_print(f'at 1/{level}')
            for plot_times in range(2):
                # plot_times == 0 -> plot all sub_sets one canvas each [Xe_1, Xe_2, Xe_com]
                # plot_time == 1 -> plot all combined sub_set on one canvas
                # [Xe_com, Ge_com]
                for sub_number, sub_set in enumerate(sets):
                    if np.any(pd.isnull(sub_set)):
                        continue
                    # with sub_set = [Xe_0, Xe_1]
                    kwargs = combined_kwargs.copy()
                    _labels = [pop_from_list(notes, [1, sub_number, i]) for i in
                               range(len(sub_set) + int(len(sub_set) > 1))]
                    kwargs['cbar_notes'] = _labels

                    if plot_times == 0:
                        plt.figure(figsize(9 + 1.5 * len(sub_set), 6))
                        kwargs['nsigma'] = min(kwargs['nsigma'], 2)
                    else:
                        if sub_number == 0:
                            plt.figure(figsize(9 + len(sets), 6))
                        kwargs['cmap'] = colormaps[sub_number]
                        kwargs['alpha'] = 0.8

                    res, _hist = combined_confidence_plot(
                        sub_set, combine=len(sub_set) > 1, show_both=plot_times == 0, **kwargs)

                    plot_prior_range(res[-1], sub_set[0])

                    # For each of the items in the sub_set plot it's best fit
                    for j, _res in enumerate(res):
                        _c = get_color(j, _range=len(sub_set), it=sub_number)
                        label = 'best fit' if j == len(
                            res) - 1 else 'best sub-fit'
                        if _labels:
                            if plot_times == 0:
                                label += pop_from_list(_labels,
                                                       j, f'\nholder{j}')
                            else:
                                label += _labels[-1]
                        plot_fit_res(_res, c=_c, label=label)
                    if plot_times == 0:
                        combined_hists.append([res[-1], _hist])
                        plot_bench(sub_set[0])
                        plot_prior_range(res[-1], sub_set[0])
                        plt.grid(axis='y')
                        plt.legend(loc='lower left')
                        name = f'set{set_number}_level1_sub{sub_number}_' + name_base
                        save_canvas(name, save_dir=f'figures/{name_base}/')
                        exec_show(show)
                        f_print(f'Plotting {name}')
            if sets and len(sets[0]):
                plot_bench(sets[0][0])
                plt.grid(axis='y')
                plt.legend(loc='upper right')
                name = f'set{set_number}_level1_' + name_base
                save_canvas(name, save_dir=f'figures/{name_base}/')
                exec_show(show)
                f_print(f'Plotting {name}')
        if level == 2 and len(combined_hists) == len(sets):
            plt.figure(figsize(10, 6))
            combined_hists = []
            f_print(f'at 2/{level}')
            warnings.warn(
                f'made it to {level} however this means we are smoothing twice, this might be tricky')
            prod = 1
            for sub_number, sub_set in enumerate(sets):
                kwargs = combined_kwargs.copy()
                kwargs['cbar_notes'] = [pop_from_list(notes, [2, i]) for i in range(
                    len(sub_set) + int(len(sub_set) > 1))]
                kwargs['cmap'] = colormaps[sub_number]
                kwargs['nsigma'] = 2
                # with sub_set = [Xe_0, Xe_1]
                res, _hist = combined_confidence_plot(sub_set,
                                                      combine=len(sub_set) > 1,
                                                      **kwargs)
                combined_hists.append([res[-1], _hist])
                _c = get_color(sub_number, len(sets))
                label = pop_from_list(notes, [2, sub_number])
                plot_fit_res(res[-1], c=_c, label='best fit' + label)
            for i, (_res, hist) in enumerate(combined_hists):
                X, Y, H = hist
                prod *= H
            first_item = sets[0][0]
            # pylint: disable=undefined-loop-variable
            _note = pop_from_list(notes, [2, i + 1], 'combination')
            res = _confidence_plot(first_item,
                                   X, Y, prod,
                                   bin_range=get_binrange(first_item),
                                   nsigma=3,
                                   cbar_note=_note,
                                   cmap=colormaps[i + 1],
                                   alpha=1)
            plot_bench(first_item, c='cyan')
            plot_prior_range(res, first_item)
            plot_fit_res(res, c='green', label='combined fit' + _note)
            plt.grid(axis='y')
            plt.legend(loc='upper right')
            name = f'set{set_number}' + name_base
            f_print(f'plotting onto {name}')
            save_canvas(name, save_dir=f'figures/{name_base}/')
            exec_show(show)
