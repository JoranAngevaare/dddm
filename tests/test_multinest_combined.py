import logging

import dddm
import os
from .test_multinest_shielded import _is_windows
import matplotlib.pyplot as plt
from tqdm import tqdm
log = logging.getLogger()


def test_combined_multinest(targets=('Xe_simple', 'Ge_simple'),):
    if _is_windows():
        return
    dddm.experiment['Combined'] = {'type': 'combined'}
    stats = dddm.CombinedInference(
        targets,
        'Combined',
    )
    update = {'prior': dddm.statistics.get_priors("Evans_2019"),
              'halo_model': dddm.SHM(),
              'type': 'SI'}
    stats.config.update(update)
    stats.copy_config(list(update.keys()))
    stats.config['tol'] = 0.5
    stats.config['nlive'] = 100
    print(f"Fitting for parameters:\n{stats.config['fit_parameters']}")
    stats.run_multinest()
    stats.save_results()
    stats.save_sub_configs()

    print('opening results')
    result_path = os.path.join(dddm.context.context['results_dir'], 'nes_mu*')
    print(os.listdir(dddm.context.context['results_dir']))
    results = dddm.ResultsManager(result_path)
    print(results)
    results.apply_mask(results.df['nlive'] > 1)
    assert results.result_cache is not None and len(results.result_cache) > 0

    if len(results.result_cache) > 30:
        raise RuntimeError(f'Too many matches for {result_path}')

    for res in tqdm(results.result_cache, desc='opening results folder'):
        print(res)
        plot = dddm.SeabornPlot(res)
        print(plot)
        plot.plot_kde(bw_adjust=0.75, alpha=0.7)
        plot.plot_sigma_contours(nsigma=3,
                                 bw_adjust=0.75,
                                 color='k',
                                 linewidths=2,
                                 linestyles=['solid', 'dashed', 'dotted'][::-1]
                                 )
        plot.plot_samples(alpha=0.2)
        plot.plot_best_fit()
        plot.plot_bench()
        plt.text(.63, 0.93,
                 'TEST',
                 transform=plt.gca().transAxes,
                 bbox=dict(alpha=0.5,
                           facecolor='gainsboro',
                           boxstyle="round",
                           ),
                 va='top',
                 ha='left',
                 )

        dddm.y_label()
        dddm.x_label()
        dddm.set_xticks_top()
        plt.grid()
        plt.clf()
        plt.close()
    try:
        results.add_result('no_such_file')
    except AssertionError:
        pass
    else:
        raise RuntimeError('No error raised')
    results._add_result('no_such_file', tolerant=True)


def test_combined_multinest_single_target():
    test_combined_multinest(targets=('Xe',))
