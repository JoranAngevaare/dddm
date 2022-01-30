from unittest import skipIf
from unittest import TestCase
import dddm
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt


@skipIf(not dddm.utils.is_installed('pymultinest'), 'pymultinest is not installed')
@skipIf(dddm.utils.is_windows(), "Multinest only works on linux")
class PymultinestTest(TestCase):
    def setUp(self) -> None:
        self.ct = dddm.test_context()

    def test_multinest_shielded_full_astrophysics(self, ):
        self.test_multinest(halo_name='shielded_shm',
                            sampler_kwargs=dict(nlive=10, tol=0.9, verbose=0),
                            fit_parameters=dddm.statistics.get_param_list())

    def test_multinest_shielded(self, ):
        self.test_multinest(halo_name='shielded_shm',
                            sampler_kwargs=dict(nlive=10, tol=0.9, verbose=0),
                            fit_parameters=dddm.statistics.get_param_list(),
                            )

    def test_multinest(self, halo_name='shm', **kwargs):
        base_config = dict(wimp_mass=50,
                           cross_section=1e-45,
                           sampler_name='multinest',
                           detector_name='Xe_simple',
                           prior="Pato_2010",
                           halo_name=halo_name,
                           detector_kwargs=None,
                           halo_kwargs=None if halo_name == 'shm' else dict(location='XENON'),
                           sampler_kwargs=dict(nlive=100, tol=0.1, verbose=0),
                           fit_parameters=('log_mass', 'log_cross_section',),
                           )
        config = {**base_config, **kwargs}
        sampler = self.ct.get_sampler_for_detector(**config)

        sampler.run()
        results = sampler.get_summary()
        fails = []
        for i, (thing, expected, avg) in enumerate(
                zip(
                    base_config.get('fit_parameters'),
                    [getattr(sampler, f) for f in base_config.get('fit_parameters')],
                    results['best_fit']
                )):
            std = np.sqrt(results['cov_matrix'][i][i])
            nsigma_off = np.abs(expected - avg) / std
            message = (f'For {thing}: expected {expected:.2f} yielded '
                       f'different results {avg:.2f} +/- {std:.2f}. Off '
                       f'by {nsigma_off:.1f} sigma')
            if nsigma_off > 4:
                fails += [message]
            print(message)

        self.assertFalse(fails, fails)
        return sampler

    def test_chain(self,
                   halo_name='shm',
                   fit_parameters=('log_mass', 'log_cross_section',).
                   ):
        sampler = self.test_multinest( detector_name=['Xe_simple', 'Ar_simple', 'Ge_simple'],
                                       prior="Pato_2010",
                                       halo_name=halo_name,
                                       detector_kwargs=None,
                                       halo_kwargs=None if halo_name == 'shm' else dict(location='XENON'),
                                       sampler_kwargs=dict(nlive=50, tol=0.1, verbose=0, detector_name='test_combined'),
                                       fit_parameters=fit_parameters,)
        sampler.save_results()
        sampler.save_sub_configs()

        print('opening results')
        print(os.listdir(sampler.results_dir))
        results = dddm.ResultsManager(sampler.results_dir)
        print(results)
        results.apply_mask(results.df['nlive'] > 1)
        assert results.result_cache is not None and len(results.result_cache) > 0

        if len(results.result_cache) > 30:
            raise RuntimeError(f'Too many matches for {sampler.results_dir}')

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

            dddm.plotting.confidence_figures.y_label()
            dddm.plotting.confidence_figures.x_label()
            dddm.plotting.confidence_figures.set_xticks_top()
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

    def test_multinest_migdal(self):
        self.test_multinest(detector_name='XENONnT_Migdal',
                            prior='migdal_wide',
                            wimp_mass=5,
                            cross_section=1e-40,
                            sampler_kwargs=dict(nlive=30, tol=0.1, verbose=0),
                            )

    def test_multinest_combined(self,
                                halo_name='shm',
                                fit_parameters=('log_mass', 'log_cross_section',)):
        self.test_multinest(
            wimp_mass=50,
            cross_section=1e-45,
            sampler_name='multinest',
            detector_name=['Xe_simple', 'Ar_simple', 'Ge_simple'],
            prior="Pato_2010",
            halo_name=halo_name,
            detector_kwargs=None,
            halo_kwargs=None if halo_name == 'shm' else dict(location='XENON'),
            sampler_kwargs=dict(nlive=50, tol=0.1, verbose=0, detector_name='test_combined'),
            fit_parameters=fit_parameters,
        )
