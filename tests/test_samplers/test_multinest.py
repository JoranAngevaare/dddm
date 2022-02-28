from unittest import TestCase, skipIf
import dddm
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


@skipIf(not dddm.utils.is_installed('pymultinest'), 'pymultinest is not installed')
@skipIf(dddm.utils.is_windows(), "Multinest only works on linux")
class PymultinestTest(TestCase):
    def setUp(self) -> None:
        self.ct = dddm.test_context()

    @skipIf(*dddm.test_utils.skip_long_test())
    def test_shielded_full_astrophysics(self, ):
        self.test(halo_name='shielded_shm',
                  sampler_kwargs=dict(nlive=10, tol=0.9, verbose=0),
                  fit_parameters=dddm.statistics.get_param_list())

    @skipIf(*dddm.test_utils.skip_long_test())
    def test_shielded(self):
        self.test(halo_name='shielded_shm',
                  sampler_kwargs=dict(nlive=10, tol=0.9, verbose=0),
                  fit_parameters=dddm.statistics.get_param_list(),
                  )

    def test(self, halo_name='shm', max_sigma_off=5, **kwargs, ):
        base_config = dict(wimp_mass=50,
                           cross_section=1e-45,
                           sampler_name='multinest',
                           detector_name='Xe_simple',
                           prior="Pato_2010",
                           halo_name=halo_name,
                           detector_kwargs=None,
                           halo_kwargs=None if halo_name == 'shm' else dict(location='XENON'),
                           sampler_kwargs=dict(nlive=100, tol=0.1, verbose=1),
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
                       f'{avg:.2f} +/- {std:.2f}. Off '
                       f'by {nsigma_off:.1f} sigma')
            if nsigma_off > 5:
                fails += [message]
            print(message)

        self.assertFalse(fails, fails)
        return sampler

    def test_chain(self,
                   halo_name='shm',
                   fit_parameters=('log_mass', 'log_cross_section',),
                   ):
        sampler = self.test(detector_name=['Xe_simple', 'Ar_simple', 'Ge_simple'],
                            sampler_name='multinest_combined',
                            prior="Pato_2010",
                            halo_name=halo_name,
                            detector_kwargs=None,
                            halo_kwargs=None if halo_name == 'shm' else dict(
                                location='XENON'),
                            sampler_kwargs=dict(nlive=50, tol=0.1, verbose=1,
                                                detector_name='test_combined'),
                            fit_parameters=fit_parameters, )
        sampler.save_results()
        sampler.save_sub_configs()
        sampler.show_corner()
        print('opening results')
        print(os.listdir(sampler.results_dir))
        results = dddm.ResultsManager(os.path.join(sampler.results_dir,
                                                   f'*{sampler.__class__.__name__[:3]}*'))
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

    # Takes too long
    # @skipIf(*dddm.test_utils.skip_long_test())
    # def test_migdal(self):
    #     self.test(detector_name='XENONnT_Migdal',
    #               prior='migdal_wide',
    #               wimp_mass=1,
    #               cross_section=1e-40,
    #               sampler_kwargs=dict(nlive=30, tol=0.9, verbose=1),
    #               )

    @skipIf(*dddm.test_utils.skip_long_test())
    def test_combined(self,
                      halo_name='shm',
                      fit_parameters=('log_mass', 'log_cross_section',)):
        self.test(
            wimp_mass=50,
            cross_section=1e-45,
            sampler_name='multinest_combined',
            detector_name=['Xe_simple', 'Ar_simple', 'Ge_simple'],
            prior="Pato_2010",
            halo_name=halo_name,
            detector_kwargs=None,
            halo_kwargs=None if halo_name == 'shm' else dict(location='XENON'),
            sampler_kwargs=dict(nlive=100, tol=0.1, verbose=1, detector_name='test_combined'),
            fit_parameters=fit_parameters,
        )
