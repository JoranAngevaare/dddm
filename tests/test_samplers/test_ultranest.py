from unittest import TestCase, skipIf
import dddm
import numpy as np


# @skipIf(dddm.utils.is_windows(), "Ultranest only works on linux")
class PymultinestTest(TestCase):
    def setUp(self) -> None:
        self.ct = dddm.test_context()

    def test(self, halo_name='shm', max_sigma_off=5, **kwargs, ):
        base_config = dict(wimp_mass=50,
                           cross_section=1e-45,
                           sampler_name='ultranest',
                           detector_name='Xe_simple',
                           prior="Pato_2010",
                           halo_name=halo_name,
                           detector_kwargs=None,
                           halo_kwargs=None if halo_name == 'shm' else dict(location='XENON'),
                           sampler_kwargs=dict(nlive=100, tol=0.9, verbose=1),
                           fit_parameters=('log_mass', 'log_cross_section',),
                           )
        config = {**base_config, **kwargs}
        sampler = self.ct.get_sampler_for_detector(**config)

        results, _ = sampler.run()

        fails = []
        for i, (thing, expected, avg, std) in enumerate(
                zip(
                    base_config.get('fit_parameters'),
                    [getattr(sampler, f) for f in base_config.get('fit_parameters')],
                    results['posterior']['mean'],
                    results['posterior']['stdev']
                )):
            nsigma_off = np.abs(expected - avg) / std
            # assert False, dict(thing=thing, expected=expected, avg=avg, nsigma_off=nsigma_off)
            message = (f'For {thing}: expected {expected:.2f} yielded '
                       f'{avg:.2f} +/- {std:.2f}. Off '
                       f'by {nsigma_off:.1f} sigma')
            if nsigma_off > max_sigma_off:
                fails += [message]
            print(message)

        self.assertFalse(fails, fails)
        return sampler

    @skipIf(*dddm.test_utils.skip_long_test())
    def test_combined(self,
                      halo_name='shm',
                      fit_parameters=('log_mass', 'log_cross_section',)):
        self.test(
            wimp_mass=50,
            cross_section=1e-45,
            sampler_name='ultranest_combined',
            detector_name=['Xe_simple', 'Ar_simple', 'Ge_simple'],
            prior="Pato_2010",
            halo_name=halo_name,
            detector_kwargs=None,
            halo_kwargs=None if halo_name == 'shm' else dict(location='XENON'),
            sampler_kwargs=dict(nlive=100, tol=0.9, verbose=1, detector_name='test_combined'),
            fit_parameters=fit_parameters,
        )
