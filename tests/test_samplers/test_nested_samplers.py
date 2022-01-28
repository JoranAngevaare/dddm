# import logging
# import tempfile
# from unittest import skipIf
# import dddm
# import matplotlib.pyplot as plt
# from unittest import TestCase
#
# log = logging.getLogger()
#
#
# @skipIf(not dddm.is_installed('pymultinest'), 'pymultinest is not installed')
# @skipIf(dddm.is_windows(), "Multinest only works on linux")
# def test_nested_simple_multinest():
#     fit_class = dddm.NestedSamplerStatModel('Xe')
#     fit_class.config['tol'] = 0.5
#     fit_class.config['nlive'] = 50
#     fit_class.set_benchmark(mass=49)
#     print(f"Fitting for parameters:\n{fit_class.config['fit_parameters']}")
#     fit_class.run_multinest()
#     fit_class.get_summary()
#
#
# @skipIf(not dddm.is_installed('pymultinest'), 'pymultinest is not installed')
# @skipIf(dddm.is_windows(), "Multinest only works on linux")
# def test_nested_astrophysics_multinest():
#     fit_unconstrained = dddm.NestedSamplerStatModel('Xe')
#     fit_unconstrained.config['tol'] = 0.1
#     fit_unconstrained.config['nlive'] = 30
#     fit_unconstrained.set_fit_parameters(fit_unconstrained.known_parameters)
#     print(f"Fitting for parameters:"
#           f"\n{fit_unconstrained.config['fit_parameters']}")
#     fit_unconstrained.run_multinest()
#     fit_unconstrained.get_summary()
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         def _ret_temp(*args):
#             return tmpdirname
#
#         dddm.utils.get_result_folder = _ret_temp
#         fit_unconstrained.save_results()
#         save_as = fit_unconstrained.get_save_dir()
#         import warnings
#         warnings.warn(save_as)
#         fit_unconstrained.check_did_run()
#         fit_unconstrained.check_did_save()
#         fit_unconstrained.show_corner()
#         r = dddm.nested_sampling.load_multinest_samples_from_file(save_as)
#         dddm.nested_sampling.multinest_corner(r)
#         plt.clf()
#         plt.close()
#
#
# class NestleTests(TestCase):
#     def setUp(self) -> None:
#         self.ct = dddm.test_context()
#
#     def test_nested_astrophysics_nestle(self):
#         detector = self.st.get_detector('Xe_simple')
#         fit_unconstrained = dddm.NestedSamplerStatModel('Xe')
#         fit_unconstrained.config['sampler'] = 'nestle'
#         fit_unconstrained.config['tol'] = 0.1
#         fit_unconstrained.config['nlive'] = 30
#         fit_unconstrained.config['max_iter'] = 2
#         fit_unconstrained.set_fit_parameters(fit_unconstrained.known_parameters)
#         print(f"Fitting for parameters:"
#               f"\n{fit_unconstrained.config['fit_parameters']}")
#         fit_unconstrained.run_nestle()
#         fit_unconstrained.get_summary()
#
#     def test_nestle():
#         stats = dddm.NestedSamplerStatModel('Xe')
#         stats.config['sampler'] = 'nestle'
#         stats.config['tol'] = 0.1
#         stats.config['nlive'] = 30
#         print('Start run')
#         stats.run_nestle()
#         print('Save results')
#         stats.save_results()
#         print('Show corner')
#         try:
#             stats.show_corner()
#         except FileNotFoundError as e:
#             print(stats.log_dict['saved_in'])
#             import os
#             print(os.listdir(stats.log_dict['saved_in']))
#             raise e
#         plt.close()
#         plt.clf()
#         print('Save & show again')
#         # Deprecate this function?
#         stats.get_tmp_dir()
#         stats.get_save_dir()
#         plt.close()
#         plt.clf()