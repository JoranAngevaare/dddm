import dddm
from unittest import skipIf


@skipIf(not dddm.is_installed('pymultinest'), 'pymultinest is not installed')
@skipIf(dddm.is_windows(), "Multinest only works on linux")
def test_nested_simple_multinest_earth_shielding():
    fit_class = dddm.NestedSamplerStatModel('Xe')
    fit_class.config['tol'] = 0.1
    fit_class.config['nlive'] = 10
    fit_class.config['earth_shielding'] = True
    fit_class.config['max_iter'] = 1
    fit_class.config['save_intermediate'] = True
    print(f"Fitting for parameters:\n{fit_class.config['fit_parameters']}")
    fit_class.run_multinest()
