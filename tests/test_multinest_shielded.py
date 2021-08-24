from sys import platform

import DirectDmTargets as dddm


def _is_windows():
    return 'win' in platform


def test_nested_simple_multinest_earth_shielding():
    if _is_windows():
        return
    fit_class = dddm.NestedSamplerStatModel('Xe')
    fit_class.config['tol'] = 0.1
    fit_class.config['nlive'] = 5
    fit_class.config['earth_shielding'] = True
    fit_class.config['max_iter'] = 1
    fit_class.config['save_intermediate'] = True
    fit_class.set_models()
    fit_class.set_benchmark()
    fit_class.print_before_run()
    print(f"Fitting for parameters:\n{fit_class.config['fit_parameters']}")
    fit_class.run_multinest()
