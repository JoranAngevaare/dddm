"""
Conversions between E_nr and E_ee for several experiments
"""

import typing as ty
import dddm
import numpy as np
from scipy.interpolate import interp1d
from functools import partial

export, __all__ = dddm.exporter()


@export
def lindhard_quenching_factor(e_nr, k, atomic_number_z):
    """
    Calculate lindhard factor L to that allows computing E_ee = L(E_nr) * E_nr

    https://arxiv.org/pdf/1608.05381.pdf
    :param e_nr: er energy in keV
    :param k: lindhard constant
    :param atomic_number_z: atomic number
    :return: lindhard factor at energy e_nr
    """
    if isinstance(e_nr, (list, tuple)):
        e_nr = np.array(e_nr)
    eps = _get_epsilon(e_nr, atomic_number_z)
    g = _get_g(eps)
    return (k * g) / (1 + k * g)


@export
def lindhard_quenching_factor_semi_conductors(e_nr,
                                              k,
                                              atomic_number_z,
                                              U,
                                              c0,
                                              c1,
                                              ):
    """
    Calculate lindhard factor L to that allows computing E_ee = L(E_nr) * E_nr

    eq. (13) from https://arxiv.org/pdf/2001.06503.pdf
    :param e_nr: er energy in keV
    :param k: lindhard constant
    :param atomic_number_z: atomic number
    :param U: see eq. (13) from https://arxiv.org/pdf/2001.06503.pdf
    :param c0: see eq. (13) from https://arxiv.org/pdf/2001.06503.pdf
    :param c1: see eq. (13) from https://arxiv.org/pdf/2001.06503.pdf
    :return: lindhard factor at energy e_nr
    """
    if isinstance(e_nr, (list, tuple)):
        e_nr = np.array(e_nr)
    u = U * _get_cz(atomic_number_z)
    # Shift epsilon_r by u! See p6. https://arxiv.org/pdf/2001.06503.pdf
    eps = _get_epsilon(e_nr, atomic_number_z) - u

    # Eq. 13 https://arxiv.org/pdf/2001.06503.pdf
    nu_bar = _get_nu_bar(eps=eps, c0=c0, c1=c1, U=U, k=k, Z=atomic_number_z)

    # Eq. 2 https://arxiv.org/pdf/2001.06503.pdf
    return (eps - nu_bar + u) / (eps + u)


def _get_nu_bar(eps, c0, c1, U, k, Z):
    # Eq. 13 https://arxiv.org/pdf/2001.06503.pdf
    nu_l = _get_nu_l(eps, k)
    return nu_l + c0 * eps ** 0.5 + c1 + _get_cz(Z) * U


def _get_nu_l(eps, k):
    g = _get_g(eps)
    return eps / (1 + k * g)


def _get_g(eps):
    """https://arxiv.org/pdf/2001.06503.pdf eq. (7)"""
    return 3 * eps ** 0.15 + 0.7 * eps ** 0.6 + eps


def _get_cz(atomic_number_z):
    """https://arxiv.org/pdf/2001.06503.pdf below eq. (1)"""
    return 11.5 * (atomic_number_z ** (-7 / 3))


def _get_epsilon(e_nr, atomic_number_z):
    """For lindhard factor"""
    return e_nr * _get_cz(atomic_number_z)


def _get_nr_resolution(energy_nr: np.ndarray,
                       energy_func: ty.Callable,
                       base_resolution: ty.Union[float, int, np.integer, np.floating, np.ndarray],
                       ) -> ty.Union[int, float, np.integer, np.floating]:
    """
    Do numerical inversion and <energy_func> to get res_nr. Equations:

    energy_X = energy_func(energy_nr)
    res_nr  = (d energy_nr)/(d energy_X) * res_X   | where res_X = base_resolution

    The goal is to obtain res_nr. Steps:
     - find energy_func_inverse:
        energy_func_inverse(energy_X) = energy_nr
     - differentiate (d energy_func_inverse(energy_X))/(d energy_X)=denergy_nr_denergy_x
     - return (d energy_nr)/(d energy_X) * res_X

    :param energy_nr: energy list in keVnr
    :param energy_func: some function that takes energy_nr and returns energy_x
    :param base_resolution: the resolution of energy_X
    :return: res_nr evaluated at energies energy_nr
    """
    low = max(np.log10(energy_nr.min()), -5)
    high = min(np.log10(energy_nr.max()), 5)
    dummy_e_nr = np.logspace(np.int64(low) - 2, np.int64(high) + 2,
                             1000)
    # Need to have dummy_e_x with large sampling
    dummy_e_x = energy_func(dummy_e_nr)

    energy_func_inverse = interp1d(dummy_e_x, dummy_e_nr, bounds_error=False,
                                   fill_value='extrapolate')
    denergy_nr_denergy_x = partial(_derivative, energy_func_inverse)
    return denergy_nr_denergy_x(a=energy_func_inverse(energy_nr)) * base_resolution


def _derivative(f, a, method='central', h=0.01):
    """
    Compute the difference formula for f'(a) with step size h.

    copied from:
        https://personal.math.ubc.ca/~pwalls/math-python/differentiation/differentiation/

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula


    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    """
    if method == 'central':
        return (f(a + h) - f(a - h)) / (2 * h)
    elif method == 'forward':
        return (f(a + h) - f(a)) / h
    elif method == 'backward':
        return (f(a) - f(a - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
