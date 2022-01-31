"""Introduce detector effects into the expected detection spectrum"""
import warnings
import numba
import numpy as np
import dddm
from .spectrum import GenSpectrum
import typing as ty
from functools import partial
from scipy.interpolate import interp1d

export, __all__ = dddm.exporter()


@export
class DetectorSpectrum(GenSpectrum):
    """
    Convolve a recoil spectrum with the detector effects:
     - background levels
     - energy resolution
     - energy threshold
    """

    def __str__(self):
        return f'Detector effects convolved {super().__repr__()}'

    def _calculate_counts(self,
                          wimp_mass: ty.Union[int, float],
                          cross_section: ty.Union[int, float],
                          poisson: bool,
                          bin_centers: np.ndarray,
                          bin_width: np.ndarray,
                          bin_edges: np.ndarray,
                          ) -> np.ndarray:
        """

        :return: spectrum taking into account the detector properties
        """
        # get the spectrum
        rates = self.spectrum_simple(bin_centers,
                                     wimp_mass=wimp_mass,
                                     cross_section=cross_section)

        # pay close attention, the events in the bg_func are already taking into
        # account the det. efficiency et cetera. Hence the number here should be
        # multiplied by the total exposure (rather than the effective exposure that
        # is multiplied by at the end of this subroutine. Hence the bg rates obtained
        # from that function is multiplied by the ratio between the two.
        rates += self.background_function(bin_centers) * (self.exposure_tonne_year /
                                                          self.effective_exposure)

        # Smear the rates with the detector resolution
        sigma = self.resolution(bin_centers)
        rates = np.array(smear_signal(rates, bin_centers, sigma, bin_width))

        # Set the rate to zero for energies smaller than the threshold
        rates = self.above_threshold(rates, bin_edges, self.energy_threshold_kev)

        # Calculate the total number of events per bin
        rates = rates * bin_width * self.effective_exposure
        return rates

    @staticmethod
    @numba.njit
    def above_threshold(rates: np.ndarray, e_bin_edges: np.ndarray, e_thr: ty.Union[float, int]):
        """
        Apply threshold to the rates. We are right edge inclusive
        bin edges : |bin0|bin1|bin2|
        e_thr     :        |
        bin0 -> 0
        bin1 -> fraction of bin1 > e_thr
        bin2 -> full content

        :param rates: bins with the number of counts
        :param e_bin_edges: 2d array of the left, right bins
        :param e_thr: energy threshold
        :return: rates with energy threshold applied
        """
        for r_i, r in enumerate(rates):
            left_edge, right_edge = e_bin_edges[r_i]
            if left_edge >= e_thr:
                # From now on all the bins will be above threshold we don't
                # have to set to 0 anymore
                break
            if right_edge <= e_thr:
                # this bin is fully below threshold
                rates[r_i] = 0
                continue
            elif left_edge <= e_thr and e_thr <= right_edge:
                fraction_above = (right_edge - e_thr) / (right_edge - left_edge)
                rates[r_i] = r * fraction_above
            else:
                print(left_edge, right_edge, e_thr)
                raise ValueError('How did this happen?')

        return rates


def smear_signal(rate: np.ndarray,
                 energy: np.ndarray,
                 sigma: np.ndarray,
                 bin_width: np.ndarray
                 ):
    """

    :param rate: counts/bin
    :param energy: energy bin_center
    :param sigma: energy resolution
    :param bin_width: should be scalar of the bin width
    :return: the rate smeared with the specified energy resolution at given
    energy

    This function takes a binned DM-spectrum and takes into account the energy
    resolution of the detector. The rate, energy and resolution should be arrays
    of equal length. The the bin_width
    """
    if np.mean(sigma) < np.mean(bin_width):
        warnings.warn(f'Resolution {np.mean(sigma)} smaller than bin_width {bin_width}!',
                      UserWarning)
        return rate
    result_buffer = np.zeros(len(rate), dtype=np.float64)
    return _smear_signal(rate, energy, sigma, bin_width, result_buffer)


@numba.njit
def _smear_signal(rate, energy, sigma, bin_width, result_buffer):
    # pylint: disable=consider-using-enumerate
    for i in range(len(energy)):
        res = 0.
        # pylint: disable=consider-using-enumerate
        for j in range(len(rate)):
            # if i == j and sigma[i] < bin_width[i]:
            #     # When the resolution at the bin of interest, just assume infinite resolution
            #     res = bin_width[j] * rate[j]
            # see formula (5) in https://arxiv.org/abs/1012.3458
            res = res + (bin_width[j] * rate[j] *
                         (1. / (np.sqrt(2. * np.pi) * sigma[j])) *
                         np.exp(-(((energy[i] - energy[j]) ** 2.) / (2. * sigma[j] ** 2.)))
                         )
            # TODO
            #  # at the end of the spectrum the bg-rate drops as the convolution does
            #  # not take into account the higher energies.
            #  weight = length / (j-length)
            #  res = res * weight
        result_buffer[i] = res
    return result_buffer


def _epsilon(e_nr, atomic_number_z):
    return 11.5 * e_nr * (atomic_number_z ** (-7 / 3))


def _g(e_nr, atomic_number_z):
    eps = _epsilon(e_nr, atomic_number_z)
    a = 3 * (eps ** 0.15)
    b = 0.7 * (eps ** 0.6)
    return a + b + eps


@export
def lindhard_quenching_factor(e_nr, k, atomic_number_z):
    """
    https://arxiv.org/pdf/1608.05381.pdf
    """
    if isinstance(e_nr, (list, tuple)):
        e_nr = np.array(e_nr)
    g = _g(e_nr, atomic_number_z)
    a = k * g
    b = 1 + k * g
    return a / b


@export
def lindhard_quenching_factor_xe(e_nr):
    """
    Xenon Lindhard nuclear quenching factor

    """
    return lindhard_quenching_factor(e_nr=e_nr, k=0.1735, atomic_number_z=54)


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
    dummy_e_nr = np.logspace(int(np.log10(energy_nr.min())-2),
                             int(np.log10(energy_nr.max())+2),
                             1000)
    # Need to have dummy_e_x with large sampling
    dummy_e_x=energy_func(dummy_e_nr)

    energy_func_inverse = interp1d(dummy_e_x, dummy_e_nr, bounds_error=False)
    denergy_nr_denergy_x = partial(_derivative, energy_func_inverse)
    return denergy_nr_denergy_x(a=energy_func_inverse(energy_nr))*base_resolution


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
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
