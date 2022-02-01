import typing as ty
import dddm
import numpy as np
from scipy.interpolate import interp1d
from functools import partial

export, __all__ = dddm.exporter()


@export
class Experiment:
    """
    Base class of experiments. To use, subclass and set the required attributes
    """
    detector_name: str = None
    target_material: str = None
    __version__: str = '0.0.0'
    e_min_kev: ty.Union[int, float] = None
    e_max_kev: ty.Union[int, float] = None
    exposure_tonne_year: ty.Union[int, float] = None
    energy_threshold_kev: ty.Union[int, float] = None
    cut_efficiency: ty.Union[int, float] = None
    detection_efficiency: ty.Union[int, float] = None
    interaction_type: str = 'SI'
    location: str = None  # Only needed when taking into account earth shielding
    n_energy_bins: int = 50

    _required_settings = ('detector_name',
                          'target_material',
                          'e_max_kev',
                          'e_min_kev',
                          'exposure_tonne_year',
                          'energy_threshold_kev',
                          'cut_efficiency',
                          'detection_efficiency',
                          'interaction_type',
                          'location',
                          'n_energy_bins',
                          )

    def __init__(self,
                 n_energy_bins=50,
                 e_min_kev=0,
                 e_max_kev=5,
                 ):
        self.n_energy_bins = n_energy_bins
        self.e_min_kev = e_min_kev
        self.e_max_kev = e_max_kev

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.detector_name}. Hash:{self.detector_hash}'

    def _check_class(self):
        if missing := [
            att
            for att in set(self._required_settings)
            if getattr(self, att) is None
        ]:
            raise NotImplementedError(f'Missing {missing} for {self}')
        assert self.interaction_type in ['SI', 'migdal_SI'], f'{self.interaction_type} unknown'
        # Should not raise a ValueError
        self.resolution(energies_in_kev=np.array([1]))
        self.background_function(energies_in_kev=np.array([1]))

    def resolution(self, energies_in_kev: np.ndarray) -> np.ndarray:
        """Return resolution at <energies [keV]>"""
        raise NotImplementedError

    def background_function(self, energies_in_kev: np.ndarray) -> np.ndarray:
        """Return background at <energies [keV>"""
        raise NotImplementedError

    @property
    def effective_exposure(self):
        return self.exposure_tonne_year * self.cut_efficiency * self.detection_efficiency

    @property
    def config(self):
        return {name: getattr(self, name) for name in self._required_settings}

    @property
    def detector_hash(self):
        return dddm.deterministic_hash(self.config)

    @staticmethod
    def _flat_resolution(n_bins: int, resolution_kev: ty.Union[float, int]):
        """Return a flat resolution spectrum over energy range"""
        return np.full(n_bins, resolution_kev)

    @staticmethod
    def _flat_background(n_bins: int, events_per_kev_tonne_year: ty.Union[float, int]):
        """Return a background for n_bins in units of  1/(keV * t * yr)"""
        return np.full(n_bins, events_per_kev_tonne_year)


def _epsilon(e_nr, atomic_number_z):
    """For lindhard factor"""
    return 11.5 * e_nr * (atomic_number_z ** (-7 / 3))


def _g(e_nr, atomic_number_z):
    """For lindhard factor"""
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
