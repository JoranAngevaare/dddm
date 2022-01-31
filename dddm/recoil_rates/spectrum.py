import numericalunits as nu
import numpy as np
import pandas as pd
import wimprates as wr
import dddm
from dddm import utils
import typing as ty
from .halo import SHM
from .halo_shielded import ShieldedSHM

export, __all__ = dddm.exporter()


@export
class GenSpectrum:
    required_detector_fields = 'name material type exp_eff'.split()

    def __init__(self,
                 dark_matter_model: ty.Union[SHM, ShieldedSHM],
                 experiment: dddm.Experiment,
                 ):
        """
        :param dark_matter_model: the dark matter model
        :param experiment: dictionary containing detector parameters
        """
        assert issubclass(experiment.__class__, dddm.Experiment)
        self.detector = experiment
        self.dm_model = dark_matter_model

    def __str__(self):
        """
        :return: sting of class info
        """
        return f'{self.dm_model} at {self.detector}'

    def get_data(self,
                 wimp_mass: ty.Union[int, float],
                 cross_section: ty.Union[int, float],
                 poisson=False,
                 return_counts=False,
                 ) -> ty.Union[pd.DataFrame, np.ndarray]:
        """
        :param wimp_mass: wimp mass (not log)
        :param cross_section: cross-section of the wimp nucleon interaction
            (not log)
        :param poisson: type bool, add poisson True or False
        :param return_counts: instead of a dataframe, return counts only
        :return: pd.DataFrame containing events binned in energy
        """
        bin_edges = self.get_bin_edges()
        bin_centers = np.mean(bin_edges, axis=1)
        bin_width = np.diff(bin_edges, axis=1)[:, 0]
        assert len(bin_centers) == len(bin_width)
        assert bin_width[0] == bin_edges[0][1] - bin_edges[0][0]
        counts = self._calculate_counts(wimp_mass=wimp_mass,
                                        cross_section=cross_section,
                                        poisson=poisson,
                                        bin_centers=bin_centers,
                                        bin_width=bin_width,
                                        bin_edges=bin_edges,
                                        )
        counts = self.set_negative_to_zero(counts)
        if return_counts:
            return counts

        result = pd.DataFrame()
        result['counts'] = counts
        result['bin_centers'] = bin_centers
        result['bin_left'] = bin_edges[:, 0]
        result['bin_right'] = bin_edges[:, 1]
        return result

    def get_counts(self,
                   wimp_mass: ty.Union[int, float],
                   cross_section: ty.Union[int, float],
                   poisson=False,
                   ) -> np.array:
        """
        :param wimp_mass: wimp mass (not log)
        :param cross_section: cross-section of the wimp nucleon interaction
            (not log)
        :param poisson: type bool, add poisson True or False
        :return: array of counts/bin
        """
        return self.get_data(wimp_mass=wimp_mass,
                             cross_section=cross_section,
                             poisson=poisson,
                             return_counts=True)

    def _calculate_counts(self,
                          wimp_mass: ty.Union[int, float],
                          cross_section: ty.Union[int, float],
                          poisson: bool,
                          bin_centers: np.ndarray,
                          bin_width: np.ndarray,
                          bin_edges: np.ndarray,
                          ) -> np.ndarray:
        counts = self.spectrum_simple(bin_centers,
                                      wimp_mass=wimp_mass,
                                      cross_section=cross_section)

        if poisson:
            counts = np.random.exponential(counts).astype(np.float)

        counts *= bin_width * self.effective_exposure
        return counts

    def spectrum_simple(self,
                        energy_bins: ty.Union[list, tuple, np.ndarray],
                        wimp_mass: ty.Union[int, float],
                        cross_section: ty.Union[int, float],
                        ):
        """
        Compute the spectrum for a given mass and cross-section
        :param wimp_mass: wimp mass (not log)
        :param cross_section: cross-section of the wimp nucleon interaction
            (not log)
        :return: returns the rate
        """

        material = self.target_material
        exp_type = self.interaction_type

        dddm.log.debug(f'Eval {wimp_mass, cross_section} for {material}-{exp_type}')

        if exp_type in ['SI']:
            rate = wr.rate_wimp_std(energy_bins,
                                    wimp_mass,
                                    cross_section,
                                    halo_model=self.dm_model,
                                    material=material
                                    )
        elif exp_type in ['migdal_SI']:
            # This integration takes a long time, hence, we will lower the
            # default precision of the scipy dblquad integration
            migdal_integration_kwargs = dict(epsabs=1e-4,
                                             epsrel=1e-4)
            convert_units = (nu.keV * (1000 * nu.kg) * nu.year)
            rate = convert_units * wr.rate_migdal(
                energy_bins * nu.keV,
                wimp_mass * nu.GeV / nu.c0 ** 2,
                cross_section * nu.cm ** 2,
                interaction='SI',
                halo_model=self.dm_model,
                material=material,
                **migdal_integration_kwargs
            )
        else:
            raise NotImplementedError(f'Unknown {exp_type}-interaction')
        return rate

    def get_bin_edges(self):
        return utils.get_bins(self.e_min_kev, self.e_max_kev, self.n_energy_bins)

    def set_negative_to_zero(self, counts: np.ndarray):
        mask = counts < 0
        if np.any(mask):
            dddm.log.warning('Finding negative rates. Doing hard override!')
            counts[mask] = 0
            return counts
        return counts

    @property
    def _allowed_requests(self):
        """Which items are we allowed to get from the experiment class"""
        allowed = list(self.detector._required_settings)
        allowed += ['effective_exposure',
                    'resolution',
                    'background_function',
                    ]
        return allowed

    def __getattr__(self, item):
        if hasattr(self.detector, item):
            if item not in self._allowed_requests:
                raise NotImplementedError(f'Ambiguous request ({item}). '
                                          f'Only allowed are:\n{self._allowed_requests}')
            return getattr(self.detector, item)
        return super().__getattribute__(item)
