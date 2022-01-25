import os
import shutil
import numericalunits as nu
import numpy as np
import pandas as pd
import verne
import wimprates as wr
import dddm
from dddm import utils, context
import typing as ty
from .halo import SHM
from .halo_shielded import ShieldedSHM
from scipy.interpolate import interp1d
export, __all__ = dddm.exporter()


@export
class GenSpectrum:
    required_detector_fields = 'name material type exp_eff'.split()

    def __init__(self,
                 wimp_mass: ty.Union[float, int],
                 wimp_nucleon_cross_section: ty.Union[float, int],
                 dark_matter_model: ty.Union[SHM, ShieldedSHM], det):
        """
        :param wimp_mass: wimp mass (not log)
        :param wimp_nucleon_cross_section: cross-section of the wimp nucleon interaction
            (not log)
        :param dark_matter_model: the dark matter model
        :param det: dictionary containing detector parameters
        """
        self._check_input_detector_config(det)

        # note that this is not in log scale!
        self.mw = wimp_mass
        self.sigma_nucleon = wimp_nucleon_cross_section

        self.dm_model = dark_matter_model
        self.config = det
        self.log = utils.get_logger(self.__class__.__name__)

    def __str__(self):
        """
        :return: sting of class info
        """
        return f"spectrum_simple of a DM model ({self.dm_model}) in a " \
               f"{self.config['name']} detector"

    def get_data(self, poisson=True):
        """

        :param poisson: type bool, add poisson True or False
        :return: pd.DataFrame containing events binned in energy
        """
        result = pd.DataFrame()
        result['counts'] = self.get_poisson_events() if poisson else self.get_events()
        bins = utils.get_bins(self.E_min, self.E_max, self.n_bins)
        result['bin_centers'] = np.mean(bins, axis=1)
        result['bin_left'] = bins[:, 0]
        result['bin_right'] = bins[:, 1]
        result = self.set_negative_to_zero(result)
        return result

    def spectrum_simple(self, benchmark):
        """
        Compute the spectrum for a given mass and cross-section
        :param benchmark: insert the kind of DM to consider (should contain Mass
         and cross-section)
        :return: returns the rate
        """
        if not isinstance(benchmark, (dict, pd.DataFrame)):
            benchmark = {'mw': benchmark[0],
                         'sigma_nucleon': benchmark[1]}
        else:
            assert 'mw' in benchmark and 'sigma_nucleon' in benchmark

        material = self.config['material']
        exp_type = self.config['type']

        self.log.debug(f'Eval {benchmark} for {material}-{exp_type}')

        if exp_type in ['SI']:
            rate = wr.rate_wimp_std(self.get_bin_centers(),
                                    benchmark["mw"],
                                    benchmark["sigma_nucleon"],
                                    halo_model=self.dm_model,
                                    material=material
                                    )
        elif exp_type in ['migdal']:
            # This integration takes a long time, hence, we will lower the
            # default precision of the scipy dblquad integration
            migdal_integration_kwargs = dict(epsabs=1e-4,
                                             epsrel=1e-4)
            convert_units = (nu.keV * (1000 * nu.kg) * nu.year)
            rate = convert_units * wr.rate_migdal(
                self.get_bin_centers() * nu.keV,
                benchmark["mw"] * nu.GeV / nu.c0 ** 2,
                benchmark["sigma_nucleon"] * nu.cm ** 2,
                # TODO should this be different for the different experiments?
                q_nr=0.15,
                halo_model=self.dm_model,
                material=material,
                **migdal_integration_kwargs
            )
        else:
            raise NotImplementedError(f'Unknown {exp_type}-interaction')
        return rate

    def set_config(self, update: dict, check_if_set: bool = True) -> None:
        """
        Update the config with the provided update
        :param update: a dictionary of items to update
        :param check_if_set: Check that a previous version is actually
            set
        :return: None
        """
        assert isinstance(update, dict)
        for key in update:
            if check_if_set and key not in self.config:
                message = f'{key} not in config of {self}'
                raise ValueError(message)

        self.config.update(update)

    def _check_input_detector_config(self, det):
        """Given the a detector config, check that all the required fields are available"""
        if not isinstance(det, dict):
            raise ValueError("Detector should be dict")
        missing = [
            field for field in self.required_detector_fields if field not in det
        ]

        if missing:
            raise ValueError(f'Missing {missing} fields in detector config, got {det}')

    def get_bin_centers(self) -> np.ndarray:
        """Given Emin and Emax, get an array with bin centers """
        return np.mean(self.get_bin_edges(), axis=1)

    def get_bin_edges(self):
        return utils.get_bins(self.E_min, self.E_max, self.n_bins)

    def get_events(self):
        """
        :return: Events (binned)
        """
        assert self.config != {}, "First enter the parameters of the detector"
        rate = self.spectrum_simple([self.mw, self.sigma_nucleon])
        bin_width = np.diff(
            utils.get_bins(self.E_min, self.E_max, self.n_bins),
            axis=1)[:, 0]
        return rate * bin_width * self.config['exp_eff']

    def get_poisson_events(self):
        """
        :return: events with poisson noise
        """
        return np.random.exponential(self.get_events()).astype(np.float)

    def set_negative_to_zero(self, result):
        mask = result['counts'] < 0
        if np.any(mask):
            self.log.warning('Finding negative rates. Doing hard override!')
            result['counts'][mask] = 0
            return result
        return result

    @property
    def E_min(self):
        return self.config.get('E_min', 0)

    @property
    def E_max(self):
        return self.config.get('E_max', 10)

    @property
    def n_bins(self):
        return self.config.get('n_energy_bins', 50)
