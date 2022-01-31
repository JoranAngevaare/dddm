from .experiment import Experiment, lindhard_quenching_factor, _get_nr_resolution
import dddm
import numpy as np
from functools import partial
from abc import ABC

export, __all__ = dddm.exporter()


class _BaseXenonNt(Experiment, ABC):
    target_material = 'Xe'
    exposure_tonne_year = 20  # https://arxiv.org/pdf/2007.08796.pdf
    location = "XENON"

    # https://arxiv.org/abs/1608.05381
    _energy_parameters = {'k': 0.1735, 'Z': 54}


@export
class XenonNtNr(_BaseXenonNt):
    detector_name = 'XENONnT_NR'
    __version__ = '0.0.0'

    # Use https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.091301
    energy_threshold_kev = 1.6  # keVnr

    # Combined cut & detection efficiency as in
    # https://arxiv.org/pdf/2007.08796.pdf
    cut_efficiency = 0.83
    detection_efficiency = 1

    interaction_type = 'SI'

    def background_function(self, energies_in_kev):
        """
        :return: NR background for Xe detector in events/keV/t/yr
        """
        # From https://arxiv.org/pdf/2007.08796.pdf
        bg_rate = 2.2e-3  # 1/(keV * t * yr)

        # Assume flat background over entire energy range
        # True to first order below 200 keV
        if (e_min := energies_in_kev[0]) > (e_max := energies_in_kev[-1]) or e_max > 200:
            mes = f'Assume flat background only below 200 keV ({e_min}, {e_max})'
            raise ValueError(mes)
        return self._flat_background(len(energies_in_kev), bg_rate)

    def resolution(self, energies_in_kev):
        """
        Use _get_nr_resolution to calculate the energy resolution.

        :param energies_in_kev: NR energies to evaluate the resolution
            function at
        :return:
        """
        energy_nr_to_energy_ee_function = partial(energy_nr_to_energy_ee,
                                                  **self._energy_parameters)

        # Now get e_ee and sigma_ee based on that we can calculate the
        # energy resolution for the NRs
        energy_ee = energy_nr_to_energy_ee_function(energies_in_kev)
        energy_res_ee = xenon_1t_er_resolution(energy_ee)

        return _get_nr_resolution(energies_in_kev,
                                  energy_nr_to_energy_ee_function,
                                  base_resolution=energy_res_ee,
                                  )


@export
class XenonNtMigdal(_BaseXenonNt):
    detector_name = 'XENONnT_Migdal'
    __version__ = '0.0.0'

    # assume https://arxiv.org/abs/2006.09721
    energy_threshold_kev = 1  # keVer

    # Combined cut & detection efficiency as in
    # https://arxiv.org/pdf/2007.08796.pdf
    cut_efficiency = 0.82
    detection_efficiency = 1

    interaction_type = 'migdal_SI'

    def resolution(self, energies_in_kev):
        """Assume the same as the 1T resolution"""
        return xenon_1t_er_resolution(energies_in_kev)

    def background_function(self, energies_in_kev):
        """
        :return: ER background for Xe detector in events/keV/t/yr
        """
        # From https://arxiv.org/pdf/2007.08796.pdf
        bg_rate = 12.3  # 1/(keV * t * yr)

        # Assume flat background over entire energy range
        # True to first order below 200 keV
        if (e_min := energies_in_kev[0]) > (e_max := energies_in_kev[-1]) or e_max > 200:
            mes = f'Assume flat background only below 200 keV ({e_min}, {e_max})'
            raise ValueError(mes)
        return self._flat_background(len(energies_in_kev), bg_rate)


def xenon_1t_er_resolution(energies_in_kev_ee):
    """
    Detector resolution of XENON1T. See e.g. 1 of
        https://journals.aps.org/prd/pdf/10.1103/PhysRevD.102.072004
    :param energies_in_kev_ee: energy in keVee
    :return: resolution at energies_in_kev
    """
    a = 0.310
    b = 0.0037
    return a * np.sqrt(energies_in_kev_ee) + b * energies_in_kev_ee


def energy_nr_to_energy_ee(energy_nr, k, Z):
    return energy_nr * lindhard_quenching_factor(energy_nr, k=k, atomic_number_z=Z)
