from .experiment import Experiment
import dddm
import numpy as np

export, __all__ = dddm.exporter()


class _BaseXenonNt(Experiment):
    target_material = 'Xe'
    exposure_tonne_year = 20  # https://arxiv.org/pdf/2007.08796.pdf
    location = "XENON"


@export
class XenonNtNr(_BaseXenonNt):
    detector_name = 'XENONnT_NR'
    __version__ = '0.0.0'

    # Slightly lower than https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.112009
    energy_threshold_kev = 2.5  # keVnr

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
        energies_in_kevee = dddm.lindhard_quenching_factor_xe(energies_in_kev) * energies_in_kev
        return xenon_1t_er_resolution(energies_in_kevee)


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
