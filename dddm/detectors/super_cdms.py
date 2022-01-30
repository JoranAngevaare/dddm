from .experiment import Experiment
import dddm

export, __all__ = dddm.exporter()


class _BaseSuperCdms(Experiment):
    """Base class of superCDMS to introduce shared properties"""
    location = "SNOLAB"


@export
class SuperCdmsHvGeNr(_BaseSuperCdms):
    detector_name = 'SuperCDMS_HV_Ge_NR'
    target_material = 'Ge'
    interaction_type = 'SI'
    __version__ = '0.0.0'
    exposure_tonne_year = 44 * 1.e-3  # Tonne year
    energy_threshold_kev = 40. / 1e3  # table VIII, Enr
    cut_efficiency = 0.85  # p. 11, right column
    detection_efficiency = 0.85  # p. 11, left column NOTE: ER type!

    def resolution(self, energies_in_kev):
        """Flat resolution"""
        e_res_nr = 10e-3  # 10 eV
        return self._flat_resolution(len(energies_in_kev), e_res_nr)

    def background_function(self, energies_in_kev):
        """Flat bg rate"""
        bg_rate_nr = 27
        return self._flat_background(len(energies_in_kev), bg_rate_nr)


@export
class SuperCdmsHvSiNr(_BaseSuperCdms):
    detector_name = 'SuperCDMS_HV_Si_NR'
    target_material = 'Si'
    interaction_type = 'SI'
    __version__ = '0.0.0'
    exposure_tonne_year = 9.6 * 1.e-3  # Tonne year
    energy_threshold_kev = 78. / 1e3  # table VIII, Enr
    cut_efficiency = 0.85  # p. 11, right column
    detection_efficiency = 0.85  # p. 11, left column NOTE: ER type!

    def resolution(self, energies_in_kev):
        """Flat resolution"""
        e_res_nr = 5e-3  # 5 eV
        return self._flat_resolution(len(energies_in_kev), e_res_nr)

    def background_function(self, energies_in_kev):
        """Flat bg rate"""
        bg_rate_nr = 300
        return self._flat_background(len(energies_in_kev), bg_rate_nr)


@export
class SuperCdmsIzipGeNr(_BaseSuperCdms):
    detector_name = 'SuperCDMS_iZIP_Ge_NR'
    target_material = 'Ge'
    interaction_type = 'SI'
    __version__ = '0.0.0'
    exposure_tonne_year = 56 * 1.e-3  # Tonne year
    energy_threshold_kev = 272. / 1e3  # table VIII, Enr
    cut_efficiency = 0.75  # p. 11, right column
    detection_efficiency = 0.85  # p. 11, left column

    def resolution(self, energies_in_kev):
        """Flat resolution"""
        e_res_nr = 100e-3  # 100 eV
        return self._flat_resolution(len(energies_in_kev), e_res_nr)

    def background_function(self, energies_in_kev):
        """Flat bg rate"""
        bg_rate_nr = 3.3e-3
        return self._flat_background(len(energies_in_kev), bg_rate_nr)


@export
class SuperCdmsIzipSiNr(_BaseSuperCdms):
    detector_name = 'SuperCDMS_iZIP_Si_NR'
    target_material = 'Si'
    interaction_type = 'SI'
    __version__ = '0.0.0'
    exposure_tonne_year = 4.8 * 1.e-3  # Tonne year
    energy_threshold_kev = 166. / 1e3  # table VIII, Enr
    cut_efficiency = 0.75  # p. 11, right column
    detection_efficiency = 0.85  # p. 11, left column

    def resolution(self, energies_in_kev):
        """Flat resolution"""
        e_res_nr = 110e-3  # 100 eV
        return self._flat_resolution(len(energies_in_kev), e_res_nr)

    def background_function(self, energies_in_kev):
        """Flat bg rate"""
        bg_rate_nr = 2.9e-3
        return self._flat_background(len(energies_in_kev), bg_rate_nr)


@export
class SuperCdmsHvGeMigdal(_BaseSuperCdms):
    detector_name = 'SuperCDMS_HV_Ge_Migdal'
    target_material = 'Ge'
    interaction_type = 'migdal_SI'
    __version__ = '0.0.0'
    exposure_tonne_year = 44 * 1.e-3  # Tonne year
    energy_threshold_kev = 100. / 1e3  # table VIII, Eph
    cut_efficiency = 0.85  # p. 11, right column
    detection_efficiency = 0.5  # p. 11, left column NOTE: migdal is ER type!

    def resolution(self, energies_in_kev):
        """Flat resolution"""
        e_res_ee = 10e-3  # 10 eV
        return self._flat_resolution(len(energies_in_kev), e_res_ee)

    def background_function(self, energies_in_kev):
        """Flat bg rate"""
        bg_rate_nr = 27
        return self._flat_background(len(energies_in_kev), bg_rate_nr)


@export
class SuperCdmsHvSiMigdal(_BaseSuperCdms):
    detector_name = 'SuperCDMS_HV_Si_Migdal'
    target_material = 'Si'
    interaction_type = 'migdal_SI'
    __version__ = '0.0.0'
    exposure_tonne_year = 9.6 * 1.e-3  # Tonne year
    energy_threshold_kev = 100. / 1e3  # table VIII, Eph
    cut_efficiency = 0.85  # p. 11, right column
    detection_efficiency = 0.675  # p. 11, left column NOTE: migdal is ER type!

    def resolution(self, energies_in_kev):
        """Flat resolution"""
        e_res_ee = 5e-3  # 5 eV
        return self._flat_resolution(len(energies_in_kev), e_res_ee)

    def background_function(self, energies_in_kev):
        """Flat bg rate"""
        bg_rate_nr = 300
        return self._flat_background(len(energies_in_kev), bg_rate_nr)


@export
class SuperCdmsIzipGeMigdal(_BaseSuperCdms):
    detector_name = 'SuperCDMS_iZIP_Ge_Migdal'
    target_material = 'Ge'
    interaction_type = 'migdal_SI'
    __version__ = '0.0.0'
    exposure_tonne_year = 56 * 1.e-3  # Tonne year
    energy_threshold_kev = 350. / 1e3  # table VIII, Eph
    cut_efficiency = 0.75  # p. 11, right column
    detection_efficiency = 0.5  # p. 11, left column NOTE: migdal is ER type!

    def resolution(self, energies_in_kev):
        """Flat resolution"""
        e_res_ee = 17e-3  # 10 eV
        return self._flat_resolution(len(energies_in_kev), e_res_ee)

    def background_function(self, energies_in_kev):
        """Flat bg rate"""
        bg_rate_nr = 22
        return self._flat_background(len(energies_in_kev), bg_rate_nr)


@export
class SuperCdmsIzipSiMigdal(_BaseSuperCdms):
    detector_name = 'SuperCDMS_iZIP_Si_Migdal'
    target_material = 'Si'
    interaction_type = 'migdal_SI'
    __version__ = '0.0.0'
    exposure_tonne_year = 4.8 * 1.e-3  # Tonne year
    energy_threshold_kev = 175. / 1e3  # table VIII, Eph
    cut_efficiency = 0.75  # p. 11, right column
    detection_efficiency = 0.675  # p. 11, left column NOTE: migdal is ER type!

    def resolution(self, energies_in_kev):
        """Flat resolution"""
        e_res_ee = 25e-3  # 10 eV
        return self._flat_resolution(len(energies_in_kev), e_res_ee)

    def background_function(self, energies_in_kev):
        """Flat bg rate"""
        bg_rate_nr = 370
        return self._flat_background(len(energies_in_kev), bg_rate_nr)
