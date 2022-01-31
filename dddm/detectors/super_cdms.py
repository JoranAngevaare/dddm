from .experiment import Experiment
from dddm.recoil_rates.detectror_spectrum import lindhard_quenching_factor
import numpy as np
import dddm
from functools import partial

export, __all__ = dddm.exporter()


class _BaseSuperCdms(Experiment):
    """Base class of superCDMS to introduce shared properties"""
    location = "SNOLAB"

    # Parameters needed for eq. 3, 4 of https://arxiv.org/pdf/1610.00006.pdf
    _energy_parameters = dict(
        si_izip={'Z': 14,
                 'k': 0.161,
                 'epsilon': 0.003,
                 'e_delta_v': 0.008,
                 'e_thr_phonon': 175e-3,
                 'sigma_phonon': 25e-3,
                 },
        si_hv={'Z': 14, 'k': 0.161, 'epsilon': 0.003,'e_delta_v': 0.1, 'e_thr_phonon': 100e-3,
               'sigma_phonon': 5e-3,
               },
        ge_hv={'Z': 32, 'k': 0.162, 'epsilon': 0.00382, 'e_delta_v': 0.1, 'e_thr_phonon': 100e-3,
               'sigma_phonon': 10e-3,},
        ge_izip={'Z': 32, 'k': 0.162, 'epsilon': 0.00382, 'e_delta_v': 0.006, 'e_thr_phonon': 350e-3,
                 'sigma_phonon': 50e-3,},
    )

    def _ee_threshold(self, det_key: str):
        """get the energy threshold (ee) based on the energy_parameters"""
        assert self.interaction_type == 'migdal_SI'
        this_conf = self._energy_parameters[det_key]
        return energy_ee_from_energy_phonon(
            e_ph=this_conf['e_thr_phonon'],
            e_delta_v=this_conf['e_delta_v'],
            epsilon=this_conf['epsilon']
        )

    def _ee_res(self, det_key: str):
        """get the energy resolution (ee) based on the energy_parameters"""
        assert self.interaction_type == 'migdal_SI'
        this_conf = self._energy_parameters[det_key]
        return energy_ee_from_energy_phonon(
            e_ph=this_conf['sigma_phonon'],
            e_delta_v=this_conf['e_delta_v'],
            epsilon=this_conf['epsilon']
        )

    def _ex_from_enr(self, det_key: str):
        assert self.interaction_type == 'SI'
        this_conf = self._energy_parameters[det_key]
        if 'izip' in det_key:
            return partial(energy_ionization_from_e_nr,
                    Z=this_conf['Z'],
                    k=this_conf['k'],
                    )
        elif 'hv' in det_key:
            return partial(energy_phonon_from_energy_nr,
                    Z=this_conf['Z'],
                    k=this_conf['k'],
                    e_delta_v=this_conf['e_delta_v'],
                    epsilon=this_conf['epsilon'],
                    )
        raise ValueError(f'got {det_key}?!')


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



# def resolution_nr_si_hv(energies_nr):
#     det_kwargs = {'Z': 14, 'k': 0.161, 'eps': 0.003, 'eta': 1, 'e_delta_v': 0.1}
#     res = 5e-3  # 5 eV phonon resolution
#     energy_function = partial(eph_from_nr, **det_kwargs)
#     return _get_nr_resolution(energies_nr, energy_function, base_resolution=res)

def energy_ee_from_energy_phonon(e_ph, e_delta_v, epsilon):
    """Eq. 4 in https://arxiv.org/abs/1610.00006 rewritten to ee
    (`y`=1) and `eta`=1"""
    return e_ph / (1 + e_delta_v / epsilon)


def energy_phonon_from_energy_nr(e_r_nr, Z, k, e_delta_v, epsilon):
    y = lindhard_quenching_factor(e_r_nr, atomic_number_z=Z, k=k)
    if not isinstance(y, np.ndarray):
        raise ValueError
    return e_r_nr * (1 + y * (e_delta_v / epsilon))


def energy_ionization_from_e_nr(e_r_nr, Z, k):
    y = lindhard_quenching_factor(e_r_nr, atomic_number_z=Z, k=k)
    if not isinstance(y, np.ndarray):
        raise ValueError
    return e_r_nr * y
