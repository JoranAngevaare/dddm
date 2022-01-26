from .experiment import Experiment
import numpy as np


class XenonSimple(Experiment):
    detector_name = 'Xe_simple'
    target_material = 'Xe'
    e_max_kev = 100
    e_min_kev = 0
    exposure_tonne_year = 5
    energy_threshold_kev = 10
    cut_efficiency = 0.8
    detection_efficiency = 0.5
    interaction_type = 'SI'
    location = 'XENON'
    n_energy_bins = 10

    def resolution(self, energies_in_kev):
        """Simple square root dependency of the energy resolution"""
        return 0.6 * np.sqrt(energies_in_kev)

    def background_function(self, energies_in_kev):
        """Assume background free detector"""
        return np.zeros(len(energies_in_kev))


class GermaniumSimple(Experiment):
    detector_name = 'Ge_simple'
    target_material = 'Ge'
    e_max_kev = 100
    e_min_kev = 0
    exposure_tonne_year = 3
    energy_threshold_kev = 10
    cut_efficiency = 0.8
    detection_efficiency = 0.9
    interaction_type = 'SI'
    location = 'SUF'
    n_energy_bins = 10

    def resolution(self, energies_in_kev):
        """Simple resolution model"""
        return np.sqrt(0.3 ** 2 + (0.06 ** 2) * energies_in_kev)

    def background_function(self, energies_in_kev):
        """Assume background free detector"""
        return np.zeros(len(energies_in_kev))


class ArgonSimple(Experiment):
    detector_name = 'Ar_simple'
    target_material = 'Ar'
    e_max_kev = 100
    e_min_kev = 0
    exposure_tonne_year = 10
    energy_threshold_kev = 30
    cut_efficiency = 0.8
    detection_efficiency = 0.8
    interaction_type = 'SI'
    location = 'XENON'  # Assume also located at LNGS
    n_energy_bins = 10

    def resolution(self, energies_in_kev):
        """Simple square root dependency of the energy resolution"""
        return 0.7 * np.sqrt(energies_in_kev)

    def background_function(self, energies_in_kev):
        """Assume background free detector"""
        return np.zeros(len(energies_in_kev))
