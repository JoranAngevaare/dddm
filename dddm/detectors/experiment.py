import typing as ty

# {'Ge_iZIP': {
#     'material': 'Ge',
#     'type': 'SI',
#     'exp': 56 * 1.e-3,  # Tonne year
#     'cut_eff': 0.75,  # p. 11, right column
#     'nr_eff': 0.85,  # p. 11, left column
#     'E_thr': 272. / 1e3,  # table VIII, Enr
#     "location": "SNOLAB",
#     'res': det_res_superCDMS100,  # table I
#     'bg_func': nr_background_superCDMS_Ge,
#     'E_max': 5,
#     'n_energy_bins': 50,
# }


class Detector:
    detector_name: str
    __version__: str = '0.0.0'
    e_max_kev : ty.Union[int, float]
    e_min_kev : ty.Union[int, float]
    exposure_tonne_year : ty.Union[int, float]
    energy_threshold_kev : ty.Union[int, float]
    cut_efficiency : ty.Union[int, float]
    detection_efficiency : ty.Union[int, float]
    interaction_type: str = 'SI'
    location: str
    n_energy_bins: int = 50

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.detector_name}'

    def _check_class(
            self,
            attributes_should_be_set: ty.Union[list, tuple]=(
                    'detector_name',
                    'e_max_kev',
                    'e_min_kev',
                    'exposure_tonne_year',
                    'energy_threshold_kev',
                    'cut_efficiency',
                    'detection_efficiency',
                    'interaction_type',
                    'location',
                    'n_energy_bins',),
    ):
        if missing := [
            att for att in attributes_should_be_set if getattr(self, att) is None
        ]:
            raise NotImplementedError(f'Missing {missing} for {self}')
        assert self.interaction_type in ['SI'], f'{self.interaction_type} unknown'
        # Should not raise a ValueError
        self.resolution(energies_in_kev=[1])
        self.background_function(energies_in_kev=[1])

    def resolution(self, energies_in_kev):
        """Return resolution at <energies [keV]>"""
        raise NotImplementedError

    def background_function(self, energies_in_kev):
        """Return background at <energies [keV>"""
        raise NotImplementedError
