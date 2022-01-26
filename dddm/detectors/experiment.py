import typing as ty
import dddm
export, __all__ = dddm.exporter()


@export
class Experiment:
    detector_name: str = None
    __version__: str = '0.0.0'
    e_max_kev: ty.Union[int, float] = None
    e_min_kev: ty.Union[int, float] = None
    exposure_tonne_year: ty.Union[int, float] = None
    energy_threshold_kev: ty.Union[int, float] = None
    cut_efficiency: ty.Union[int, float] = None
    detection_efficiency: ty.Union[int, float] = None
    interaction_type: str = 'SI'
    location: str = None
    n_energy_bins: int = 50

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.detector_name}'

    def _check_class(
            self,
            attributes_should_be_set: ty.Union[list, tuple] = (
                    'detector_name',
                    'e_max_kev',
                    'e_min_kev',
                    'exposure_tonne_year',
                    'energy_threshold_kev',
                    'cut_efficiency',
                    'detection_efficiency',
                    'interaction_type',
                    'location',
                    'n_energy_bins',
            ),
    ):
        missing = []
        for att in attributes_should_be_set:
            if getattr(self, att) is None:
                missing.append(att)
        if missing:
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

    @property
    def config(self):
        required_configs = ('detector_name',
                            'e_max_kev',
                            'e_min_kev',
                            'exposure_tonne_year',
                            'energy_threshold_kev',
                            'cut_efficiency',
                            'detection_efficiency',
                            'interaction_type',
                            'location',
                            'n_energy_bins',
                            'resolution',
                            'background_function'
                            )
        config = {name: getattr(self, name) for name in required_configs}
        return config

    @property
    def detector_hash(self):
        return dddm.hashablize(self.config)
