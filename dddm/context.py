"""Setup the file structure for the software. Specifies several folders:
software_dir: path of installation
"""

import inspect
import os
import warnings
from socket import getfqdn

import pandas as pd
from immutabledict import immutabledict
import typing as ty
import dddm
import numpy as np

export, __all__ = dddm.exporter()
__all__ += ['log']

context = {}
log = dddm.utils.get_logger('dddm')
_naive_tmp = '/tmp/'
_host = getfqdn()

base_detectors = [
    dddm.detectors.examples.XenonSimple,
    dddm.detectors.examples.ArgonSimple,
    dddm.detectors.examples.GermaniumSimple,
    dddm.detectors.xenon_nt.XenonNtNr,
    dddm.detectors.xenon_nt.XenonNtMigdal,
    dddm.detectors.super_cdms.SuperCdmsHvGeNr,
    dddm.detectors.super_cdms.SuperCdmsHvSiNr,
    dddm.detectors.super_cdms.SuperCdmsIzipGeNr,
    dddm.detectors.super_cdms.SuperCdmsIzipSiNr,
    dddm.detectors.super_cdms.SuperCdmsHvGeMigdal,
    dddm.detectors.super_cdms.SuperCdmsHvSiMigdal,
    dddm.detectors.super_cdms.SuperCdmsIzipGeMigdal,
    dddm.detectors.super_cdms.SuperCdmsIzipSiMigdal,
]


class Context:
    """Centralized object for managing:
     - configurations
     - files
     - detector objects
    """

    _directories = None
    _detector_registry = None
    _samplers = immutabledict({
        'nestle': dddm.samplers.nestle.NestleSampler,
        'multinest': dddm.samplers.pymultinest.MultiNestSampler,
        'emcee': dddm.samplers.emcee.MCMCStatModel,
        'multinest_combined': dddm.samplers.multi_detectors.CombinedMultinest,
        'nestle_combined': dddm.samplers.multi_detectors.CombinedNestle,
    })
    _halo_classes = immutabledict({
        'shm': dddm.SHM,
        'shielded_shm': dddm.ShieldedSHM,
    })

    def register(self, detector: dddm.Experiment):
        """Register a detector to the context"""
        if self._detector_registry is None:
            self._detector_registry = {}
        existing_detector = self._detector_registry.get(detector.detector_name)
        if existing_detector is not None:
            log.warning(f'replacing {existing_detector} with {detector}')
        self._check_detector_is_valid(detector)
        self._detector_registry[detector.detector_name] = detector

    def set_paths(self, paths: dict, tolerant=False):
        if self._directories is None:
            self._directories = {}
        for reference, path in paths.items():
            if not os.path.exists(path):
                try:
                    os.mkdir(path)
                except Exception as e:
                    if tolerant:
                        warnings.warn(f'Could not find {path} for {reference}', UserWarning)
                    else:
                        raise FileNotFoundError(
                            f'Could not find {path} for {reference}'
                        ) from e

        result = {**self._directories.copy(), **paths}
        self._directories = result

    def show_folders(self):
        result = {'name': list(self._directories.keys())}
        result['path'] = [self._directories[name] for name in result['name']]
        result['exists'] = [os.path.exists(p) for p in result['path']]
        result['n_files'] = [(len(os.listdir(p)) if os.path.exists(p) else 0) for p in
                             result['path']]
        return pd.DataFrame(result)

    def get_detector(self, detector: str, **kwargs):
        if detector not in self._detector_registry:
            raise NotImplementedError(f'{detector} not in {self.detectors}')
        return self._detector_registry[detector](**kwargs)

    def get_sampler_for_detector(self,
                                 wimp_mass,
                                 cross_section,
                                 sampler_name: str,
                                 detector_name: ty.Union[str, list, tuple],
                                 prior: ty.Union[str, dict],
                                 halo_name='shm',
                                 detector_kwargs: dict = None,
                                 halo_kwargs: dict = None,
                                 sampler_kwargs: dict = None,
                                 fit_parameters=dddm.statistics.get_param_list(),
                                 ):
        self._check_sampler_args(wimp_mass, cross_section, sampler_name, detector_name, prior,
                                 halo_name, detector_kwargs, halo_kwargs, sampler_kwargs,
                                 fit_parameters)
        sampler_class = self._samplers[sampler_name]

        # If any class needs any of the paths, provide those here.
        sampler_kwargs = self._add_folders_to_kwargs(sampler_class, sampler_kwargs)
        halo_kwargs = self._add_folders_to_kwargs(
            self._halo_classes.get(halo_name), halo_kwargs)

        halo_model = self._halo_classes[halo_name](**halo_kwargs)
        # TODO instead, create a super detector instead of smaller ones
        if isinstance(detector_name, (list, tuple)):
            if not sampler_class.allow_multiple_detectors:
                raise NotImplementedError(f'{sampler_class} does not allow multiple detectors')

            detector_instance = [
                self.get_detector(
                    det,
                    **self._add_folders_to_kwargs(self._detector_registry.get(det),
                                                  detector_kwargs)
                )
                for det in detector_name]
            if halo_name == 'shielded_shm':
                if len(locations := {d.location for d in detector_instance}) > 1:
                    raise ValueError(
                        f'Running with multiple locations for shielded_shm is not allowed. Got {locations}')
                halo_kwargs.setdefault('log_mass', np.log10(wimp_mass))
                halo_kwargs.setdefault('log_cross_section', np.log10(cross_section))
                halo_kwargs.setdefault('location', list(locations)[0])

            spectrum_instance = [dddm.DetectorSpectrum(
                experiment=d, dark_matter_model=halo_model)
                for d in detector_instance]
        else:
            detector_kwargs = self._add_folders_to_kwargs(
                self._detector_registry.get(detector_name), detector_kwargs)
            detector_instance = self.get_detector(detector_name, **detector_kwargs)
            spectrum_instance = dddm.DetectorSpectrum(experiment=detector_instance,
                                                      dark_matter_model=halo_model)
        if isinstance(prior, str):
            prior = dddm.get_priors(prior)
        return sampler_class(wimp_mass=wimp_mass,
                             cross_section=cross_section,
                             spectrum_class=spectrum_instance,
                             prior=prior,
                             fit_parameters=fit_parameters,
                             **sampler_kwargs
                             )

    def _check_sampler_args(self,
                            wimp_mass,
                            cross_section,
                            sampler_name: str,
                            detector_name: ty.Union[str, list, tuple],
                            prior: ty.Union[str, dict],
                            halo_name='shm',
                            detector_kwargs: dict = None,
                            halo_kwargs: dict = None,
                            sampler_kwargs: dict = None,
                            fit_parameters=dddm.statistics.get_param_list(),
                            ):
        for det in dddm.utils.to_str_tuple(detector_name):
            assert det in self._detector_registry, f'{det} is unknown'
        assert wimp_mass < 200 and wimp_mass > 0.001, f'{wimp_mass} invalid'
        assert np.log10(cross_section) < -20 and np.log10(
            cross_section) > -60, f'{cross_section} invalid'
        assert sampler_name in self._samplers, f'choose from {self._samplers}, got {sampler_name}'
        assert isinstance(prior, (str, dict, immutabledict)), f'invalid {prior}'
        assert halo_name in self._halo_classes, f'invalid {halo_name}'

    def _add_folders_to_kwargs(self, function, current_kwargs: ty.Union[None, dict]) -> dict:
        if function is None:
            return
        if current_kwargs is None:
            current_kwargs = {}
        takes = inspect.getfullargspec(function).args
        for directory, path in self._directories.items():
            if directory in takes:
                current_kwargs.update({directory: path})
        return current_kwargs

    @property
    def detectors(self):
        return sorted(list(self._detector_registry.keys()))

    @staticmethod
    def _check_detector_is_valid(detector: dddm.Experiment):
        detector()._check_class()


@export
def base_context():
    context = Context()
    installation_folder = dddm.__path__[0]

    default_context = {
        'software_dir': installation_folder,
        'results_dir': os.path.join(installation_folder, 'DD_DM_targets_data'),
        'spectra_files': os.path.join(installation_folder, 'DD_DM_targets_spectra'),
        'verne_folder': _get_verne_folder(),
        'verne_files': _get_verne_folder(),
        'tmp_folder': get_temp(),
    }
    context.set_paths(default_context)
    for detector in base_detectors:
        context.register(detector)
    return context


def _get_verne_folder():
    if not dddm.utils.is_installed('verne'):
        return './verne'
    import verne
    return os.path.join(os.path.split(verne.__path__[0])[0], 'results')


def get_temp():
    if 'TMPDIR' in os.environ and os.access(os.environ['TMPDIR'], os.W_OK):
        tmp_folder = os.environ['TMPDIR']
    elif 'TMP' in os.environ and os.access(os.environ['TMP'], os.W_OK):
        tmp_folder = os.environ['TMP']
    elif os.path.exists(_naive_tmp) and os.access(_naive_tmp, os.W_OK):
        tmp_folder = _naive_tmp
    else:
        raise FileNotFoundError('No temp folder available')
    return tmp_folder


def open_save_dir(save_as, base_dir=None, force_index=False, _hash=None):
    """

    :param save_as: requested name of folder to open in the result folder
    :param base_dir: folder where the save_as dir is to be saved in.
        This is the results folder by default
    :param force_index: option to force to write to a number (must be an
        override!)
    :param _hash: add a has to save_as dir to avoid duplicate naming
        conventions while running multiple jobs
    :return: the name of the folder as was saveable (usually input +
        some number)
    """
    if base_dir is None:
        raise ValueError(save_as, base_dir, force_index, _hash)
    if force_index:
        results_path = os.path.join(base_dir, save_as + str(force_index))
    elif _hash is None:
        if force_index is not False:
            raise ValueError(
                f'do not set _hash to {_hash} and force_index to '
                f'{force_index} simultaneously'
            )
        results_path = dddm.utils._folders_plus_one(base_dir, save_as)
    else:
        results_path = os.path.join(base_dir, save_as + '_HASH' + str(_hash))

    dddm.utils.check_folder_for_file(os.path.join(results_path, "some_file_goes_here"))
    log.info('open_save_dir::\tusing ' + results_path)
    return results_path
