"""Setup the file structure for the software. Specifies several folders:
software_dir: path of installation
"""

import logging
import os
import warnings
from socket import getfqdn
from immutabledict import immutabledict
import typing as ty
import verne
import dddm

export, __all__ = dddm.exporter()
__all__ += ['log']

context = {}
log = dddm.utils.get_logger('dddm')
_naive_tmp = '/tmp/'
_host = getfqdn()

base_detectors = [
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
        'multinest_combined': dddm.samplers.multi_detectors.CombinedInference,
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
        sampler_class = self._samplers[sampler_name]

        sampler_kwargs = {} if not sampler_kwargs else sampler_kwargs
        halo_kwargs = {} if not halo_kwargs else halo_kwargs
        detector_kwargs = {} if not detector_kwargs else detector_kwargs

        halo_model = self._halo_classes[halo_name](**halo_kwargs)
        # TODO instead, create a super detector instead of smaller ones
        if isinstance(detector_name, (list, tuple)):
            if not sampler_class.allow_multiple_detectors:
                raise NotImplementedError(f'{sampler_class} does not allow multiple detectors')
            detector_instance = [self.get_detector(det, **detector_kwargs) for det in detector_name]
            spectrum_instance = [dddm.DetectorSpectrum(experiment=d,
                                                       dark_matter_model=halo_model)
                                 for d in detector_instance]
        else:
            detector_instance = self.get_detector(detector_name, **detector_kwargs)
            spectrum_instance = dddm.DetectorSpectrum(experiment=detector_instance,
                                                      dark_matter_model=halo_model)
        if isinstance(prior, str):
            prior = dddm.get_priors(prior)
        sampler_instance = sampler_class(wimp_mass=wimp_mass,
                                         cross_section=cross_section,
                                         spectrum_class=spectrum_instance,
                                         prior=prior,
                                         tmp_folder=self._directories['tmp_folder'],
                                         fit_parameters=fit_parameters,
                                         **sampler_kwargs
                                         )
        return sampler_instance

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
    verne_folder = os.path.join(os.path.split(verne.__path__[0])[0], 'results')
    default_context = {
        'software_dir': installation_folder,
        'results_dir': os.path.join(installation_folder, 'DD_DM_targets_data'),
        'spectra_files': os.path.join(installation_folder, 'DD_DM_targets_spectra'),
        'verne_folder': verne_folder,
        'verne_files': verne_folder,
        'tmp_folder': get_temp(),
    }
    context.set_paths(default_context)
    for detector in base_detectors:
        context.register(detector)
    return context


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


# def get_default_context():
#     log.info(f'Host: {_host}')
#
#     # Generally people will end up here
#     log.info(f'context.py::\tunknown host {_host} be careful here')
#     installation_folder = dddm.__path__[0]
#     verne_folder = os.path.join(os.path.split(verne.__path__[0])[0], 'results')
#     default_context = {
#         'software_dir': installation_folder,
#         'results_dir': os.path.join(installation_folder, 'DD_DM_targets_data'),
#         'spectra_files': os.path.join(installation_folder, 'DD_DM_targets_spectra'),
#         'verne_folder': verne_folder,
#         'verne_files': verne_folder,
#     }
#
#     tmp_folder = get_temp()
#     log.debug(f"Setting tmp folder to {tmp_folder}")
#     assert os.path.exists(tmp_folder), f"No tmp folder at {tmp_folder}"
#     default_context['tmp_folder'] = tmp_folder
#     for name in ['results_dir', 'spectra_files']:
#         log.debug(f'context.py::\tlooking for {name} in {default_context}')
#         if not os.path.exists(default_context[name]):
#             try:
#                 os.mkdir(default_context[name])
#             except Exception as e:
#                 log.warning(
#                     f'Could not find nor make {default_context[name]}. Tailor '
#                     f'context.py to your needs. Could not initialize folders '
#                     f'correctly because of {e}.')
#     for key, path in default_context.items():
#         if not os.path.exists(path):
#             log.warning(f'No folder at {path}')
#     return default_context
#
#
# def get_stbc_context(check=True):
#     UserWarning('Hardcoding context is deprecated and will be removed soon')
#     log.info(f'Host: {_host}')
#
#     stbc_context = {
#         'software_dir': '/project/xenon/jorana/software/DD_DM_targets/',
#         'results_dir': '/data/xenon/joranang/dddm/results/',
#         'spectra_files': '/dcache/xenon/jorana/dddm/spectra/',
#         'verne_folder': '/project/xenon/jorana/software/verne/',
#         'verne_files': '/dcache/xenon/jorana/dddm/verne/'}
#
#     tmp_folder = get_temp()
#     if not os.path.exists(tmp_folder) and check:
#         raise FileNotFoundError(f"Cannot find tmp folder at {tmp_folder}")
#     stbc_context['tmp_folder'] = tmp_folder
#     for key, path in stbc_context.items():
#         if not os.path.exists(path) and check:
#             raise FileNotFoundError(f'No folder at {path}')
#     return stbc_context
#
#
# def set_context(config: ty.Union[dict, immutabledict]):
#     context.update(config)
#
#
# if 'stbc' in _host or 'nikhef' in _host:
#     set_context(get_stbc_context())
# else:
#     set_context(get_default_context())


def load_folder_from_context(request):
    """

    :param request: request a named path from the context
    :return: the path that is requested
    """
    if request in context:
        folder = context[request]
    else:
        raise FileNotFoundError(f'Requesting {request} but that is not in {context.keys()}')
    if not os.path.exists(folder):
        raise FileNotFoundError(f'Could not find {folder} (requested was {request}')
    # Should end up here:
    return folder


def get_result_folder(*args):
    """
    bridge to work with old code when context was not yet implemented
    """
    if args:
        log.warning(
            f'get_result_folder::\tfunctionality deprecated ignoring {args}')
    log.info(
        f'get_result_folder::\trequested folder is {context["results_dir"]}')
    return load_folder_from_context('results_dir')


def get_verne_folder():
    """
    bridge to work with old code when context was not yet implemented
    """
    return load_folder_from_context('verne_files')


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
        base_dir = get_result_folder()
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
