"""Setup the file structure for the software. Specifies several folders:
software_dir: path of installation
"""

import logging
import os
from socket import getfqdn
from immutabledict import immutabledict
import typing as ty
import verne
import dddm
export, __all__ = dddm.exporter()

# __all__ += ['context']

context = {}
log = logging.getLogger()
_naive_tmp = '/tmp/'
_host = getfqdn()


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


def get_default_context():
    log.info(f'Host: {_host}')

    # Generally people will end up here
    log.info(f'context.py::\tunknown host {_host} be careful here')
    installation_folder = dddm.__path__[0]
    verne_folder = os.path.join(os.path.split(verne.__path__[0])[0], 'results')
    default_context = {
        'software_dir': installation_folder,
        'results_dir': os.path.join(installation_folder, 'DD_DM_targets_data'),
        'spectra_files': os.path.join(installation_folder, 'DD_DM_targets_spectra'),
        'verne_folder': verne_folder,
        'verne_files': verne_folder,
    }

    tmp_folder = get_temp()
    log.debug(f"Setting tmp folder to {tmp_folder}")
    assert os.path.exists(tmp_folder), f"No tmp folder at {tmp_folder}"
    default_context['tmp_folder'] = tmp_folder
    for name in ['results_dir', 'spectra_files']:
        log.debug(f'context.py::\tlooking for {name} in {default_context}')
        if not os.path.exists(default_context[name]):
            try:
                os.mkdir(default_context[name])
            except Exception as e:
                log.warning(
                    f'Could not find nor make {default_context[name]}. Tailor '
                    f'context.py to your needs. Could not initialize folders '
                    f'correctly because of {e}.')
    for key, path in default_context.items():
        if not os.path.exists(path):
            log.warning(f'No folder at {path}')
    return default_context


def get_stbc_context(check=True):
    UserWarning('Hardcoding context is deprecated and will be removed soon')
    log.info(f'Host: {_host}')

    stbc_context = {
        'software_dir': '/project/xenon/jorana/software/DD_DM_targets/',
        'results_dir': '/data/xenon/joranang/dddm/results/',
        'spectra_files': '/dcache/xenon/jorana/dddm/spectra/',
        'verne_folder': '/project/xenon/jorana/software/verne/',
        'verne_files': '/dcache/xenon/jorana/dddm/verne/'}

    tmp_folder = get_temp()
    if not os.path.exists(tmp_folder) and check:
        raise FileNotFoundError(f"Cannot find tmp folder at {tmp_folder}")
    stbc_context['tmp_folder'] = tmp_folder
    for key, path in stbc_context.items():
        if not os.path.exists(path) and check:
            raise FileNotFoundError(f'No folder at {path}')
    return stbc_context


def set_context(config: ty.Union[dict, immutabledict]):
    context.update(config)


if 'stbc' in _host or 'nikhef' in _host:
    set_context(get_stbc_context())
else:
    set_context(get_default_context())


def load_folder_from_context(request):
    """

    :param request: request a named path from the context
    :return: the path that is requested
    """
    try:
        folder = context.context[request]
    except KeyError:
        log.info(f'Requesting {request} but that is not in {context.context.keys()}')
        raise KeyError
    if not os.path.exists(folder):
        raise FileNotFoundError(f'Could not find {folder}')
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
        f'get_result_folder::\trequested folder is {context.context["results_dir"]}')
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
    # log.debug(
    #     f'Other files in {results_path} base are {os.listdir(os.path.split(results_path)[0])}')
    return results_path
