"""Setup the file structure for the software. Specifies several folders:
software_dir: path of installation
"""

import logging
import os
from socket import getfqdn
from immutabledict import immutabledict
import typing as ty
import DirectDmTargets
import verne

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
    installation_folder = DirectDmTargets.__path__[0]
    verne_folder = os.path.join(os.path.split(verne.__path__[0])[0], 'results')
    default_context = {
        'software_dir': installation_folder,
        'results_dir': os.path.join(installation_folder, 'DD_DM_targets_data'),
        'spectra_files': os.path.join(installation_folder, 'DD_DM_targets_spectra'),
        'verne_folder': verne_folder,
        'verne_files': verne_folder,
    }

    tmp_folder=get_temp()
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
    for key in default_context.keys():
        if not os.path.exists(default_context[key]):
            log.warning(f'No folder at {default_context[key]}')
    return default_context


def get_stbc_context(check=True):
    UserWarning('Hardcoding context is deprecated and will be removed soon')
    log = logging.getLogger()
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
    for key in stbc_context.keys():
        if not os.path.exists(stbc_context[key]) and check:
            raise FileNotFoundError(f'No folder at {stbc_context[key]}')
    return stbc_context


def set_context(config: ty.Union[dict, immutabledict]):
    context.update(config)


if 'stbc' in _host or 'nikhef' in _host:
    set_context(get_stbc_context())
else:
    set_context(get_default_context())
