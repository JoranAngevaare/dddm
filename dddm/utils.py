"""Basic functions for saving et cetera"""
import logging
import os
import uuid
import numpy as np
import pandas as pd
import socket
import sys
import datetime
from importlib import import_module
from git import Repo, InvalidGitRepositoryError
from functools import lru_cache
import typing as ty
from collections import defaultdict
from platform import python_version
from base64 import b32encode
from hashlib import sha1
from collections.abc import Mapping
from immutabledict import immutabledict
import json


def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    stolen from
    https://github.com/AxFoundation/strax/blob/d3608efc77acd52e1d5a208c3092b6b45b27a6e2/strax/utils.py#46
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter(export_self=True)


@export
def to_str_tuple(x: ty.Union[str, bytes, list, tuple, pd.Series, np.ndarray]) -> ty.Tuple[str]:
    """
    Convert any sensible instance to a tuple of strings
    stolen from
    https://github.com/AxFoundation/strax/blob/d3608efc77acd52e1d5a208c3092b6b45b27a6e2/strax/utils.py#242
    """
    if isinstance(x, (str, bytes)):
        return (x,)
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, pd.Series):
        return tuple(x.values.tolist())
    elif isinstance(x, np.ndarray):
        return tuple(x.tolist())
    raise TypeError(f"Expected string or tuple of strings, got {type(x)}")


@export
def print_versions(
        modules=('dddm', 'numpy', 'numba', 'wimprates'),
        print_output=True,
        include_python=True,
        return_string=False,
        include_git=True
):
    """
    Print versions of modules installed.

    :param modules: Modules to print, should be str, tuple or list. E.g.
        print_versions(modules=('numpy', 'dddm',))
    :param return_string: optional. Instead of printing the message,
        return a string
    :param include_git: Include the current branch and latest
        commit hash
    :return: optional, the message that would have been printed
    """
    versions = defaultdict(list)
    if include_python:
        versions['module'] = ['python']
        versions['version'] = [python_version()]
        versions['path'] = [sys.executable]
        versions['git'] = [None]
    for m in to_str_tuple(modules):
        result = _version_info_for_module(m, include_git=include_git)
        if result is None:
            continue
        version, path, git_info = result
        versions['module'].append(m)
        versions['version'].append(version)
        versions['path'].append(path)
        versions['git'].append(git_info)
    df = pd.DataFrame(versions)
    info = f'Host {socket.getfqdn()}\n{df.to_string(index=False)}'
    if print_output:
        print(info)
    if return_string:
        return info
    return df


def _version_info_for_module(module_name, include_git):
    try:
        mod = import_module(module_name)
    except (ModuleNotFoundError, ImportError):
        print(f'{module_name} is not installed')
        return
    git = None
    version = mod.__dict__.get('__version__', None)
    module_path = mod.__dict__.get('__path__', [None])[0]
    if include_git:
        try:
            repo = Repo(module_path, search_parent_directories=True)
        except InvalidGitRepositoryError:
            # not a git repo
            pass
        else:
            try:
                branch = repo.active_branch
            except TypeError:
                branch = 'unknown'
            try:
                commit_hash = repo.head.object.hexsha
            except TypeError:
                commit_hash = 'unknown'
            git = f'branch:{branch} | {commit_hash[:7]}'
    return version, module_path, git


def check_folder_for_file(file_path):
    """
    :param file_path: path with one or more subfolders
    """
    last_folder = os.path.split(file_path)[0]
    log.debug(
        f'making path for {file_path}. Requested folder is {last_folder}')
    os.makedirs(last_folder, exist_ok=True)

    if not os.path.exists(last_folder):
        raise OSError(f'Could not make {last_folder} for saving {file_path}')


def now(tstart=None):
    """

    :return: datetime.datetime string with day, hour, minutes
    """
    res = datetime.datetime.now().isoformat(timespec='minutes')
    if tstart:
        res += f'\tdt=\t{(datetime.datetime.now() - tstart).seconds} s'
    return res


@export
def is_windows():
    return 'win' in sys.platform


@export
@lru_cache
def is_installed(module):
    """Try to import <module>, return False if not installed"""
    try:
        import_module(module)
        return True
    except (ModuleNotFoundError, ImportError):
        return False


def is_savable_type(item):
    """

    :param item: input of any type.
    :return: bool if the type is saveable by checking if it is in a limitative list
    """
    savables = (list, int, str, bool)
    return isinstance(item, savables)


def convert_dic_to_savable(config):
    """

    :param config: some dictionary to save
    :return: string-like object that should be savable.
    """
    result = config.copy()
    for key in result.keys():
        if is_savable_type(result[key]):
            pass
        elif isinstance(result[key], dict):
            result[key] = convert_dic_to_savable(result[key])
        else:
            result[key] = str(result[key])
    return result


def _strip_save_to_int(f, save_as):
    try:
        return int(f.split(save_as)[-1])
    except (ValueError, IndexError):
        return -1


def _folders_plus_one(root_dir, save_as):
    # Set to -1 (+1 = 0 ) for the first directory. e.g. rootdir does not exist
    n_last = -1

    if os.path.exists(root_dir):
        files = os.listdir(root_dir)
        if numbers := [_strip_save_to_int(f, save_as) for f in files]:
            n_last = max(numbers)
    return os.path.join(root_dir, save_as + str(n_last + 1))


def str_in_list(string, _list):
    """checks if sting is in any of the items in _list
    if so return that item"""
    for name in _list:
        if string in name:
            return name
    raise FileNotFoundError(f'No name named {string} in {_list}')


def is_str_in_list(string, _list):
    """checks if sting is in any of the items in _list.
    :return bool:"""
    # log.debug(f'is_str_in_list::\tlooking for {string} in {_list}')
    for name in _list:
        if string in name:
            log.debug(f'is_str_in_list::\t{string} is in  {name}!')
            return True
        # log.debug(f'is_str_in_list::\t{string} is not in  {name}')
    return False


def add_temp_to_csv(abspath):
    assert '.csv' in abspath, f"{abspath} is not .csv"
    abspath = abspath.replace('.csv', f'_temp_{os.getpid()}.csv')
    return abspath


def unique_hash():
    return uuid.uuid4().hex[15:]


def remove_nan(x, maskable=False):
    """
    :param x: float or array
    :param maskable: array to take into consideration when removing NaN and/or
    inf from x
    :return: x where x is well defined (not NaN or inf)
    """
    if not isinstance(maskable, bool):
        assert_string = f"match length maskable ({len(maskable)}) to length array ({len(x)})"
        assert len(x) == len(maskable), assert_string
    if maskable is False:
        mask = ~not_nan_inf(x)
        return masking(x, mask)
    return masking(x, ~not_nan_inf(maskable) ^ not_nan_inf(x))


def not_nan_inf(x):
    """
    :param x: float or array
    :return: array of True and/or False indicating if x is nan/inf
    """
    if np.shape(x) == () and x is None:
        x = np.nan
    try:
        return np.isnan(x) ^ np.isinf(x)
    except TypeError:
        return np.array([not_nan_inf(xi) for xi in x])


def masking(x, mask):
    """
    :param x: float or array
    :param mask: array of True and/or False
    :return: x[mask]
    """
    assert len(x) == len(
        mask), f"match length mask {len(mask)} to length array {len(x)}"
    try:
        return x[mask]
    except TypeError:
        return np.array([x[i] for i in range(len(x)) if mask[i]])


def bin_edges(a, b, n):
    """
    :param a: lower limit
    :param b: upper limit
    :param n: number of bins
    :return: bin edges for n bins

    """
    _, edges = np.histogram(np.linspace(a, b), bins=n)
    return edges


def get_bins(a, b, n) -> np.ndarray:
    """
    :param a: lower limit
    :param b: upper limit
    :param n: number of bins
    :return: center of bins
    """
    result = np.vstack((bin_edges(a, b, n)[:-1], bin_edges(a, b, n)[1:]))
    return np.transpose(result)


def get_logger(name, level='INFO', path=None) -> logging.Logger:
    """
    Get logger with handler in nice format
    :param name: name of the logger
    :param level: logging level
    :param path: where to save the log files
    :return: logger
    """
    level = level.upper()
    new_log = logging.getLogger(name)
    if not hasattr(logging, level):
        raise ValueError(f'{level} is invalid for logging')
    new_log.setLevel(getattr(logging, level))
    new_log.handlers = [FormattedHandler(path=path)]
    return new_log


class FormattedHandler(logging.Handler):
    def __init__(self, *args, path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path

    def emit(self, record):
        m = self.formatted_message(record)
        self.write(m)
        # Strip \n
        print(m[:-1])

    def write(self, m):
        if self.path is None:
            return
        self.f = open(self.path, 'a')
        self.f.write(m)

    @staticmethod
    def formatted_message(record):
        func_line = f'{record.funcName} (L{record.lineno})'
        date = datetime.datetime.fromtimestamp(record.created)
        date.isoformat(sep=' ')
        return (f"{date.isoformat(sep=' ')} | "
                f"{record.name[:8]} |"
                f"{record.levelname.upper():8} | "
                f"{func_line:20} | "
                f"{record.getMessage()}\n"
                )


def _immutable_to_dict(some_dict):
    new_dict = {}
    for k, v in some_dict.items():
        if isinstance(v, immutabledict):
            v = _immutable_to_dict(v)
        new_dict[k] = v
    return new_dict


def hashablize(obj):
    """
    Convert a container hierarchy into one that can be hashed.
    See http://stackoverflow.com/questions/985294
    """
    if isinstance(obj, Mapping):
        # Convert immutabledict etc for json decoding
        obj = dict(obj)
    try:
        hash(obj)
    except TypeError:
        if isinstance(obj, dict):
            return tuple((k, hashablize(v)) for (k, v) in sorted(obj.items()))
        elif isinstance(obj, np.ndarray):
            return tuple(obj.tolist())
        elif hasattr(obj, '__iter__'):
            return tuple(hashablize(o) for o in obj)
        else:
            raise TypeError("Can't hashablize object of type %r" % type(obj))
    else:
        return obj


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types
    Edited from mpl3d: mpld3/_display.py
    """

    def default(self, obj):
        try:
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return [self.default(item) for item in iterable]
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@export
def deterministic_hash(thing, length=10):
    """
    Return a base32 lowercase string of length determined from hashing
    a container hierarchy
    """
    hashable = hashablize(thing)
    jsonned = json.dumps(hashable, cls=NumpyJSONEncoder)
    # disable bandit
    digest = sha1(jsonned.encode('ascii')).digest()
    return b32encode(digest)[:length].decode('ascii').lower()


log = get_logger('dddm')
