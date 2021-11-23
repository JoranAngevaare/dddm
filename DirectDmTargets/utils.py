"""Basic functions for saving et cetera"""

import datetime
import logging
import os
import uuid

import numpy as np
from DirectDmTargets import context

log = logging.getLogger()


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


def is_savable_type(item):
    """

    :param item: input of any type.
    :return: bool if the type is saveable by checking if it is in a limitative list
    """
    savables = (list, np.ndarray, int, str, np.int32, np.int64, np.float32, bool, np.float64)
    if isinstance(item, savables):
        return True
    return False


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
        numbers = [_strip_save_to_int(f, save_as) for f in files]
        if numbers:
            n_last = max(numbers)
    return os.path.join(root_dir, save_as + str(n_last + 1))


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
        results_path = _folders_plus_one(base_dir, save_as)
    else:
        results_path = os.path.join(base_dir, save_as + '_HASH' + str(_hash))

    check_folder_for_file(os.path.join(results_path, "some_file_goes_here"))
    log.info('open_save_dir::\tusing ' + results_path)
    # log.debug(
    #     f'Other files in {results_path} base are {os.listdir(os.path.split(results_path)[0])}')
    return results_path


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


def add_pid_to_csv_filename(name):
    """
    :param name: takes name
    :return: abs_file_name, exist_csv
    """
    UserWarning('add_pid_to_csv_filename is deprecated')
    assert '.csv' in name, f"{name} is not .csv"
    # where to look
    requested_folder = os.path.split(name)[0]
    # what to look for
    file_name = os.path.split(name)[-1].replace('.csv', "")

    if os.path.exists(name) and not os.stat(name).st_size:
        # Check that the file we are looking for is not an empty file, that
        # would be bad.
        log.warning(f"WARNING:\t removing empty file {name}")
        os.remove(name)

    # What can we see
    if not os.path.exists(requested_folder):
        exist_csv = False
        if context._host not in name:
            abs_file_name = add_host_and_pid_to_csv_filename(name)
        else:
            abs_file_name = name
        return exist_csv, abs_file_name

    files_in_folder = os.listdir(requested_folder)
    log.debug(f'VerneSHM::\tlooking for "{file_name}" in "{requested_folder}".'
              f'\n\tDoes it have the right file?\n\t'
              f'{is_str_in_list(file_name, files_in_folder)}'
              # f'That folder has "{files_in_folder}".'
              f' ')

    if is_str_in_list(file_name, files_in_folder):
        log.debug(
            f'VerneSHM::\tUsing {str_in_list(file_name, files_in_folder)} since it has {file_name}')
        exist_csv = True
        abs_file_name = os.path.join(requested_folder,
                                     str_in_list(file_name, files_in_folder))
        log.info(f'VerneSHM::\tUsing {abs_file_name} as input')
    else:
        log.info("VerneSHM::\tNo file found")
        exist_csv = False
        if context._host not in name:
            abs_file_name = add_host_and_pid_to_csv_filename(name)
        else:
            abs_file_name = name

    return exist_csv, os.path.abspath(abs_file_name)


def add_host_and_pid_to_csv_filename(csv_name):
    UserWarning('add_host_and_pid_to_csv_filename is deprecated')
    return csv_name.replace('.csv', f'-H{context._host}-P{os.getpid()}.csv')


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
    result = np.vstack((bin_edges(a, b, n)[0:-1], bin_edges(a, b, n)[1:]))
    return np.transpose(result)


def get_logger(name, level='INFO', path=None) -> logging.Logger:
    """
    Get logger with hander in nice format
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
