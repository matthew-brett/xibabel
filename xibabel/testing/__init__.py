""" Testing utilties and data
"""

from pathlib import Path
from subprocess import check_call

import yaml

DATA_PATH = Path(__file__).parent

with open(DATA_PATH / 'test_files.yml') as fobj:
    _FILE_CONFIG = yaml.load(fobj, Loader=yaml.SafeLoader)
with open(DATA_PATH / 'test_sets.yml') as fobj:
    _SET_CONFIG = yaml.load(fobj, Loader=yaml.SafeLoader)


_FILE_SOURCES = {}
for source, filenames in _FILE_CONFIG.items():
    for filename in filenames:
        _FILE_SOURCES[filename] = source


def get_datalad_file(path_str):
    path_str = _source2path_str(path_str)
    rel_path = DATA_PATH / path_str
    if not rel_path.is_file():
        check_call(['datalad', 'get', str(rel_path)])
    assert rel_path.is_file()
    return rel_path.resolve()


class TestFileError(Exception):
    """ Error for missing or damaged test file """


def _source2path_str(path_or_str):
    if isinstance(path_or_str, str):
        path_or_str = Path(path_or_str)
    if not path_or_str.is_absolute():
        return str(path_or_str)
    return str(path_or_str.relative_to(DATA_PATH))


def get_file(path_str):
    path_str = _source2path_str(path_str)
    source = _FILE_SOURCES.get(path_str)
    if source is None:
        raise TestFileError(
            f'No record of "{path_str}" as test file')
    return _SOURCE2FETCHER[source](path_str)


def get_set(set_name):
    out_paths = set()
    set_dict = _SET_CONFIG.get(set_name)
    if set_dict is None:
        raise TestFileError(f'No set called "{set_name}"')
    for subset in set_dict.get('superset_of', []):
        out_paths = out_paths.union(get_set(subset))
    for path_str in set_dict.get('files', []):
        out_paths.add(get_file(path_str))
    return out_paths


_SOURCE2FETCHER = {'datalad': get_datalad_file}
