""" Testing utilties and data

We currently only have one data source — `https://www.datalad.org`_.

The datasets we are using come from the `OpenNeuro collection
<https://docs.openneuro.org/git>`_.

To add a file to the data collection:

* ``git submodule add`` the relevant Datalad repository.  For example::

    datalad install https://datasets.datalad.org/openneuro/ds000466

* Add the test files to `test_files.yml`, and maybe `test_sets.yml`, if you
  want to include them in a testing set.
"""

import os
from pathlib import Path
from subprocess import check_call

import yaml

MOD_DIR = Path(__file__).parent

with open(MOD_DIR / 'test_files.yml') as fobj:
    _DEF_FILES_CONFIG = yaml.load(fobj, Loader=yaml.SafeLoader)
with open(MOD_DIR / 'test_sets.yml') as fobj:
    _DEF_SETS_CONFIG = yaml.load(fobj, Loader=yaml.SafeLoader)


class TestFileError(Exception):
    """ Error for missing or damaged test file """


class Fetcher:

    def __init__(self,
                 data_path=None,
                 files_config=None,
                 sets_config=None
                ):
        data_path = data_path if data_path else self.get_data_path()
        self.data_path = Path(data_path).expanduser().resolve()
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._files_config = files_config if files_config else _DEF_FILES_CONFIG
        self._sets_config = sets_config if sets_config else _DEF_SETS_CONFIG
        self._file_sources = self._parse_configs()

    def _parse_configs(self):
        out = {}
        for source, info in self._files_config.items():
            if source == 'datalad':
                for repo in info:
                    url = repo['repo']
                    root = url.split('/')[-1]
                    for filename in repo['files']:
                        out[f'{root}/{filename}'] = {'type': 'datalad',
                                                     'repo': url}
            else:
                raise TestFileError(f'Do not recognize source "{source}"')
        return out

    def get_data_path(self):
        return os.environ.get('XIB_DATA_PATH', '~/.xibabel/data')

    def have_file(self, path):
        if not path.is_file():  # Can be True regardless on Windows.
            return False
        if path.is_symlink():  # Appears to be True on Unices.
            return True
        # By exploration on Windows.
        with open(path, 'rb') as fobj:
            start = fobj.read(15)
        return start != b'/annex/objects/'

    def _get_datalad_file(self, path_str, repo_url):
        path_str = self._source2path_str(path_str)
        file_path = (self.data_path / path_str).resolve()
        repo_str, file_str = path_str.split('/', 1)
        repo_path = self.data_path / repo_str
        if not repo_path.is_dir():
            check_call(['datalad', 'install', repo_url],
                       cwd=self.data_path)
        if not self.have_file(file_path):
            check_call(['datalad', 'get', file_str], cwd=repo_path)
        return file_path

    def _source2path_str(self, path_or_str):
        if isinstance(path_or_str, str):
            path_or_str = Path(path_or_str)
        if path_or_str.is_absolute():
            path_or_str = path_or_str.relative_to(self.data_path)
        return '/'.join(path_or_str.parts)

    def get_file(self, path_str):
        path_str = self._source2path_str(path_str)
        source = self._file_sources.get(path_str)
        if source is None:
            raise TestFileError(
                f'No record of "{path_str}" as test file')
        if source['type'] == 'datalad':
            return self._get_datalad_file(path_str, source['repo'])
        raise TestFileError(f'Unexpected source "{source}"')

    def get_set(self, set_name):
        out_paths = set()
        set_dict = self._sets_config.get(set_name)
        if set_dict is None:
            raise TestFileError(f'No set called "{set_name}"')
        for subset in set_dict.get('superset_of', []):
            out_paths = out_paths.union(self.get_set(subset))
        for path_str in set_dict.get('files', []):
            out_paths.add(self.get_file(path_str))
        return out_paths


fetcher = Fetcher()
DATA_PATH = fetcher.data_path

# Example files
JC_EG_FUNC = (DATA_PATH / 'ds000009' / 'sub-07' / 'func' /
              'sub-07_task-balloonanalogrisktask_bold.nii.gz')
JC_EG_ANAT= (DATA_PATH / 'ds000009' / 'sub-07' / 'anat' /
             'sub-07/anat/sub-07_T1w.nii.gz')
JH_EG_FUNC = (DATA_PATH / 'ds000105' / 'sub-1' / 'func' /
              'sub-1_task-objectviewing_run-01_bold.nii.gz')
