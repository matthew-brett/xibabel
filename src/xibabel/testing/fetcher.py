""" Fetching and caching data from Datalad stores

We currently only have one data source — `https://www.datalad.org`_.

The datasets we are using come from the `OpenNeuro collection
<https://docs.openneuro.org/git>`_.

To add a file to the data collection, see `test_files.yml`.  Add a new entry
under an existing or new Datalad repository, following the examples.  Then
consider adding that file, with the repository name prefix, to the
`test_sets.yml` file.
"""

import os
from pathlib import Path
from subprocess import check_call
from collections import abc

import yaml


class TestFileError(Exception):
    """ Error for missing or damaged test file """


class Fetcher:

    def __init__(self,
                 config_path=Path(__file__).parent,
                 data_path=None,
                 files_config='test_files.yml',
                 sets_config='test_sets.yml',
                ):
        data_path = data_path if data_path else self.get_data_path()
        self.data_path = Path(data_path).expanduser().resolve()
        self.data_path.mkdir(parents=True, exist_ok=True)
        config_path = Path(config_path)
        self._files_config = self._get_config(files_config, config_path)
        self._sets_config = self._get_config(sets_config, config_path)
        self._file_sources = self._parse_configs()

    def _get_config(self, config, config_path):
        if isinstance(config, abc.Mapping):
            return config
        if isinstance(config, str):
            config = Path(config)
        if not config.is_absolute():
            config = config_path / config
        with open(config) as fobj:
            return yaml.load(fobj, Loader=yaml.SafeLoader)

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
        """ True if `path` is local file

        We need this function to deal with Datalad, which used ``git annex`` to
        store links to larger files.  This function needs to check whether the file
        we have is the actual file, or a link (sort-of-thing) to the file, that
        still needs a ``datalad get`` to fetch the actual contents.
        """
        # Also consider output of `git annex whereis` on `path`.  This shows
        # one entry for [here] if the file is present.  Test on Windows.
        if not path.is_file():  # Can be True for git annex links on Windows.
            return False
        if path.is_symlink():  # Appears to be True on Unices.
            return True
        # By exploration on Windows - detecting git annex placeholder for file.
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

    def skip_without_file(self, path):
        """ Pytest decorator to skip test if file not available
        """
        import pytest
        return pytest.mark.skipif(
            not self.have_file(path),
            reason=f'This test requires file {path}')
