# Pytest configuration

from pathlib import Path
import os
import sys
import requests
import socket
import shutil
import time

from xibabel.testing import (JC_EG_FUNC, JC_EG_FUNC_JSON, JC_EG_ANAT,
                             JC_EG_ANAT_JSON, JH_EG_FUNC)


import pytest
from xprocess import ProcessStarter


HERE = Path(__file__).parent
TEST_APP_PORT = 8999

TEST_FILES = (
    JC_EG_FUNC,
    JC_EG_FUNC_JSON,
    JC_EG_ANAT,
    JC_EG_ANAT_JSON,
    JH_EG_FUNC,
)


class URLGetter:

    def __init__(self, server_path):
        self.server_path = server_path

    def make_url(self, url):
        return f'http://localhost:{TEST_APP_PORT}/{url}'

    def get(self, url, *args, **kwargs):
        return requests.get(self.make_url(url), *args, **kwargs)

    def read_text(self, url):
        return self.get(url).text

    def read_bytes(self, url):
        return self.get(url, stream=True).content

    def _write_to(self, url, contents, meth_name):
        path = url.replace('/', os.path.sep)
        getattr(self.server_path / path, meth_name)(contents)
        return self.make_url(url)

    def write_bytes_to(self, url, contents):
        return self._write_to(url, contents, 'write_bytes')

    def write_text_to(self, url, contents):
        return self._write_to(url, contents, 'write_text')


@pytest.fixture
def fserver(xprocess, tmp_path_factory, scope='session'):
    # We need our own server rather than pytest-httpserver
    # in order to support range requests.

    server_path = tmp_path_factory.mktemp("files")

    class Starter(ProcessStarter):

        n_tries = 10
        try_sleep = 1

        def startup_check(self):
            # From: https://github.com/piotrekw/tornado-pytest
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            for i in range(self.n_tries):
                result = sock.connect_ex(('127.0.0.1', TEST_APP_PORT))
                if result == 0:
                    return True
                time.sleep(self.try_sleep)
            return False

        terminate_on_interrupt = True

        args = [sys.executable,
                str(HERE / 'run_server.py'),
                str(server_path),
                '-p', str(TEST_APP_PORT)]

    os.environ['APP_PORT'] = str(TEST_APP_PORT)
    xprocess.ensure("fserver", Starter, restart=True)

    for tf in TEST_FILES:
        shutil.copy2(tf, server_path)

    yield URLGetter(server_path)

    # clean up whole process tree afterwards
    xprocess.getinfo("fserver").terminate()
