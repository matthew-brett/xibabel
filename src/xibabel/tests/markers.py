""" Test markers
"""

from importlib.util import find_spec

import pytest

nipy_test = pytest.mark.skipif(not find_spec('nipy'),
                               reason="Need Nipy for this test")

h5netcdf_test = pytest.mark.skipif(not find_spec('h5netcdf'),
                                   reason='Need h5netcdf module for test')
