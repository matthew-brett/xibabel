""" Testing utilties and data

See :mod:`fetcher` for the data fetcher, and instructions for adding files.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..loaders import drop_suffix
from .fetcher import Fetcher


fetcher = Fetcher(config_path=Path(__file__).parent)
get_set = fetcher.get_set
get_file = fetcher.get_file
have_file = fetcher.have_file
DATA_PATH = fetcher.data_path
skip_without_file = fetcher.skip_without_file


# Example files
JC_EG_FUNC = (DATA_PATH / 'ds000009' / 'sub-07' / 'func' /
              'sub-07_task-balloonanalogrisktask_bold.nii.gz')
JC_EG_FUNC_JSON = drop_suffix(JC_EG_FUNC, ('.nii.gz',)).with_suffix('.json')
JC_EG_ANAT= (DATA_PATH / 'ds000009' / 'sub-07' / 'anat' /
             'sub-07_T1w.nii.gz')
JC_EG_ANAT_JSON = drop_suffix(JC_EG_ANAT, ('.nii.gz',)).with_suffix('.json')
JH_EG_FUNC = (DATA_PATH / 'ds000105' / 'sub-1' / 'func' /
              'sub-1_task-objectviewing_run-01_bold.nii.gz')


def _dts_st(dts, dtt, op=all):
    return op(np.issubdtype(dt, dtt) for dt in dts)


def arr_dict_allclose(dict1, dict2, *args, **kwargs):
    r""" True if two dicts are equal or allclose, where dicts may contain arrays

    Parameters
    ----------
    dict1 : object
        object (usually dictionary).
    dict2 : object
        object (usually dictionary).
    \*args : sequence
        Positional arguments to pass to allclose.
    \**args : sequence
        Keyword arguments to pass to allclose.

    Returns
    -------
    isclose : bool
        True if all elements in `dict2` are equal to or np.allclose to
        equivalent elements in `dict1`.
    """
    if not (isinstance(dict1, Mapping) and isinstance(dict2, Mapping)):
        return False
    if not set(dict1) == set(dict2):
        return False
    for key, value1 in dict1.items():
        value2 = dict2[key]
        if isinstance(value1, Mapping):
            if not isinstance(value2, Mapping):
                return False
            if not arr_dict_allclose(value1, value2):
                return False
            continue
        v1_arr = np.array(value1)
        v2_arr = np.array(value2)
        dts = v1_arr.dtype, v2_arr.dtype
        if _dts_st(dts, np.inexact, any) and _dts_st(dts, np.number, all):
            if not np.allclose(v1_arr, v2_arr, *args, **kwargs):
                return False
        elif isinstance(value1, np.ndarray) or isinstance(value2, np.ndarray):
            if np.any(v1_arr != value2):
                return False
        elif value1 != value2:
            return False
    return True
