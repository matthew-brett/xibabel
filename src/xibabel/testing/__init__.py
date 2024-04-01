""" Testing utilties and data

See :mod:`fetcher` for the data fetcher, and instructions for adding files.
"""

from pathlib import Path

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
