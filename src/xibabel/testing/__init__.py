""" Testing utilties and data

See :mod:`fetcher` for the data fetcher, and instructions for adding files.
"""

from pathlib import Path

from .fetcher import Fetcher


fetcher = Fetcher(config_path=Path(__file__).parent)
DATA_PATH = fetcher.data_path
skip_without_file = fetcher.skip_without_file


# Example files
JC_EG_FUNC = (DATA_PATH / 'ds000009' / 'sub-07' / 'func' /
              'sub-07_task-balloonanalogrisktask_bold.nii.gz')
JC_EG_ANAT= (DATA_PATH / 'ds000009' / 'sub-07' / 'anat' /
             'sub-07_T1w.nii.gz')
JH_EG_FUNC = (DATA_PATH / 'ds000105' / 'sub-1' / 'func' /
              'sub-1_task-objectviewing_run-01_bold.nii.gz')
