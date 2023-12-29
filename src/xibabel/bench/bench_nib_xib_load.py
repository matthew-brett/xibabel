import sys

import numpy as np  # noqa:F401
import numpy.testing as npt

import nibabel as nib  # noqa:F401
import xibabel as xib  # noqa:F401

from nibabel.testing import data_path


EG_4D = data_path / 'example4d.nii.gz'


def bench_img_load():
    sys.stdout.flush()
    print("\nNibabel / Xibabel load")
    print("--------------------")
    nib_time = npt.measure(f'nib.load("{EG_4D}")', 10)
    print(f'Nibabel load {nib_time:6.2f}')
    xib_time = npt.measure(f'np.array(xib.load("{EG_4D}"))', 10)
    print(f'Xibabel load {xib_time:6.2f}')
    sys.stdout.flush()
