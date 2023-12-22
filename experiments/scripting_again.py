""" Scripting again

cd ../testing
datalad install https://datasets.datalad.org/openneuro/ds000105
datalad get ds000105/sub-1
"""

from pathlib import Path

import numpy as np
rng = np.random.default_rng()

import pandas as pd

import matplotlib.pyplot as plt

import xarray as xr

import nibabel as nib
import xibabel as xib

img_path_root = (Path(__file__).parent.parent / 'testing' /
            'ds000105' /
            'sub-1' /
            'func' /
            'sub-1_task-objectviewing_run-01_')


bold_path = img_path_root.with_name(img_path_root.name + 'bold.nii.gz')
tsv_path = img_path_root.with_name(img_path_root.name + '_events.tsv')

nib_img = nib.load(bold_path)
vol_shape = nib_img.shape[:-1]
n_vols = nib_img.shape[-1]

# A fake design matrix.  I'll fix this up.
design_matrix = np.hstack(
    [rng.normal(size=(n_vols, 2)), np.ones((n_vols, 1))])
# For notation!
X = design_matrix
# Analysis the old way.

# The 4D array (i, j, k, time)
data = nib_img.get_fdata()

# Reshape to time by (i * j * k)
vols_by_voxels = np.reshape(data, (-1, n_vols)).T

# Run estimation with B = X+ Y
# To get B such that errors (E = Y - X @ B) are minimized in the sense
# of having the smallest np.sum(E ** 2, axis=0) for each column.
# Betas for each voxel (n_cols_in_X by n_voxels).
B = np.linalg.pinv(X) @ vols_by_voxels

# Contrast applied.  Two slopes compared
c = np.array([[-1, 1, 0]])
con_vec = c @ B

con_arr = np.reshape(con_vec, vol_shape)

# OK - that was the crappy way.
# Now the awesome way.
xib_img = xib.load(bold_path)
# assert isinstance(xib_img, xr.DataArray)

# Make the design
xesign = xr.DataArray(
    np.linalg.pinv(X),
    dims=['p', 'time'])

# Make the data chunky.
chunked = xib_img.chunk({'k': 5})

# Do the estimation
xB = xr.dot(xesign, xib_img, dim=['time', 'time'])

xC = xr.DataArray(con_arr, dims=['c', 'p'])

x_c_arr = xr.dot(xC, xB, dim=['p', 'p'])

# I'm guessing this will already be 3D...

# Questions:
# Does this really save memory?
# Is there a better storage format to make this out-of-memory or some similar
# optimization?

