""" Init for Xibabel package
"""
import json
import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import xarray as xr
import dask.array as da
import nibabel as nib
from nipy.algorithms.diagnostics.timediff import time_slice_diffs

import psutil

__version__ = "0.0.1a0"

logger = logging.getLogger(__name__)

class Study:
    def __init__(self, name, subjects):
        self.name = name
        self.subjects = subjects
    def __repr__(self):
        return f"{__class__.__name__} {repr(self.name)} with " + \
                f"{len(self.subjects)} subjects"

class Subject:
    def __init__(self, name, anatomicals, functionals):
        self.name = name
        self.anatomicals = anatomicals
        self.functionals = functionals

    def __repr__(self):
        return f"{__class__.__name__} {repr(self.name)} with " + \
                f"{len(self.anatomicals)} anatomical runs, and " + \
                f"{len(self.functionals)} functional runs"

    def __len__(self):
        # Can be used for filtering out subjects not yet fetched by datalad
        return len(self.anatomicals) + len(self.functionals)

@dataclass
class InvalidBIDSImage:
    data: None
    error: str


class Diagnostics:
    def __init__(self, run):
        self.run = run

    def find_outliers(self):
        return time_slice_diffs(self.run.dataobj)

def load_study(dir_path, exclude_subjects_without_data_files=True):
    # Not sold if we want such a long parameter name.
    path  = pathlib.Path(dir_path)
    subject_paths = path.glob("sub-*")
    subjects = dict(load_subject(s) for s in subject_paths)
    if exclude_subjects_without_data_files:
        subjects={s: d for (s,d) in subjects.items() if len(d) > 0}
    return Study(path.name, subjects=subjects)


def load_subject(sub_path):
    if type(sub_path) == str:
        sub_path = pathlib.Path(sub_path)
    subject_id = sub_path.name
    # Being a bit cheeky here, either we'll load data from a single session at
    # the subject level
    functionals = dict(load_runs(sub_path / "func"))
    anatomicals = dict(load_runs(sub_path / "anat"))
    # Or we'll get all of them from the session levels
    sessions_paths = sub_path.glob("ses-*")
    for ses_path in sessions_paths:
        functionals |= dict(load_runs(ses_path / "func"))
        anatomicals |= dict(load_runs(ses_path / "anat"))

    return subject_id, Subject(subject_id, anatomicals, functionals)

def load_runs(data_path):
    # TODO: also load json sidecar and xib-ify the data here
    runs_paths = data_path.rglob("sub-*.nii.gz")
    # TODO exists check necessary for subjects not yet fetched by datalad
    return dict((r.name, load(r)) for r in runs_paths if r.exists())


class FDataObj:
    """ Wrapper for dataobj that returns floating point values from indexing.
    """

    def __init__(self, dataobj, dtype=np.float64):
        dtype = np.dtype(dtype)
        if not issubclass(dtype.type, np.inexact):
            raise ValueError(f'{dtype} should be floating point type')
        self._dataobj = dataobj
        self.dtype = dtype
        self.shape = dataobj.shape
        self.ndim = dataobj.ndim

    def __getitem__(self, slicer):
        return np.asanyarray(self._dataobj[slicer], dtype=self.dtype)

    def chunk_sizes(self, maxchunk=None):
        sizes = [None] * self.ndim
        if maxchunk is None:
            maxchunk = psutil.virtual_memory().available / 10
        # Assume memory fastest changing in first dimension (F order).
        item_size = self.dtype.itemsize
        for axis_no in range(self.ndim, -1, -1):
            data_size = np.prod(self.shape[:axis_no + 1]) * item_size
            if data_size < maxchunk:
                return sizes
            n_chunks = data_size // maxchunk
            if n_chunks:
                sizes[axis_no] = maxchunk // item_size
                return sizes
            sizes[axis_no] = 1
        return sizes


def load(file_path, format=None):
    if type(file_path) == str:
        file_path = pathlib.Path(file_path)
    if format and format.lower() == "zarr":
        return load_zarr(file_path)
    is_bids = format and format.lower() == "bids"
    if format and not is_bids:
        logger.warn("unknown format %r", format)
        raise ValueError(f"Unknown format '{format}': must be either 'bids' or 'zarr'")
    img = nib.load(file_path)
    # cut off .nii an .nii.gz
    base = file_path.name.split('.')[0]
    sidecar_file = file_path.with_name(base+".json")
    if is_bids:
        if not sidecar_file.exists():
            logger.warn("Invalid BIDS image, file missing %s", sidecar_file)
            return InvalidBIDSImage(data=img, error="sidecar file missing", )
            TR = 1 # TODO or try to get it from img
        with sidecar_file.open() as f:
            sidecar = json.load(f)
        TR = sidecar["RepetitionTime"]
    else:
        sidecar = {}
        TR = 1
    if img.ndim == 4:
        time_coords = np.arange(0, (img.shape[-1]) * TR, TR)
        # leaving this as img.dataobj will error on inside numeric routine of
        # numpy during xarray.DataArray creation if not all coords are
        # specified
        #  > numpy/core/numeric.py:330, in full(shape, fill_value, dtype, order, like)
        # ValueError: could not broadcast input array from shape (40,64,64,121) into shape (1,1,1,121)
    # Anatomical scans don't have time... is this a dumb thing to do?
    elif img.ndim == 3:
        time_coords = np.array([0])
        img.dataobj = img.dataobj[..., np.newaxis]
    else:
        return InvalidBIDSImage(data=img, error="data not 3- or 4-dimensional")
    #return data, sidecar
    # TODO get affine, too
    dataobj = FDataObj(img.dataobj)
    return xr.DataArray(da.from_array(dataobj, chunks=dataobj.chunk_sizes()),
                        dims=["i", "j", "k", "time"],
                        coords={ "time": xr.DataArray(time_coords, dims=["time"],
                                                      attrs={"units": "s"})
                                },
                        # zarr can't serialize numpy arrays as attrs
                        attrs={"sidecar": sidecar, })#"header": dict(img.header),
                               #"affine": img.affine.tolist()})


def diagnostics(run):
    return Diagnostics(run)

def remove_outliers(run, outliers, threshold):
    pass
