""" Init for Xibabel package
"""
import json
import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import xarray as xr
import nibabel as nib
from nipy.algorithms.diagnostics.timediff import time_slice_diffs

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

def load(file_path, format="BIDS"):
    if type(file_path) == str:
        file_path = pathlib.Path(file_path)
    if format.lower() == "zarr":
        return load_zarr(file_path)
    if format.lower() != "bids":
        logger.warn("unknown format %r", format)
        raise ValueError(f"Unknown format '{format}': must be either 'bids' or 'zarr'")
    img= nib.load(file_path)
    # cut off .nii an .nii.gz
    base = file_path.name.split('.')[0]
    sidecar_file = file_path.with_name(base+".json")
    if not sidecar_file.exists():
        logger.warn("Invalid BIDS image, file missing %s", sidecar_file)
        return InvalidBIDSImage(data=img, error="sidecar file missing", )
    with sidecar_file.open() as f:
        sidecar = json.load(f)
    TR = sidecar["RepetitionTime"]
    if img.ndim == 4:
        time_coords = np.arange(0, (img.shape[-1]) * TR, TR)
        # leaving this as img.dataobj will error on inside numeric routine of
        # numpy during xarray.DataArray creation if not all coords are
        # specified
        data = np.array(img.dataobj)
    # Anatomical scans don't have time... is this a dumb thing to do?
    elif img.ndim == 3:
        time_coords = np.array([0])
        data = img.dataobj[..., np.newaxis]
    else:
        return InvalidBIDSImage(data=img, error="data not 3- or 4-dimensional")
    #return data, sidecar
    # TODO get affine, too
    return xr.DataArray(data,
                        dims=["i", "j", "k", "time"],
                        coords={ "time": xr.DataArray(time_coords, dims=["time"],
                                                      attrs={"units": "s"})
                                },
                        attrs={"sidecar": sidecar, "header": dict(img.header),
                               "affine": img.affine.tolist()})


def diagnostics(run):
    return Diagnostics(run)

def remove_outliers(run, outliers, threshold):
    pass
