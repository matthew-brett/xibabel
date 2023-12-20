""" Init for Xibabel package
"""
import pathlib

import nibabel as nib
from nipy.algorithms.diagnostics.timediff import time_slice_diffs

__version__ = "0.0.1a0"

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
    subject_id = sub_path.name
    functionals = dict(load_runs(sub_path / "func"))
    anatomicals = dict(load_runs(sub_path / "anat"))

    return subject_id, Subject(subject_id, anatomicals, functionals)

def load_runs(data_path):
    # TODO: also load json sidecar and xib-ify the data here
    runs_paths = data_path.rglob("sub-*.nii.gz")
    # TODO exists check necessary for subjects not yet fetched by datalad
    return dict((r.name, nib.load(r)) for r in runs_paths if r.exists())

def diagnostics(run):
    return Diagnostics(run)

def remove_outliers(run, outliers, threshold):
    pass
