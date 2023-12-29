""" Load various data formats into xibabel images
"""

from pathlib import Path
import json
from dataclasses import dataclass
import logging
import importlib

import numpy as np
import psutil

import nibabel as nib
from nibabel.spatialimages import HeaderDataError
from nibabel.filename_parser import splitext_addext
import xarray as xr
import dask.array as da

from .xutils import merge


logger = logging.getLogger(__name__)


def max_available_div(div=10):
    """ Set default chunk as fraction of maximum aailable memory
    """
    return psutil.virtual_memory().available / div


MAXCHUNK_STRATEGY = max_available_div


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
        self.order = getattr(dataobj, 'order', None)

    def __getitem__(self, slicer):
        """ Return image data as floating point type ``self.dtype``.
        """
        return np.asanyarray(self._dataobj[slicer], dtype=self.dtype)

    def chunk_sizes(self, maxchunk=None):
        """ Calculate chunk sizes for dataobj shape

        Parameters
        ----------
        maxchunk : None or int, optional
            The largest allowable chunk sizes in bytes.

        Returns
        -------
        chunk_sizes : list
            Chunk sizes for Dask array creation, being number of elements in
            one chunk over all axes of array in ``self.dataobj``.
        """
        sizes = [None] * self.ndim
        if maxchunk is None:
            maxchunk = MAXCHUNK_STRATEGY()
        chunk_size = np.prod(self.shape) * self.dtype.itemsize
        if chunk_size <= maxchunk:
            return sizes
        axis_nos = range(self.ndim)
        if self.order == 'F':  # Assume C order by default.
            axis_nos = axis_nos[::-1]
        for axis_no in axis_nos:
            chunk_size //= self.shape[axis_no]
            n_chunks = maxchunk // chunk_size
            if n_chunks:
                sizes[axis_no] = int(n_chunks)
                return sizes
            sizes[axis_no] = 1
        return sizes


@dataclass
class InvalidBIDSImage:
    data: None
    error: str


dim_recoder = {
    None: None,
    0: 'i',
    1: 'j',
    2: 'k'}


time_unit_scaler = {
    'sec': 1,
    'msec': 1 / 1000,
    'usec': 1 / 1_000_000}


class NiftiWrapper:

    def __init__(self, header):
        self.header = nib.Nifti1Header.from_header(header)

    def get_dim_labels(self):
        freq_dim, phase_dim, slice_dim = self.header.get_dim_info()
        return {'xib-FrequencyEncodingDirection': dim_recoder[freq_dim],
                'PhaseEncodingDirection': dim_recoder[phase_dim],
                'SliceEncodingDirection': dim_recoder[slice_dim]}

    def get_slice_timing(self):
        hdr = self.header
        freq_dim, phase_dim, slice_dim = hdr.get_dim_info()
        if not slice_dim:
            return None
        duration = hdr.get_slice_duration()
        if duration == 0:
            return None
        slice_start, slice_end = hdr['slice_start'], hdr['slice_end']
        n_slices = hdr.get_n_slices()
        if slice_start != 0 or slice_end != n_slices - 1:
            return None
        try:
            return list(hdr.get_slice_times())
        except HeaderDataError:
            return None

    def get_repetition_time(self):
        hdr = self.header
        zooms = hdr.get_zooms()
        if len(zooms) < 4:
            return None
        time_zoom = zooms[3]
        space_units, time_units = hdr.get_xyzt_units()
        if time_units == 'unknown':
            return None
        return time_unit_scaler[time_units] * time_zoom

    def get_affines(self):
        """ Collect valid affines
        """
        hdr = self.header
        affines = {}
        for affine_type in 'qform', 'sform':
            code = hdr.get_value_label(affine_type + '_code')
            if code != 'unknown':
                affine = getattr(hdr, 'get_' + affine_type)()
                affines[code] = affine.tolist()
        return affines

    def to_meta(self):
        meta = self.get_dim_labels()
        meta['SliceTiming'] = self.get_slice_timing()
        meta['RepetitionTime'] = self.get_repetition_time()
        meta['xib-affines'] = self.get_affines()
        return {k: v for k, v in meta.items() if v is not None}


def wrap_header(header):
    # We could try extracting more information from other file types, but
    return NiftiWrapper(header)


def load_nibabel(file_path):
    img = nib.load(file_path)
    return img, wrap_header(img.header).to_meta()


def _guess_format(file_path):
    suff = Path(file_path).suffix
    if suff == '.json':
        return 'bids'
    if suff == '.ximg':
        return 'zarr'
    if suff == '.nc':
        return 'netcdf'
    # Default defers to Nibabel.
    return None


def load_zarr(file_path):
    return xr.load_dataarray(file_path, engine='zarr')


class XibFileError(Exception):
    """ Error from Xibabel file operations
    """


def load_netcdf(file_path):
    if importlib.util.find_spec('netCDF4') is None:
        raise XibFileError('Please install netcdf4 module to load netCDF')
    img = xr.load_dataarray(file_path,
                            engine=xr.backends.NetCDF4BackendEntrypoint)
    img.attrs = json.loads(img.attrs.get('__json__', ''))
    return img


def load(file_path, format=None):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if format is None:
        format = _guess_format(file_path)
    else:
        format = format.lower()
    if format and format == "zarr":
        return load_zarr(file_path)
    if format and format == "netcdf":
        return load_netcdf(file_path)
    is_bids = format and format == "bids"
    if format and not is_bids:
        raise XibFileError(
            f"Unknown format '{format}': must be None, 'bids' or 'zarr'")
    img, meta = load_nibabel(file_path)
    base = Path(splitext_addext(file_path)[0])
    # cut off .nii an .nii.gz
    if is_bids:
        sidecar_file = base.with_suffix(".json")
        if not sidecar_file.exists():
            logger.warn("Invalid BIDS image, file missing %s", sidecar_file)
            return InvalidBIDSImage(data=img, error="sidecar file missing", )
        with sidecar_file.open() as f:
            sidecar = json.load(f)
        meta = merge(meta, sidecar)
    coords = {}
    if (TR := meta.get("RepetitionTime")):
        time_coords = np.arange(0, (img.shape[-1]) * TR, TR)
        coords = {
            "time":
            xr.DataArray(time_coords, dims=["time"], attrs={"units": "s"})}
    # TODO get affine, too
    dataobj = FDataObj(img.dataobj)
    return xr.DataArray(da.from_array(dataobj, chunks=dataobj.chunk_sizes()),
                        dims=["i", "j", "k", "time"][:dataobj.ndim],
                        coords=coords,
                        name=base.name,
                        # zarr can't serialize numpy arrays as attrs
                        attrs={"meta": meta}) #"header": dict(img.header),
                               #"affine": img.affine.tolist()})


def save(obj, file_path, format=None):
    file_path = Path(file_path)
    format = _guess_format(file_path)
    if format == 'zarr':
        return obj.to_zarr(file_path, mode='w')
    elif format == 'netcdf':
        out = obj.copy()  # Shallow copy by default.
        out.attrs = {'__json__': json.dumps(obj.attrs)}
        return out.to_netcdf(file_path)
    raise XibFileError(f'Saving in format "{format}" not yet supported')
