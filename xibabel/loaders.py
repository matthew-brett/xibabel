""" Load various data formats into xibabel images
"""

from pathlib import Path
import json
from dataclasses import dataclass
import logging

import numpy as np
import psutil

import nibabel as nib
import xarray as xr
import dask.array as da


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
        axis_nos = range(self.ndim)
        item_size = self.dtype.itemsize
        chunk_size = np.prod(self.shape) * item_size
        if chunk_size <= maxchunk:
            return sizes
        if self.order == 'F':  # Assume C order by default.
            axis_nos = axis_nos[::-1]
        for axis_no in axis_nos:
            chunk_size //= self.shape[axis_no]
            n_chunks = maxchunk // chunk_size
            if n_chunks > 1:
                sizes[axis_no] = int(n_chunks)
                return sizes
            sizes[axis_no] = 1
        return sizes


def load_zarr(file_path):
    raise NotImplementedError


@dataclass
class InvalidBIDSImage:
    data: None
    error: str



def load(file_path, format=None):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if format and format.lower() == "zarr":
        return load_zarr(file_path)
    is_bids = format and format.lower() == "bids"
    if format and not is_bids:
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
        # ValueError: could not broadcast input array from shape (40,64,64,121)
        # into shape (1,1,1,121)
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