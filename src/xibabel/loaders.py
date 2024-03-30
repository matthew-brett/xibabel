""" Load various data formats into xibabel images
"""

from pathlib import Path
import json
import logging
import importlib
import os.path as op

import numpy as np
import psutil

import nibabel as nib
from nibabel.spatialimages import HeaderDataError
from nibabel.filename_parser import parse_filename
from nibabel.fileholders import FileHolder

import fsspec
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


def load_zarr(url_or_path):
    return xr.open_dataarray(url_or_path, engine='zarr')


class XibError(Exception):
    """ Errors from Xibabel file inference and operations
    """


class XibFormatError(XibError):
    """ Errors from Xibabel format specification or inference
    """


class XibFileError(XibError):
    """ Error from Xibabel file operations
    """


_JSON_MARKER = '__json__'


def _json_attrs2attrs(attrs):
    out = {}
    for key, value in attrs.items():
        if (isinstance(value, list) and
            len(value) == 2 and
            value[0] == _JSON_MARKER):
            value = json.loads(value[1])
        out[key] = value
    return out


def _1d_arrayable(v):
    try:
        arr = np.array(v)
    except ValueError:
        return False
    return arr.ndim < 2


def _attrs2json_attrs(attrs):
    out = {}
    for key, value in attrs.items():
        if (isinstance(value, dict) or
            (isinstance(value, (list, tuple)) and not _1d_arrayable(value))):
            value = [_JSON_MARKER, json.dumps(value)]
        out[key] = value
    return out


def _check_netcdf():
    if importlib.util.find_spec('h5netcdf') is None:
        raise XibFileError('Please install h5netcdf module to load netCDF')


def load_netcdf(url_or_path):
    _check_netcdf()
    with fsspec.open(url_or_path) as fobj:
        img = xr.open_dataarray(fobj, engine='h5netcdf')
    img.attrs = _json_attrs2attrs(img.attrs)
    return img


VALID_URL_SCHEMES = {
    # List from https://docs.python.org/3/library/urllib.parse.html
    # plus others supported by fsspec (see below).
    'file',
    'ftp',
    'gopher',
    'hdl',
    'http',
    'https',
    'imap',
    'mailto',
    'mms',
    'news',
    'nntp',
    'prospero',
    'rsync',
    'rtsp',
    'rtsps',
    'rtspu',
    'sftp',
    'shttp',
    'sip',
    'sips',
    'snews',
    'svn',
    'svn+ssh',
    'telnet',
    'wais',
    'ws',
    'wss',  # End of Python doc list.
    # Following all supported via fsspec
    'gs',  # Google Storage (fsspec)
    'adl',  # Azure Data Lake Gen 1
    'abfs',  # Azure Blob storage.
    'az',  # Azure Data Lake Gen 2
}


def load(url_or_path, format=None):
    if format is None:
        format = PROCESSORS.guess_format(url_or_path)
    return PROCESSORS.get_loader(format)(url_or_path)


_VALID_FILE_EXTS = ('.nii', '.nii.gz')


def drop_suffix(in_path, suffix):
    """ Drop suffix in `suffixes` from str or ``Path`` `in_path`

    `suffix` can be ``str`` (suffix to drop) or sequence.  If sequence, drops
    first matching suffix in sequence `suffix`.

    Parameters
    ----------
    in_path : str or Path
        Path from which to drop suffix.
    suffix : str or sequence
        If ``str``, suffix to drop.  If sequence, search for each suffix, and
        drop first found suffix from `in_path`.

    Returns
    -------
    out_path : str or Path
        Return type matches that of `in_path`.
    """
    if not hasattr(in_path, 'is_file'):
        return _drop_suffix_str(in_path, suffix)
    return in_path.with_name(_drop_suffix_str(in_path.name, suffix))


def _drop_suffix_str(path_str, suffix):
    suffixes = (suffix,) if isinstance(suffix, str) else suffix
    for suffix in suffixes:
        if path_str.endswith(suffix):
            return path_str[:-(len(suffix))]
    return path_str


def replace_suffix(in_path, old_suffix, new_suffix):
    """ Replace `in_path` suffix with `new_suffix`, allowing for `old_suffix`

    Always replace suffix of `in_path`, if present, but allowing for any
    special (multi-dot) suffixes in `old_sequence`.

    `old_suffix` can be ``str`` (suffix to drop) or sequence.  If sequence,
    drops first matching suffix in sequence `suffixes` before appending
    `new_suffix`.

    Parameters
    ----------
    in_path : str or Path
        Path from which to replace suffix.
    old_suffix : str or sequence
        If ``str``, suffix to drop before replacing.  If sequence, search for
        each suffix, and drop first found suffix from `in_path`.
    new_suffix : str
        Suffix to append.

    Returns
    -------
    out_path : str or Path
        Return type matches that of `in_path`.
    """
    is_path = hasattr(in_path, 'is_file')
    path_str = in_path.name if is_path else in_path
    dropped = _drop_suffix_str(path_str, old_suffix)
    replaced = op.splitext(dropped)[0] + new_suffix
    return in_path.with_name(replaced) if is_path else replaced


def _valid_or_raise(fs, url_base, exts):
    for ext in exts:
        target_url = url_base + ext
        if fs.exists(target_url):
            return fsspec.open(target_url, compression='infer')
    msg_suffix = ('one of' if len(exts) > 1 else '') + ', '.join(exts)
    raise XibFileError(
        f"No valid file matching '{url_base}' + {msg_suffix}")


def load_bids(url_or_path, *, require_json=True):
    """ Load image from BIDS-format data at `url_or_path`

    `url_or_path` may point directly to ``.json`` file, or to Nibabel format
    file, for which we expect a ``.json`` file to be present.

    In the second case, if `require_json` is True, and we cannot find a
    matching `.json` file, raise error, otherwise return image as best read.

    Parameters
    ----------
    url_or_path : str or Path
    require_json : {True, False}, optional, keyword-only
        If True, raise error if `url_or_path` is an image, and there is no
        matching JSON file.

    Returns
    -------
    bids_ximg : Xibabel image

    Raises
    ------
    XibFileError
        If `require_json` is True, `url_or_path` does not name a ``.json`` file
        and there is no ``.json`` file corresponding to `url_or_path`.
    """
    sidecar = {}
    # If url_or_path has .json suffix, search for matching image file.
    if str(url_or_path).endswith('.json'):
        sidecar_file = fsspec.open(url_or_path)
        fs_file = _valid_or_raise(sidecar_file.fs,
                                  drop_suffix(url_or_path, '.json'),
                                  _VALID_FILE_EXTS)
    else:  # Image file extensions.  Search for JSON.
        fs_file = fsspec.open(url_or_path, compression='infer')
        fs = fs_file.fs
        url_uncomp = drop_suffix(url_or_path, _comp_exts())
        url_json = replace_suffix(url_uncomp, (), '.json')
        if fs.exists(url_json):
            with fs.open(url_json) as f:
                sidecar = json.load(f)
        elif require_json:
            raise XibFileError(
                f'BIDS loading {url_or_path} but no corresponding '
                f'{url_json} file, and `require_json` is True')
    img, meta = _nibabel2img_meta(fs_file)
    return _img_meta2ximg(img, merge(meta, sidecar), url_or_path)


def load_nibabel(url_or_path):
    return load_bids(url_or_path, require_json=False)


def _comp_exts():
    return tuple(f'.{ext}' for ext in fsspec.utils.compressions)


def _path2class(filename):
    compression_exts = _comp_exts()
    for klass in nib.all_image_classes:
        base, ext, gzext, ftype = parse_filename(filename,
                                                 klass.files_types,
                                                 compression_exts)
        if ftype == 'image':
            return klass
    raise XibFileError(f'No single-file Nibabel class for {filename}')


class FSFileHolder(FileHolder):

    def __del__(self):
        self.fileobj.close()


def _nibabel2img_meta(fs_file):
    # Identify relevant files from fs_file
    # Make file_map with opened files.
    if 'local' in fs_file.fs.protocol:
        img = nib.load(fs_file.path)
    else:  # Not local - use stream interface.
        img_klass = _path2class(fs_file.full_name)
        # We are passing out opened fsspec files.
        fh = FileHolder(fs_file.full_name, fs_file.open())
        img = img_klass.from_file_map({'image': fh})
    return img, wrap_header(img.header).to_meta()


def _img_meta2ximg(img, meta, url_or_path):
    coords = {}
    if (TR := meta.get("RepetitionTime")):
        time_coords = np.arange(0, (img.shape[-1]) * TR, TR)
        coords = {
            "time":
            xr.DataArray(time_coords, dims=["time"], attrs={"units": "s"})}
    dataobj = FDataObj(img.dataobj)
    return xr.DataArray(da.from_array(dataobj, chunks=dataobj.chunk_sizes()),
                        dims=["i", "j", "k", "time"][:dataobj.ndim],
                        coords=coords,
                        name=_url2name(url_or_path),
                        # NB: zarr can't serialize numpy arrays as attrs
                        attrs={"meta": meta}) #"header": dict(img.header),


def _url2name(url_or_path):
    name = drop_suffix(url_or_path, _comp_exts())
    return Path(name).stem


def save(obj, url_or_path, format=None):
    if format is None:
        format = PROCESSORS.guess_format(url_or_path)
    format = 'bids' if format is None else format
    return PROCESSORS.get_saver(format)(obj, url_or_path)


def save_zarr(obj, file_path):
    return obj.to_zarr(file_path, mode='w')


def save_netcdf(obj, file_path):
    _check_netcdf()
    out = obj.copy()  # Shallow copy by default.
    out.attrs = _attrs2json_attrs(out.attrs)
    return out.to_netcdf(file_path, engine='h5netcdf')


class Processors:

    format_processors = {
        'zarr': dict(exts=('ximg',),
                     loader=load_zarr,
                     saver=save_zarr),
        'netcdf': dict(exts=('nc',),
                     loader=load_netcdf,
                     saver=save_netcdf),
        'bids': dict(exts=('json',),
                     loader=load_bids,
                     saver=None),
        'nibabel': dict(exts=(),  # Defer to Nibabel for extensions
                     loader=load_nibabel,
                     saver=None),
    }

    def __init__(self):
        self.loaders = set()
        self.savers = set()
        self.ext2fmt = {}
        for fmt, info in self.format_processors.items():
            if info['loader']:
                self.loaders.add(fmt)
            if info['saver']:
                self.savers.add(fmt)
            for ext in info['exts']:
                self.ext2fmt[ext] = fmt

    def get_loader(self, fmt):
        return self.get_processor(fmt, 'loader')

    def get_saver(self, fmt):
        return self.get_processor(fmt, 'saver')

    def get_processor(self, fmt, ptype='loader'):
        fmt = 'nibabel' if fmt is None else fmt
        pset = (self.loaders if ptype == 'loader'
                else self.savers)
        if fmt not in pset:
            raise XibFormatError(
                f"Cannot use format '{fmt}' as {ptype} for image; "
                f"valid formats are {','.join(pset)}")
        return self.format_processors[fmt][ptype]

    def guess_format(self, file_path):
        suff = str(file_path).split('.')[-1]
        return self.ext2fmt.get(suff)


PROCESSORS = Processors()
