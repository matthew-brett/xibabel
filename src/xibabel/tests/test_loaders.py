""" Test loaders
"""

from pathlib import Path
import os
from importlib.util import find_spec
import gzip
from itertools import product
import json
import shutil

import numpy as np

import nibabel as nib

from xibabel import loaders
from xibabel.loaders import (FDataObj, load_bids, load_nibabel, load, save,
                             PROCESSORS, _json_attrs2attrs, drop_suffix,
                             replace_suffix, _attrs2json_attrs, hdr2meta,
                             _path2class, XibFileError, to_nifti,
                             _ni_sort_expand_dims)
from xibabel.xutils import merge
from xibabel.testing import (JC_EG_FUNC, JC_EG_FUNC_JSON, JC_EG_ANAT,
                             JC_EG_ANAT_JSON, JH_EG_FUNC, skip_without_file,
                             fetcher)

import pytest

rng = np.random.default_rng()


class FakeProxy:

    def __init__(self, array, order=None):
        self._array = array
        self.order = order
        self.ndim = array.ndim
        self.shape = array.shape

    def __getitem__(self, slicer):
        return self._array[slicer]

    def __array__(self):
        return self[:]


def test_fdataobj_basic():
    arr = np.arange(24).reshape((2, 3, 4))
    proxy = FakeProxy(arr)
    fproxy = FDataObj(proxy)
    assert fproxy.ndim == arr.ndim
    assert fproxy.shape == arr.shape
    assert fproxy.order is None
    assert np.all(fproxy[1, 2, :] == arr[1, 2, :])
    assert arr[1, 2, :].dtype == np.arange(2).dtype
    assert fproxy.dtype == np.dtype(np.float64)
    assert fproxy[1, 2, :].dtype == np.dtype(np.float64)


def test_chunking(monkeypatch):
    arr_shape = 10, 20, 30
    arr = rng.normal(size=arr_shape)
    fproxy = FDataObj(FakeProxy(arr))
    f64s = np.dtype(float).itemsize
    monkeypatch.setattr(loaders, "MAXCHUNK_STRATEGY", lambda : arr.size * f64s)
    assert fproxy.chunk_sizes() == [None, None, None]
    monkeypatch.setattr(loaders, "MAXCHUNK_STRATEGY",
                        lambda : arr.size * f64s - 1)
    assert fproxy.chunk_sizes() == [9, None, None]
    c_fproxy = FDataObj(FakeProxy(arr, order='C'))
    assert c_fproxy.chunk_sizes() == [9, None, None]
    f_fproxy = FDataObj(FakeProxy(arr, order='F'))
    assert f_fproxy.chunk_sizes() == [None, None, 29]
    monkeypatch.setattr(loaders, "MAXCHUNK_STRATEGY",
                        lambda : arr.size * f64s / 2 - 1)
    assert fproxy.chunk_sizes() == [4, None, None]
    assert c_fproxy.chunk_sizes() == [4, None, None]
    assert f_fproxy.chunk_sizes() == [None, None, 14]
    monkeypatch.setattr(loaders, "MAXCHUNK_STRATEGY",
                        lambda : arr.size * f64s / 10 - 1)
    assert fproxy.chunk_sizes() == [1, 19, None]
    assert c_fproxy.chunk_sizes() == [1, 19, None]
    assert f_fproxy.chunk_sizes() == [None, None, 2]
    monkeypatch.setattr(loaders, "MAXCHUNK_STRATEGY",
                        lambda : arr.size * f64s / 30 - 1)
    assert fproxy.chunk_sizes() == [1, 6, None]
    assert c_fproxy.chunk_sizes() == [1, 6, None]
    assert f_fproxy.chunk_sizes() == [None, 19, 1]


def out_back(img, out_path):
    if out_path.is_file():
        os.unlink(out_path)
    nib.save(img, out_path)
    img = nib.load(out_path)
    return img, hdr2meta(img.header)


def test_nibabel_tr(tmp_path):
    # Default image.
    arr = np.zeros((2, 3, 4))
    img = nib.Nifti1Image(arr, np.eye(4), None)
    out_path = tmp_path / 'test.nii'
    back_img, meta = out_back(img, out_path)
    exp_meta = {'xib-affines': {'aligned': np.eye(4).tolist()}}
    assert meta == exp_meta
    arr = np.zeros((2, 3, 4, 6))
    img = nib.Nifti1Image(arr, np.eye(4), None)
    back_img, meta = out_back(img, out_path)
    assert meta == exp_meta
    img.header.set_xyzt_units('mm', 'sec')
    back_img, meta = out_back(img, out_path)
    exp_meta.update({'RepetitionTime': 1.0})
    assert meta == exp_meta
    img.header.set_xyzt_units('mm', 'msec')
    back_img, meta = out_back(img, out_path)
    assert meta['RepetitionTime'] == 1 / 1000
    img.header.set_xyzt_units('mm', 'usec')
    back_img, meta = out_back(img, out_path)
    assert meta['RepetitionTime'] == 1 / 1_000_000


def test_nibabel_slice_timing(tmp_path):
    # Default image.
    arr = np.zeros((2, 3, 4, 5))
    img = nib.Nifti1Image(arr, np.eye(4), None)
    out_path = tmp_path / 'test.nii'
    back_img, meta = out_back(img, out_path)
    exp_meta = {'xib-affines': {'aligned': np.eye(4).tolist()}}
    assert meta == exp_meta
    img.header.set_dim_info(None, None, 1)
    back_img, meta = out_back(img, out_path)
    assert meta == merge(exp_meta, {'SliceEncodingDirection': 'j'})
    img.header.set_dim_info(1, 0, 2)
    back_img, meta = out_back(img, out_path)
    exp_dim = merge(exp_meta,
                    {'PhaseEncodingDirection': 'i',
                     'xib-FrequencyEncodingDirection': 'j',
                     'SliceEncodingDirection': 'k'})
    assert meta == exp_dim
    img.header.set_slice_duration(1 / 4)
    back_img, meta = out_back(img, out_path)
    assert meta == exp_dim
    img.header['slice_start'] = 0
    back_img, meta = out_back(img, out_path)
    assert meta == exp_dim
    img.header['slice_end'] = 3
    back_img, meta = out_back(img, out_path)
    assert meta == exp_dim
    img.header['slice_code'] = 4  # NIFTI_SLICE_ALT_DEC
    back_img, meta = out_back(img, out_path)
    exp_timed = exp_dim.copy()
    exp_timed['SliceTiming'] = [0.75, 0.25, 0.5, 0]
    assert meta == exp_timed


def test_guess_format():
    root = Path('foo') / 'bar' / 'baz.suff'
    for v, exp in ((root, None),
                   (root.with_suffix('.nii'), None),
                   (root.with_suffix('.json'), 'bids'),
                   (root.with_suffix('.ximg'), 'zarr'),
                   (root.with_suffix('.nc'), 'netcdf'),
                   (root.with_suffix('.foo'), None)):

        assert PROCESSORS.guess_format(v) == exp
        assert PROCESSORS.guess_format(str(v)) == exp


def test__path2class():
    for url, exp_class in (
        ('/foo/bar/sub-07_T1w.nii.gz', nib.Nifti1Image),
        ('http://localhost:8999/sub-07_T1w.nii.gz', nib.Nifti1Image),
        ('/foo/bar/sub-07_T1w.nii', nib.Nifti1Image),
        ('http://localhost:8999/sub-07_T1w.nii', nib.Nifti1Image),
        ('/foo/bar/sub-07_T1w.mnc', nib.Minc1Image),
        ('http://localhost:8999/sub-07_T1w.mnc', nib.Minc1Image),
    ):
        assert _path2class(url) == exp_class


def test_drop_suffix():
    for inp, suffixes, exp_out in (
        ('foo/bar', ['.nii'], 'foo/bar'),
        ('foo/bar', '.nii', 'foo/bar'),
        ('foo/bar.baz', ['.nii'], 'foo/bar.baz'),
        ('foo/bar.nii', ['.nii'], 'foo/bar'),
        ('foo/bar.nii', '.nii', 'foo/bar'),
        ('foo/bar.nii.gz', ['.nii'], 'foo/bar.nii.gz'),
        ('foo/bar.nii.gz', ['.nii.gz', '.nii'], 'foo/bar'),
    ):
        assert drop_suffix(inp, suffixes) == exp_out
        assert drop_suffix(Path(inp), suffixes) == Path(exp_out)


def test_replace_suffix():
    for inp, suffixes, new_suffix, exp_out in (
        ('foo/bar', ['.nii'], '.json', 'foo/bar.json'),
        ('foo/bar', '.nii', '.json', 'foo/bar.json'),
        ('foo/bar.baz', ['.nii'], '.boo', 'foo/bar.boo'),
        ('foo/bar.nii', ['.nii'], '.boo', 'foo/bar.boo'),
        ('foo/bar.nii', '.nii', '.boo', 'foo/bar.boo'),
        ('foo/bar.nii.gz', ['.nii'], '.boo', 'foo/bar.nii.boo'),
        ('foo/bar.nii.gz', ['.nii.gz', '.nii'], '.boo', 'foo/bar.boo'),
    ):
        assert replace_suffix(inp, suffixes, new_suffix) == exp_out
        assert replace_suffix(Path(inp), suffixes, new_suffix) == Path(exp_out)


def test_json_attrs():
    # Test utilities to load / save JSON attrs
    d = {'foo': 1, 'bar': [2, 3]}
    assert _attrs2json_attrs(d) == d
    assert _json_attrs2attrs(d) == d
    dd = {'foo': 1, 'bar': {'baz': 4}}
    ddj = {'foo': 1, 'bar': ['__json__', '{"baz": 4}']}
    assert _attrs2json_attrs(dd) == ddj
    assert _json_attrs2attrs(ddj) == dd
    arr = rng.integers(0, 10, size=(3, 4)).tolist()
    arr_j = json.dumps(arr)
    dd = {'foo': 1, 'bar': {'baz': [2, 3]}, 'baf': arr}
    ddj = {'foo': 1,
           'bar': ['__json__', '{"baz": [2, 3]}'],
           'baf': ['__json__', arr_j]}
    assert _attrs2json_attrs(dd) == ddj
    assert _json_attrs2attrs(ddj) == dd


@skip_without_file(JC_EG_FUNC)
def test_nib_loader_jc():
    img = nib.load(JC_EG_FUNC)
    ximg = load_nibabel(JC_EG_FUNC)
    assert ximg.attrs == JC_EG_FUNC_META
    assert np.all(np.array(ximg) == img.get_fdata())


@skip_without_file(JH_EG_FUNC)
def test_nib_loader_jh():
    img = nib.load(JH_EG_FUNC)
    ximg = load_nibabel(JH_EG_FUNC)
    assert ximg.attrs == {'RepetitionTime': 2.5,
                          'xib-affines':
                          {'scanner': img.affine.tolist()}
                         }


if fetcher.have_file(JC_EG_FUNC):
    img = nib.load(JC_EG_FUNC)
    JC_EG_FUNC_META = json.loads(JC_EG_FUNC_JSON.read_text())
    JC_EG_FUNC_META_RAW = {
        'xib-FrequencyEncodingDirection': 'i',
         'PhaseEncodingDirection': 'j',
         'SliceEncodingDirection': 'k',
         'RepetitionTime': 2.0,
         'xib-affines':
         {'scanner': img.affine.tolist()}
    }
    JC_EG_FUNC_META.update(JC_EG_FUNC_META_RAW)


if fetcher.have_file(JC_EG_ANAT):
    img = nib.load(JC_EG_ANAT)
    JC_EG_ANAT_META = json.loads(JC_EG_ANAT_JSON.read_text())
    JC_EG_ANAT_META_RAW = {
        'xib-FrequencyEncodingDirection': 'j',
         'PhaseEncodingDirection': 'i',
         'SliceEncodingDirection': 'k',
         'xib-affines':
         {'scanner': img.affine.tolist()}
    }
    JC_EG_ANAT_META.update(JC_EG_ANAT_META_RAW)


@skip_without_file(JC_EG_ANAT)
def test_anat_loader():
    img = nib.load(JC_EG_ANAT)
    for loader, in_path in product(
        (load, load_bids, load_nibabel),
        (JC_EG_ANAT, str(JC_EG_ANAT),
         JC_EG_ANAT_JSON, str(JC_EG_ANAT_JSON))):
        ximg = loader(in_path)
        assert ximg.shape == (176, 256, 256)
        assert ximg.name == JC_EG_ANAT.name.split('.')[0]
        assert ximg.attrs == JC_EG_ANAT_META
        assert np.all(np.array(ximg) == img.get_fdata())


@skip_without_file(JC_EG_ANAT)
def test_anat_loader_http(fserver):
    nb_img = nib.load(JC_EG_ANAT)
    # Read nibabel from HTTP
    # Original gz
    name_gz = JC_EG_ANAT.name
    # Uncompressed, no gz
    name_no_gz = JC_EG_ANAT.with_suffix('').name
    out_path = fserver.server_path / name_no_gz
    with gzip.open(JC_EG_ANAT, 'rb') as f:
        out_path.write_bytes(f.read())
    for name in (name_gz, name_no_gz):
        out_url = fserver.make_url(name)
        ximg = load(out_url)
        # Check we can read the data
        ximg.compute()
        # Check parameters
        assert ximg.shape == (176, 256, 256)
        assert ximg.name == JC_EG_ANAT.name.split('.')[0]
        assert ximg.attrs == JC_EG_ANAT_META
        assert np.all(np.array(ximg) == nb_img.get_fdata())


@skip_without_file(JC_EG_ANAT)
def test_anat_loader_http_params(fserver, tmp_path):
    # Test params get passed through in kwargs.
    nb_img = nib.load(JC_EG_ANAT)
    out_url = 'simplecache::' + fserver.make_url(JC_EG_ANAT.name)
    out_cache = tmp_path / 'files'
    assert not out_cache.is_dir()
    ximg = load(out_url,
                simplecache={'cache_storage': str(out_cache)})
    assert np.all(np.array(ximg) == nb_img.get_fdata())
    assert out_cache.is_dir()
    assert len(list(out_cache.glob('*'))) == 2  # JSON and Nifti


@skip_without_file(JC_EG_ANAT)
def test_round_trip(tmp_path):
    ximg = load(JC_EG_ANAT)
    assert ximg.shape == (176, 256, 256)
    out_path = tmp_path / 'out.ximg'
    save(ximg, out_path)
    back = load(out_path)
    assert back.shape == (176, 256, 256)
    assert back.attrs == JC_EG_ANAT_META
    # And again
    save(ximg, out_path)
    back = load(out_path)
    assert back.attrs == JC_EG_ANAT_META
    # With url
    back = load(f'file:///{out_path}')
    assert back.attrs == JC_EG_ANAT_META


@pytest.mark.skipif(not find_spec('h5netcdf'),
                    reason='Need h5netcdf module for test')
@skip_without_file(JC_EG_ANAT)
def test_round_trip_netcdf(tmp_path):
    ximg = load(JC_EG_ANAT)
    out_path = tmp_path / 'out.nc'
    save(ximg, out_path)
    back = load(out_path)
    assert back.shape == (176, 256, 256)
    assert back.attrs == JC_EG_ANAT_META
    back = load(f'file:///{out_path}')
    assert back.attrs == JC_EG_ANAT_META


def test_tornado(fserver):
    # Test static file server for URL reads
    fserver.write_text_to('text_file', 'some text')
    fserver.write_bytes_to('binary_file', b'binary')
    response = fserver.get('text_file')
    assert response.status_code == 200
    assert response.text == 'some text'
    assert fserver.read_text('text_file') == 'some text'
    assert fserver.read_bytes('text_file') == b'some text'
    assert fserver.read_bytes('binary_file') == b'binary'


@pytest.mark.skipif(not find_spec('h5netcdf'),
                    reason='Need h5netcdf module for test')
@skip_without_file(JC_EG_ANAT)
def test_round_trip_netcdf_url(fserver):
    ximg = load(JC_EG_ANAT)
    save(ximg, fserver.server_path / 'out.nc')
    out_url = fserver.make_url('out.nc')
    back = load(out_url)
    assert back.shape == (176, 256, 256)
    assert back.attrs == JC_EG_ANAT_META


@skip_without_file(JC_EG_ANAT)
def test_matching_img_error(tmp_path):
    out_json = tmp_path / JC_EG_ANAT_JSON.name
    with pytest.raises(XibFileError, match='does not appear to exist'):
        load(out_json)
    shutil.copy2(JC_EG_ANAT_JSON, tmp_path)
    with pytest.raises(XibFileError, match='No valid file matching'):
        load(out_json)
    out_img = tmp_path / JC_EG_ANAT.name
    shutil.copy2(JC_EG_ANAT, tmp_path)
    back = load(out_img)
    assert back.shape == (176, 256, 256)
    assert back.attrs == JC_EG_ANAT_META
    os.unlink(out_img)
    with pytest.raises(XibFileError, match='does not appear to exist'):
        load(out_img)
    shutil.copy2(JC_EG_ANAT, tmp_path)
    os.unlink(out_json)
    back = load(out_img)
    assert back.attrs == JC_EG_ANAT_META_RAW
    back = load_bids(out_img, require_json=False)
    assert back.attrs == JC_EG_ANAT_META_RAW
    with pytest.raises(XibFileError, match='`require_json` is True'):
        load_bids(out_img, require_json=True)


def test_ni_sort_expand_dims():
    assert _ni_sort_expand_dims([]) == ([],
                                        ['i', 'j', 'k'],
                                        [0, 1, 2])
    assert _ni_sort_expand_dims(['time']) == (['time'],
                                              ['i', 'j', 'k'],
                                              [0, 1, 2])
    assert (_ni_sort_expand_dims(['j']) ==
            (['j'],
             ['i', 'k'],
             [0, 2]))
    assert (_ni_sort_expand_dims(['time', 'j']) ==
            (['j', 'time'],
             ['i', 'k'],
             [0, 2]))
    assert (_ni_sort_expand_dims(['j', 'i']) ==
            (['i', 'j'],
             ['k'],
             [2]))
    assert (_ni_sort_expand_dims(['time', 'j', 'k']) ==
            (['j', 'k', 'time'],
             ['i'],
             [0]))


@skip_without_file(JC_EG_ANAT)
@skip_without_file(JC_EG_FUNC)
def test_to_nifti():
    for pth in (JC_EG_ANAT, JC_EG_FUNC):
        orig_img = nib.load(JC_EG_ANAT)
        orig_data = orig_img.get_fdata()
        ximg = load(JC_EG_ANAT)
        # Check data is the same for basic load.
        assert np.all(ximg == orig_data)
        # Basic conversion.
        img = to_nifti(ximg)
        assert np.all(img.get_fdata() == orig_data)
        # assert wrap_header(img.header).to_meta() == JC_EG_ANAT_META_RAW
        img = to_nifti(ximg.T)
        assert np.all(img.get_fdata() == orig_data)
        # assert wrap_header(img.header).to_meta() == JC_EG_ANAT_META_RAW
        img = to_nifti(ximg.T.sel(k=32))  # Drop k axis
        assert np.all(img.get_fdata() == orig_data[:, :, 32:33])
        # assert wrap_header(img.header).to_meta() == JC_EG_ANAT_META_RAW
