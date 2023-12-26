""" Test loaders
"""

import numpy as np

from xibabel import loaders
from xibabel.loaders import FDataObj

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
