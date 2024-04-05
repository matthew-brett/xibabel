""" Testing testing (... testing).
"""

import numpy as np

import xibabel.testing as xit


def test_assert_dict_allclose():
    assert xit.arr_dict_allclose({}, {})
    assert not xit.arr_dict_allclose({}, 1)
    assert xit.arr_dict_allclose({'foo': 1}, {'foo': 1})
    assert not xit.arr_dict_allclose({'foo': 1}, {'bar': 1})
    assert not xit.arr_dict_allclose({'foo': 1}, {'foo': 2})
    assert not xit.arr_dict_allclose({'foo': 1}, {'foo': 1, 'bar': 2})
    assert xit.arr_dict_allclose({'foo': np.array([1, 2, 3])},
                                {'foo': np.array([1, 2, 3])})
    assert not xit.arr_dict_allclose({'foo': np.array([1, 2, 3])},
                                    {'bar': np.array([2, 2, 3])})
    # Test atol
    assert xit.arr_dict_allclose({'foo': np.array([1, 2, 0])},
                                {'foo': np.array([1, 2, 1.1e-9])})
    assert not xit.arr_dict_allclose({'foo': np.array([1, 2, 0])},
                                    {'foo': np.array([1, 2, 1.1e-8])})
    assert xit.arr_dict_allclose({'foo': np.array([1, 2, 0])},
                                {'foo': np.array([1, 2, 1.1e-8])},
                                atol=1e-7)
    # Also works with lists.
    assert xit.arr_dict_allclose({'foo': [1, 2, 0]},
                                 {'foo': [1, 2, 1.1e-9]})
    # Test rtol
    assert xit.arr_dict_allclose({'foo': np.array([1, 2, 10])},
                                {'foo': np.array([1, 2, 10 + 1.1e-5])})
    assert not xit.arr_dict_allclose({'foo': np.array([1, 2, 10])},
                                    {'foo': np.array([1, 2, 10 + 1.1e-4])})
    assert xit.arr_dict_allclose({'foo': np.array([1, 2, 10])},
                                {'foo': np.array([1, 2, 10 + 1.1e-4])},
                                rtol=1e-3)
    # Recursive.
    assert xit.arr_dict_allclose({'foo': {'bar': np.array([1, 2, 3])}},
                                {'foo': {'bar': np.array([1, 2, 3])}})
    assert not xit.arr_dict_allclose({'foo': {'bar': np.array([1, 2, 3])}},
                                    {'foo': {'bar': np.array([1, 2, 3.1])}})

    # Different types
    assert not xit.arr_dict_allclose({'foo': ['bar', 'boo', 'baz']},
                                     {'foo': np.array([1, 2, 3.1])})
