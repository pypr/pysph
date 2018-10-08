from unittest import TestCase
import pytest
import numpy as np

from pysph.cpy.config import get_config
from pysph.cpy.array import Array
import pysph.cpy.array as array


test_all_backends = pytest.mark.parametrize('backend',
                                            ['cython', 'opencl', 'cuda'])


def make_dev_array(backend, n=16):
    dev_array = Array(np.int32, n=n, backend=backend)
    dev_array.fill(0)
    dev_array[0] = 1
    return dev_array


def check_import(backend):
    if backend == 'opencl':
        pytest.importorskip('pyopencl')
    if backend == 'cuda':
        pytest.importorskip('pycuda')


@test_all_backends
def test_reserve(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)

    # When
    dev_array.reserve(64)

    # Then
    assert len(dev_array.get_data()) == 64
    assert dev_array.length == 16
    assert dev_array[0] == 1


@test_all_backends
def test_resize_with_reallocation(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)

    # When
    dev_array.resize(64)

    # Then
    assert len(dev_array.get_data()) == 64
    assert dev_array.length == 64
    assert dev_array[0] == 1


@test_all_backends
def test_resize_without_reallocation(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend, n=128)

    # When
    dev_array.resize(64)

    # Then
    assert len(dev_array.get_data()) == 128
    assert dev_array.length == 64
    assert dev_array[0] == 1


@test_all_backends
def test_copy(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)

    # When
    dev_array_copy = dev_array.copy()

    # Then
    print(dev_array.dev, dev_array_copy.dev)
    assert np.all(dev_array.get() == dev_array_copy.get())

    dev_array_copy[0] = 2
    assert dev_array[0] != dev_array_copy[0]


@test_all_backends
def test_append_with_reallocation(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)

    # When
    dev_array.append(2)

    # Then
    assert dev_array[-1] == 2
    assert len(dev_array.get_data()) == 32


@test_all_backends
def test_append_without_reallocation(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)
    dev_array.reserve(20)

    # When
    dev_array.append(2)

    # Then
    assert dev_array[-1] == 2
    assert len(dev_array.get_data()) == 20


@test_all_backends
def test_extend(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)
    new_array = 2 + array.zeros(64, dtype=np.int32, backend=backend)

    # When
    dev_array.extend(new_array)

    # Then
    old_nparr = dev_array.get()
    new_nparr = new_array.get()
    assert np.all(old_nparr[-len(new_array)] == new_nparr)


@test_all_backends
def test_remove(backend):
    check_import(backend)

    # Given
    dev_array = Array(np.int32, backend=backend)
    orig_array = array.arange(0, 16, 1, dtype=np.int32,
                              backend=backend)
    dev_array.set_data(orig_array)
    indices = array.arange(0, 8, 1, dtype=np.int32, backend=backend)

    # When
    dev_array.remove(indices)

    # Then
    assert np.all(dev_array.get() == (8 + indices).get())


@test_all_backends
def test_align(backend):
    check_import(backend)

    # Given
    dev_array = Array(np.int32, backend=backend)
    orig_array = array.arange(0, 16, 1, dtype=np.int32, backend=backend)
    dev_array.set_data(orig_array)
    indices = array.arange(15, -1, -1, dtype=np.int32, backend=backend)

    # When
    dev_array.align(indices)

    # Then
    assert np.all(dev_array.get() == indices.get())


@test_all_backends
def test_squeeze(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)
    dev_array.fill(2)
    dev_array.reserve(32)
    assert dev_array.alloc == 32

    # When
    dev_array.squeeze()

    # Then
    assert dev_array.alloc == 16


@test_all_backends
def test_copy_values(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)
    dev_array.fill(2)

    dest = array.empty(8, dtype=np.int32, backend=backend)
    indices = array.arange(0, 8, 1, dtype=np.int32, backend=backend)

    # When
    dev_array.copy_values(indices, dest)

    # Then
    assert np.all(dev_array[:len(indices)].get() == dest.get())


@test_all_backends
def test_min_max(backend):
    check_import(backend)

    # Given
    dev_array = make_dev_array(backend)
    dev_array.fill(2)
    dev_array[0], dev_array[1] = 1, 10

    # When
    dev_array.update_min_max()

    # Then
    assert dev_array.minimum == 1
    assert dev_array.maximum == 10
