import unittest
import numpy as np

from pytest import importorskip

from ..config import use_config
from ..array import wrap
from ..types import annotate, declare
from ..low_level import (
    Cython, Kernel, LocalMem, local_barrier, GID_0, LDIM_0, LID_0,
    nogil, prange, parallel
)


class TestKernel(unittest.TestCase):
    def test_simple_kernel_opencl(self):
        importorskip('pyopencl')

        # Given
        @annotate(gdoublep='x, y', a='float', size='int')
        def knl(x, y, a, size):
            i = declare('int')
            i = GID_0*LDIM_0 + LID_0
            if i < size:
                y[i] = x[i]*a

        x = np.linspace(0, 1, 1000)
        y = np.zeros_like(x)
        x, y = wrap(x, y, backend='opencl')

        # When
        k = Kernel(knl, backend='opencl')
        a = 21.0
        k(x, y, a, 1000)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, x.data * a))

    def test_simple_kernel_cuda(self):
        importorskip('pycuda')

        # Given
        @annotate(gdoublep='x, y', a='float', size='int')
        def knl(x, y, a, size):
            i = declare('int')
            i = GID_0*LDIM_0 + LID_0
            if i < size:
                y[i] = x[i]*a

        x = np.linspace(0, 1, 1000)
        y = np.zeros_like(x)
        x, y = wrap(x, y, backend='cuda')

        # When
        k = Kernel(knl, backend='cuda')
        a = 21.0
        k(x, y, a, 1000)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, x.data * a))

    def test_kernel_with_local_memory_opencl(self):
        importorskip('pyopencl')

        # Given
        @annotate(gdoublep='x, y', xc='ldoublep', a='float')
        def knl(x, y, xc, a):
            i, lid = declare('int', 2)
            lid = LID_0
            i = GID_0 * LDIM_0 + lid

            xc[lid] = x[i]

            local_barrier()

            y[i] = xc[lid] * a

        x = np.linspace(0, 1, 1024)
        y = np.zeros_like(x)
        xc = LocalMem(1, backend='opencl')

        x, y = wrap(x, y, backend='opencl')

        # When
        k = Kernel(knl, backend='opencl')
        a = 21.0
        k(x, y, xc, a)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, x.data * a))

    def test_kernel_with_local_memory_cuda(self):
        importorskip('pycuda')

        # Given
        @annotate(gdoublep='x, y', xc='ldoublep', a='float')
        def knl(x, y, xc, a):
            i, lid = declare('int', 2)
            lid = LID_0
            i = GID_0 * LDIM_0 + lid

            xc[lid] = x[i]

            local_barrier()

            y[i] = xc[lid] * a

        x = np.linspace(0, 1, 1024)
        y = np.zeros_like(x)
        xc = LocalMem(1, backend='cuda')

        x, y = wrap(x, y, backend='cuda')

        # When
        k = Kernel(knl, backend='cuda')
        a = 21.0
        k(x, y, xc, a)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, x.data * a))


@annotate(double='x, y, a', return_='double')
def func(x, y, a):
    return x * y * a


@annotate(doublep='x, y', a='double', n='int', return_='double')
def knl(x, y, a, n):
    i = declare('int')
    s = declare('double')
    s = 0.0
    for i in range(n):
        s += func(x[i], y[i], a)
    return s


@annotate(n='int', doublep='x, y', a='double')
def cy_extern(x, y, a, n):
    i = declare('int')
    with nogil, parallel():
        for i in prange(n):
            y[i] = x[i] * a


class TestCython(unittest.TestCase):
    def test_cython_code_with_return_and_nested_call(self):
        # Given
        n = 1000
        x = np.linspace(0, 1, n)
        y = x.copy()
        a = 2.0

        # When
        cy = Cython(knl)
        result = cy(x, y, a, n)

        # Then
        self.assertAlmostEqual(result, np.sum(x * y * a))

    def test_cython_with_externs(self):
        # Given
        n = 1000
        x = np.linspace(0, 1, n)
        y = np.zeros_like(x)
        a = 2.0

        # When
        with use_config(use_openmp=True):
            cy = Cython(cy_extern)

        cy(x, y, a, n)

        # Then
        self.assertTrue(np.allclose(y, x * a))
