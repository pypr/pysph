import unittest
import numpy as np

from pytest import importorskip

from ..array import wrap
from ..types import annotate, declare, GID_0, LDIM_0, LID_0
from ..low_level import Kernel, LocalMem, local_barrier


class TestKernel(unittest.TestCase):

    def setUp(self):
        importorskip('pyopencl')

    def test_simple_kernel(self):
        # Given
        @annotate(gdoublep='x, y', a='float')
        def knl(x, y, a):
            i = declare('int')
            i = GID_0*LDIM_0 + LID_0
            y[i] = x[i]*a

        x = np.linspace(0, 1, 1000)
        y = np.zeros_like(x)
        x, y = wrap(x, y, backend='opencl')

        # When
        k = Kernel(knl, backend='opencl')
        a = 21.0
        k(x, y, a)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, x.data*a))

    def test_kernel_with_local_memory(self):
        # Given
        @annotate(gdoublep='x, y', xc='ldoublep', a='float')
        def knl(x, y, xc, a):
            i, lid = declare('int', 2)
            lid = LID_0
            i = GID_0*LDIM_0 + lid

            xc[lid] = x[i]

            local_barrier()

            y[i] = xc[lid]*a

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
        self.assertTrue(np.allclose(y.data, x.data*a))
