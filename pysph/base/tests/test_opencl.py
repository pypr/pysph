import unittest
from pytest import importorskip
import numpy as np


class DeviceArrayTestCase(unittest.TestCase):
    def setUp(self):
        cl = importorskip("pyopencl")
        from pysph.base.opencl import DeviceArray
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.dev_array = DeviceArray(self.queue, np.int32)

    def tearDown(self):
        cl = importorskip("pyopencl")
        from pysph.base.opencl import DeviceArray
        self.dev_array = DeviceArray(self.queue, np.int32)

    def test_reserve(self):
        self.dev_array.reserve(64)
        assert len(self.dev_array.get_array()) == 64
        assert self.dev_array.length == 0

    def test_resize_with_reallocation(self):
        self.dev_array.resize(64)
        assert len(self.dev_array.get_array()) == 64
        assert self.dev_array.length == 64

    def test_resize_without_reallocation(self):
        cl = importorskip("pyopencl")
        from pysph.base.opencl import DeviceArray
        self.dev_array = DeviceArray(self.queue, np.int32, n=128)

        self.dev_array.resize(64)
        assert len(self.dev_array.get_array()) == 128
        assert self.dev_array.length == 64

    def test_copy(self):
        dev_array_copy = self.dev_array.copy()
        assert np.all(self.dev_array.data.get() == dev_array_copy.data.get())

