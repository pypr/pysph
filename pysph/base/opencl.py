"""Common OpenCL related functionality.
"""

import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: 401

from .config import get_config

_ctx = None
_queue = None


def get_context():
    global _ctx
    if _ctx is None:
        _ctx = cl.create_some_context()
    return _ctx


def set_context(ctx):
    global _ctx
    _ctx = ctx


def get_queue():
    global _queue
    if _queue is None:
        _queue = cl.CommandQueue(get_context())
    return _queue


def set_queue(q):
    global _queue
    _queue = q


class DeviceHelper(object):
    def __init__(self, particle_array):
        self._particle_array = pa = particle_array
        self._queue = q = get_queue()
        self._props = []
        use_double = get_config().use_double
        self._dtype = np.float64 if use_double else np.float32

        for prop, ary in pa.properties.items():
            a_gpu = cl.array.to_device(q, self._get_array(ary))
            setattr(self, prop, a_gpu)
            self._props.append(prop)
        for prop, ary in pa.constants.items():
            a_gpu = cl.array.to_device(q, self._get_array(ary))
            setattr(self, prop, a_gpu)
            self._props.append(prop)

    def _get_array(self, ary):
        ctype = ary.get_c_type()
        if ctype in ['float', 'double']:
            return ary.get_npy_array().astype(self._dtype)
        else:
            return ary.get_npy_array()

    def _get_prop_or_const(self, prop):
        pa = self._particle_array
        return pa.properties.get(prop, pa.constants.get(prop))

    def push(self, *args):
        if len(args) == 0:
            args = self._props
        for arg in args:
            getattr(self, arg).set(
                self._get_array(self._get_prop_or_const(arg))
            )

    def pull(self, *args):
        if len(args) == 0:
            args = self._props
        for arg in args:
            self._get_prop_or_const(arg).set_data(
                getattr(self, arg).get()
            )
