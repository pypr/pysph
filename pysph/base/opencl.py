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
    """Manages the arrays contained in a particle array on the device.

    Note that it converts the data to a suitable type depending on the value of
    get_config().use_double. Further, note that it assumes that the names of
    constants and properties do not clash.

    """
    def __init__(self, particle_array):
        self._particle_array = pa = particle_array
        self._queue = get_queue()
        use_double = get_config().use_double
        self._dtype = np.float64 if use_double else np.float32
        self._data = {}
        self._props = []
        self._alloc = 0

        for prop, ary in pa.properties.items():
            self.add_prop(prop, ary)
        for prop, ary in pa.constants.items():
            self.add_prop(prop, ary)
        if self._data:
            self._alloc = len(self._data['x'])

    def _get_array(self, ary):
        ctype = ary.get_c_type()
        if ctype in ['float', 'double']:
            return ary.get_npy_array().astype(self._dtype)
        else:
            return ary.get_npy_array()

    def _get_prop_or_const(self, prop):
        pa = self._particle_array
        return pa.properties.get(prop, pa.constants.get(prop))

    def add_prop(self, name, carray):
        """Add a new property or constant given the name and carray, note
        that this assumes that this property is already added to the
        particle array.
        """
        np_array = self._get_array(carray)
        g_ary = cl.array.empty(self._queue, carray.alloc, np_array.dtype)
        view = g_ary[:carray.length]
        view.set(np_array)
        self._data[name] = g_ary
        setattr(self, name, view)
        if name in self._particle_array.properties:
            self._props.append(name)

    def max(self, arg):
        return float(cl.array.max(getattr(self, arg)).get())

    def pull(self, *args):
        if len(args) == 0:
            args = self._data.keys()
        for arg in args:
            self._get_prop_or_const(arg).set_data(
                getattr(self, arg).get()
            )

    def push(self, *args):
        if len(args) == 0:
            args = self._data.keys()
        for arg in args:
            getattr(self, arg).set(
                self._get_array(self._get_prop_or_const(arg))
            )

    def remove_prop(self, name):
        if name in self._props:
            self._props.remove(name)
        if name in self._data:
            del self._data[name]
            delattr(self, name)

    def resize(self, new_size):
        if new_size > self._alloc:
            for prop in self._props:
                old_prop = self._data[prop]
                new_prop = cl.array.empty(
                    self._queue, new_size, dtype=old_prop.dtype
                )
                sz = min(len(old_prop), new_size)
                new_prop[:sz] = old_prop[:sz]
                self._data[prop] = new_prop
            self._alloc = new_size

        for prop in self._props:
            setattr(self, prop, self._data[prop][:new_size])
