"""Common OpenCL related functionality.
"""

from __future__ import print_function
import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: 401
from collections import defaultdict
from operator import itemgetter

from .config import get_config

_ctx = None
_queue = None
_profile_info = defaultdict(float)


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
        properties = None
        if get_config().profile:
            properties = cl.command_queue_properties.PROFILING_ENABLE
        _queue = cl.CommandQueue(get_context(), properties=properties)
    return _queue


def set_queue(q):
    global _queue
    _queue = q


def profile(name, event):
    global _profile_info
    event.wait()
    time = (event.profile.end - event.profile.start) * 1e-9
    _profile_info[name] += time


def print_profile():
    global _profile_info
    _profile_info = sorted(_profile_info.items(), key=itemgetter(1),
                           reverse=True)
    if len(_profile_info) == 0:
        print("No profile information available")
        return
    print("{:<30} {:<30}".format('Kernel', 'Time'))
    tot_time = 0
    for kernel, time in _profile_info:
        print("{:<30} {:<30}".format(kernel, time))
        tot_time += time
    print("Total profiled time: %g secs" % tot_time)


def profile_kernel(kernel, name):
    def _profile_knl(*args):
        event = kernel(*args)
        profile(name, event)
        return event
    if get_config().profile:
        return _profile_knl
    else:
        return kernel


class DeviceArray(object):
    def __init__(self, dtype, n=0):
        self.queue = get_queue()
        length = n
        if n == 0:
            n = 16
        data = cl.array.empty(self.queue, n, dtype)
        self.set_data(data)
        self.length = length
        self._update_array_ref()

    def _update_array_ref(self):
        self.array = self._data[:self.length]

    def resize(self, size):
        self.reserve(size)
        self.length = size
        self._update_array_ref()

    def reserve(self, size):
        if size > self.alloc:
            new_data = cl.array.empty(self.queue, size, self.dtype)
            new_data[:self.alloc] = self._data
            self._data = new_data
            self.alloc = size
            self._update_array_ref()

    def set_data(self, data):
        self._data = data
        self.length = data.size
        self.alloc = data.size
        self.dtype = data.dtype
        self._update_array_ref()

    def get_data(self):
        return self._data

    def copy(self):
        arr_copy = DeviceArray(self.dtype)
        arr_copy.set_data(self.array.copy())
        return arr_copy

    def fill(self, value):
        self.array.fill(value)


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

        for prop, ary in pa.properties.items():
            self.add_prop(prop, ary)
        for prop, ary in pa.constants.items():
            self.add_prop(prop, ary)

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
        g_ary = DeviceArray(np_array.dtype, n=carray.length)
        g_ary.array.set(np_array)
        self._data[name] = g_ary
        setattr(self, name, g_ary.array)
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
        for prop in self._props:
            self._data[prop].resize(new_size)
            setattr(self, prop, self._data[prop].array)
