"""Common OpenCL related functionality.
"""

from __future__ import print_function
import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: 401
import pyopencl.algorithm
import pyopencl.tools
from pyopencl.scan import GenericScanKernel
from pyopencl.elementwise import ElementwiseKernel
from collections import defaultdict
from operator import itemgetter
from mako.template import Template

import logging
logger = logging.getLogger()

from .config import get_config

_ctx = None
_queue = None
_profile_info = defaultdict(float)


# args: uint* indices, dtype array, int length
REMOVE_KNL = Template(r"""//CL//
        unsigned int idx = indices[n - 1 - i];
        array[idx] = array[length - 1 - i];
""", disable_unicode=True)


# args: tag_array, tag, indices, head
REMOVE_INDICES_KNL = Template(r"""//CL//
        if(tag_array[i] == tag)
            indices[atomic_inc(&head[0])] = i;
""", disable_unicode=True)


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


def get_elwise_kernel(kernel_name, args, src, preamble=""):
    ctx = get_context()
    knl = ElementwiseKernel(
        ctx, args, src,
        kernel_name, preamble=preamble
    )
    return profile_kernel(knl, kernel_name)


class DeviceArray(object):
    def __init__(self, dtype, n=0):
        self.queue = get_queue()
        self.ctx = get_context()
        length = n
        if n == 0:
            n = 16
        data = cl.array.empty(self.queue, n, dtype)
        self.minimum = 0
        self.maximum = 0
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

    def update_min_max(self):
        self.minimum = cl.array.min(self.array).get()
        self.maximum = cl.array.max(self.array).get()

    def fill(self, value):
        self.array.fill(value)

    def append(self, value):
        if self.length >= self.alloc:
            self.reserve(2 * self.length)
        self._data[self.length] = value
        self.length += 1
        self._update_array_ref()

    def extend(self, cl_arr):
        if self.length + len(cl_arr) > self.alloc:
            self.reserve(self.length + len(cl_arr))
        self._data[-len(cl_arr):] = cl_arr
        self.length += len(cl_arr)
        self._update_array_ref()

    def remove(self, indices, input_sorted=False):
        if len(indices) > self.length:
            return

        if not input_sorted:
            radix_sort = cl.algorithm.RadixSort(
                self.ctx,
                "unsigned int* indices",
                scan_kernel=GenericScanKernel, key_expr="indices[i]",
                sort_arg_names=["indices"]
                )

            (sorted_indices,), event = radix_sort(indices)

        else:
            sorted_indices = indices

        args = "uint* indices, %(dtype)s* array, uint length" % \
                {"dtype" : cl.tools.dtype_to_ctype(self.dtype)}
        src = REMOVE_KNL.render()
        remove = get_elwise_kernel("remove", args, src)

        remove(sorted_indices, self.array, self.length)
        self.length -= len(indices)
        self._update_array_ref()

    def align(self, indices):
        self.set_data(cl.array.take(self.array, indices))

    def squeeze(self):
        self.set_data(self._data[:self.length])

    def copy_values(self, indices, dest):
        dest[:len(indices)] = self.array[indices]


class DeviceHelper(object):
    """Manages the arrays contained in a particle array on the device.

    Note that it converts the data to a suitable type depending on the value of
    get_config().use_double. Further, note that it assumes that the names of
    constants and properties do not clash.

    """

    def __init__(self, particle_array):
        self._particle_array = pa = particle_array
        self._queue = get_queue()
        self._ctx = get_context()
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

    def _check_property(self, prop):
        """Check if a property is present or not """
        if prop in self._props:
            return
        else:
            raise AttributeError, 'property %s not present'%(prop)

    def get_number_of_particles(self, real=False):
        if real:
            return self.num_real_particles
        else:
            if len(self._props) > 0:
                prop0 = self._data[self._props[0]]
                return len(prop0.array)
            else:
                return 0

    def align(self, indices):
        for prop in self._props:
            self._data[prop].align(indices)
            setattr(self, prop, self._data[prop].array)

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

    def get_device_array(self, prop):
        if prop in self._props:
            return self._data[prop]

    def max(self, arg):
        return float(cl.array.max(getattr(self, arg)).get())

    def update_min_max(self, props=None):
        """Update the min,max values of all properties """
        if props:
            for prop in props:
                array = self._data[prop]
                array.update_min_max()
        else:
            for prop in self._props:
                array = self._data[prop]
                array.update_min_max()

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

    def align_particles(self):
        tag_arr = getattr(self, 'tag')
        num_particles = self.get_number_of_particles()
        self.num_real_particles = cl.array.sum(tag_arr == 0)

        indices = cl.array.arange(self._queue, 0, num_particles, 1,
                dtype=np.uint32)

        radix_sort = cl.algorithm.RadixSort(
            self._ctx,
            "unsigned int* indices, unsigned int* tags",
            scan_kernel=GenericScanKernel, key_expr="tags[i]",
            sort_arg_names=["indices"]
        )

        (sorted_indices,), event = radix_sort(indices, tag_arr)
        self.align(sorted_indices)

    def remove_particles(self, indices):
        """ Remove particles whose indices are given in index_list.

        We repeatedly interchange the values of the last element and values from
        the index_list and reduce the size of the array by one. This is done for
        every property that is being maintained.

        Parameters
        ----------

        indices : array
            an array of indices, this array can be a list, numpy array
            or a LongArray.

        Notes
        -----

        Pseudo-code for the implementation::

            if index_list.length > number of particles
                raise ValueError

            sorted_indices <- index_list sorted in ascending order.

            for every every array in property_array
                array.remove(sorted_indices)

        """
        if len(indices) > self.get_number_of_particles():
            msg = 'Number of particles to be removed is greater than'
            msg += 'number of particles in array'
            raise ValueError, msg

        radix_sort = cl.algorithm.RadixSort(
            self._ctx,
            "unsigned int* indices",
            scan_kernel=GenericScanKernel, key_expr="indices[i]",
            sort_arg_names=["indices"]
            )

        (sorted_indices,), event = radix_sort(indices)

        for prop in self._props:
            self._data[prop].remove(sorted_indices, 1)
            setattr(self, prop, self._data[prop].array)

        if len(indices) > 0:
            self.align_particles()

    def remove_tagged_particles(self, tag):
        """ Remove particles that have the given tag.

        Parameters
        ----------

        tag : int
            the type of particles that need to be removed.

        """
        tag_array = getattr(self, 'tag')

        remove_places = tag_array == tag
        num_indices = int(cl.array.sum(remove_places).get())

        if num_indices == 0:
            return

        indices = cl.array.empty(self._queue, num_indices, np.uint32)
        head = cl.array.zeros(self._queue, 1, np.uint32)

        args = "uint* tag_array, uint tag, uint* indices, uint* head"
        src = REMOVE_INDICES_KNL.render()

        # find the indices of the particles to be removed.
        remove_indices = get_elwise_kernel("remove_indices", args, src)

        remove_indices(tag_array, tag, indices, head)

        # remove the particles.
        self.remove_particles(indices)

    def add_particles(self, **particle_props):
        """
        Add particles in particle_array to self.

        Parameters
        ----------

        particle_props : dict
            a dictionary containing cl arrays for various particle
            properties.

        Notes
        -----

         - all properties should have same length arrays.
         - all properties should already be present in this particles array.
           if new properties are seen, an exception will be raised.
           properties.

        """
        pa = self._particle_array

        if len(particle_props) == 0:
            return

        # check if the input properties are valid.
        for prop in particle_props:
            self._check_property(prop)

        num_extra_particles = len(list(particle_props.values())[0])
        old_num_particles = self.get_number_of_particles()
        new_num_particles = num_extra_particles + old_num_particles

        for prop in self._props:
            arr = self._data[prop]

            if prop in particle_props.keys():
                s_arr = particle_props[prop]
                arr.extend(s_arr)
            else:
                arr.resize(new_num_particles)
                # set the properties of the new particles to the default ones.
                arr.array[old_num_particles:] = pa.default_values[prop]

            setattr(self, prop, arr.array)

        if num_extra_particles > 0:
            # make sure particles are aligned properly.
            self.align_particles()
