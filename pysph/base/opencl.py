"""Common OpenCL related functionality.
"""

import logging
import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: 401
import pyopencl.algorithm
import pyopencl.tools  # noqa: 401
from pyopencl.scan import GenericScanKernel
from pyopencl.reduction import ReductionKernel
from pyopencl.elementwise import ElementwiseKernel
from mako.template import Template

from pysph.cpy.config import get_config
from pysph.cpy.opencl import (  # noqa: 401
    get_context, get_queue, profile, profile_kernel,
    print_profile, set_context, set_queue
)
import pysph.base.particle_array


logger = logging.getLogger()


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
        self.dtype = dtype
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
        self.minimum = float(cl.array.min(self.array).get())
        self.maximum = float(cl.array.max(self.array).get())

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
            msg = 'Number of indices to be removed is greater than'
            msg += 'number of indices in array'
            raise ValueError(msg)

        if_remove = DeviceArray(np.int32, n=self.length)
        if_remove.fill(0)
        new_array = self.copy()

        fill_if_remove_knl = get_elwise_kernel(
            "fill_if_remove_knl",
            "int* indices, int* if_remove",
            "if_remove[indices[i]] = 1;"
        )

        fill_if_remove_knl(indices, if_remove.array)

        remove_knl = GenericScanKernel(
            self.ctx, np.int32,
            arguments="__global int *if_remove,\
            __global %(dtype)s *array,\
            __global %(dtype)s *new_array" %
            {"dtype": cl.tools.dtype_to_ctype(self.dtype)},
            input_expr="if_remove[i]",
            scan_expr="a+b", neutral="0",
            output_statement="""
            if(!if_remove[i]) new_array[i - item] = array[i];
            """)

        remove_knl(if_remove.array, self.array, new_array.array)

        self.set_data(new_array.array[:-len(indices)])

    def align(self, indices):
        self.set_data(cl.array.take(self.array, indices))

    def squeeze(self):
        self.set_data(self._data[:self.length])

    def copy_values(self, indices, dest):
        dest[:len(indices)] = cl.array.take(self.array, indices)


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
        self.properties = []
        self.constants = []

        for prop, ary in pa.properties.items():
            self.add_prop(prop, ary)
        for prop, ary in pa.constants.items():
            self.add_const(prop, ary)

        self.align_particles()

    def _get_array(self, ary):
        ctype = ary.get_c_type()
        if ctype in ['float', 'double']:
            return ary.get_npy_array().astype(self._dtype)
        else:
            return ary.get_npy_array()

    def _get_prop_or_const(self, prop):
        pa = self._particle_array
        return pa.properties.get(prop, pa.constants.get(prop))

    def _add_prop_or_const(self, name, carray):
        """Add a new property or constant given the name and carray, note
        that this assumes that this property is already added to the
        particle array.
        """
        np_array = self._get_array(carray)
        g_ary = DeviceArray(np_array.dtype, n=carray.length)
        g_ary.array.set(np_array)
        self._data[name] = g_ary
        setattr(self, name, g_ary.array)

    def get_number_of_particles(self, real=False):
        if real:
            return self.num_real_particles
        else:
            if len(self.properties) > 0:
                prop0 = self._data[self.properties[0]]
                return len(prop0.array)
            else:
                return 0

    def align(self, indices):
        for prop in self.properties:
            self._data[prop].align(indices)
            setattr(self, prop, self._data[prop].array)

    def add_prop(self, name, carray):
        """Add a new property given the name and carray, note
        that this assumes that this property is already added to the
        particle array.
        """
        if name in self._particle_array.properties and \
                name not in self.properties:
            self._add_prop_or_const(name, carray)
            self.properties.append(name)

    def add_const(self, name, carray):
        """Add a new constant given the name and carray, note
        that this assumes that this property is already added to the
        particle array.
        """
        if name in self._particle_array.constants and \
                name not in self.constants:
            self._add_prop_or_const(name, carray)
            self.constants.append(name)

    def update_prop(self, name, dev_array):
        """Add a new property to DeviceHelper. Note that this property
        is not added to the particle array itself"""
        self._data[name] = dev_array
        setattr(self, name, dev_array.array)
        if name not in self.properties:
            self.properties.append(name)

    def update_const(self, name, dev_array):
        """Add a new constant to DeviceHelper. Note that this property
        is not added to the particle array itself"""
        self._data[name] = dev_array
        setattr(self, name, dev_array.array)
        if name not in self.constants:
            self.constants.append(name)

    def get_device_array(self, prop):
        if prop in self.properties or prop in self.constants:
            return self._data[prop]

    def max(self, arg):
        return float(cl.array.max(getattr(self, arg)).get())

    def update_min_max(self, props=None):
        """Update the min,max values of all properties """
        props = props if props else self.properties
        for prop in props:
            array = self._data[prop]
            array.update_min_max()

    def pull(self, *args):
        p = self._particle_array
        if len(args) == 0:
            args = self._data.keys()
        for arg in args:
            arg_cpu = getattr(self, arg).get()
            if arg in p.properties or arg in p.constants:
                pa_arr = self._get_prop_or_const(arg)
            else:
                if arg in self.properties:
                    p.add_property(arg)
                if arg in self.constants:
                    p.add_constant(arg)
                pa_arr = self._get_prop_or_const(arg)
            if arg_cpu.size != pa_arr.length:
                pa_arr.resize(arg_cpu.size)
            pa_arr.set_data(arg_cpu)
        p.set_num_real_particles(self.num_real_particles)

    def push(self, *args):
        if len(args) == 0:
            args = self._data.keys()
        for arg in args:
            cl_arr = cl.array.to_device(
                get_queue(), self._get_array(
                    self._get_prop_or_const(arg)))
            self._data[arg].set_data(cl_arr)
            setattr(self, arg, self._data[arg].array)

    def _check_property(self, prop):
        """Check if a property is present or not """
        if prop in self.properties or prop in self.constants:
            return
        else:
            raise AttributeError('property %s not present' % (prop))

    def remove_prop(self, name):
        if name in self.properties:
            self.properties.remove(name)
        if name in self._data:
            del self._data[name]
            delattr(self, name)

    def resize(self, new_size):
        for prop in self.properties:
            self._data[prop].resize(new_size)
            setattr(self, prop, self._data[prop].array)

    def align_particles(self):
        tag_arr = self._data['tag'].array

        if len(tag_arr) == 0:
            self.num_real_particles = 0
            return

        num_particles = len(tag_arr)

        new_indices = cl.array.empty(self._queue, num_particles,
                                     dtype=np.int32)
        num_real_particles = cl.array.empty(self._queue, 1,
                                            dtype=np.int32)

        align_particles_knl = GenericScanKernel(
            self._ctx, np.int32,
            arguments="__global int *tag_arr, __global int *new_indices, \
                    __global int *num_real_particles, int num_particles",
            input_expr="tag_arr[i] == 0",
            scan_expr="a+b", neutral="0",
            output_statement="""
            int t = last_item + i - prev_item;
            int idx = tag_arr[i] ? t : prev_item;
            new_indices[idx] = i;
            if(i == num_particles - 1)
                num_real_particles[0] = last_item;
            """
        )

        align_particles_knl(tag_arr, new_indices, num_real_particles,
                            num_particles)

        self.align(new_indices)

        self.num_real_particles = int(num_real_particles.get())

    def remove_particles_bool(self, if_remove):
        """ Remove particle i if if_remove[i] is True
        """
        num_indices = int(cl.array.sum(if_remove).get())

        if num_indices == 0:
            return

        num_particles = self.get_number_of_particles()
        new_indices = DeviceArray(np.uint32, n=num_particles)
        num_removed_particles = cl.array.empty(self._queue, 1, dtype=np.int32)

        remove_knl = GenericScanKernel(
            self._ctx, np.int32,
            arguments="__global int *if_remove, __global uint *new_indices, \
                    __global int *num_removed_particles, int num_particles",
            input_expr="if_remove[i]",
            scan_expr="a+b", neutral="0",
            output_statement="""
            if(!if_remove[i]) new_indices[i - item] = i;
            if(i == num_particles - 1) num_removed_particles[0] = last_item;
            """
        )

        remove_knl(if_remove, new_indices.array, num_removed_particles,
                   num_particles)

        new_num_particles = num_particles - int(num_removed_particles.get())

        for prop in self.properties:
            self._data[prop].align(new_indices.array[:new_num_particles])
            setattr(self, prop, self._data[prop].array)

    def remove_particles(self, indices):
        """ Remove particles whose indices are given in index_list.

        We repeatedly interchange the values of the last element and
        values from the index_list and reduce the size of the array
        by one. This is done for every property that is being maintained.

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
            raise ValueError(msg)

        num_particles = self.get_number_of_particles()
        if_remove = DeviceArray(np.int32, n=num_particles)
        if_remove.fill(0)

        fill_if_remove_knl = get_elwise_kernel(
            "fill_if_remove_knl",
            "uint* indices, int* if_remove, int size",
            "if(indices[i] < size) if_remove[indices[i]] = 1;"
        )

        fill_if_remove_knl(indices, if_remove.array, num_particles)

        self.remove_particles_bool(if_remove.array)

        if len(indices) > 0:
            self.align_particles()

    def remove_tagged_particles(self, tag):
        """ Remove particles that have the given tag.

        Parameters
        ----------

        tag : int
            the type of particles that need to be removed.

        """
        tag_array = self.tag
        if_remove = (tag_array == tag).astype(np.int32)
        self.remove_particles_bool(if_remove)

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

        for prop in self.properties:
            arr = self._data[prop]

            if prop in particle_props.keys():
                s_arr = particle_props[prop]
                arr.extend(s_arr)
            else:
                arr.resize(new_num_particles)
                # set the properties of the new particles to the default ones.
                arr.array[old_num_particles:] = pa.default_values[prop]

            self.update_prop(prop, arr)

        if num_extra_particles > 0:
            # make sure particles are aligned properly.
            self.align_particles()

        return 0

    def extend(self, num_particles):
        """ Increase the total number of particles by the requested amount

        New particles are added at the end of the list, you may
        have to manually call align_particles later.
        """
        if num_particles <= 0:
            return

        old_size = self.get_number_of_particles()
        new_size = old_size + num_particles

        for prop in self.properties:
            arr = self._data[prop]
            arr.resize(new_size)
            arr.array[old_size:] = self._particle_array.default_values[prop]
            self.update_prop(prop, arr)

    def append_parray(self, parray):
        """ Add particles from a particle array

        properties that are not there in self will be added
        """
        if parray.gpu is None:
            parray.set_device_helper(DeviceHelper(parray))

        if parray.gpu.get_number_of_particles() == 0:
            return

        num_extra_particles = parray.gpu.get_number_of_particles()
        old_num_particles = self.get_number_of_particles()
        new_num_particles = num_extra_particles + old_num_particles

        # extend current arrays by the required number of particles
        self.extend(num_extra_particles)

        for prop_name in parray.gpu.properties:
            if prop_name in self.properties:
                arr = self._data[prop_name]
                source = parray.gpu.get_device_array(prop_name)
                arr.array[old_num_particles:] = source.array
            else:
                # meaning this property is not there in self.
                dtype = parray.gpu.get_device_array(prop_name).dtype
                arr = DeviceArray(dtype, n=new_num_particles)
                arr.fill(parray.default_values[prop_name])
                self.update_prop(prop_name, arr)

                # now add the values to the end of the created array
                dest = self._data[prop_name]
                source = parray.gpu.get_device_array(prop_name)
                dest.array[old_num_particles:] = source.array

        for const in parray.gpu.constants:
            if const not in self.constants:
                arr = parray.gpu.get_device_array(const)
                self.update_const(const, arr.copy())

        if num_extra_particles > 0:
            self.align_particles()

    def extract_particles(self, indices, props=None):
        """Create new particle array for particles with given indices

        Parameters
        ----------

        indices : cl.array.Array
            indices of particles to be extracted.

        props : list
            the list of properties to extract, if None all properties
            are extracted.

        """
        result_array = pysph.base.particle_array.ParticleArray()
        result_array.set_device_helper(DeviceHelper(result_array))

        if props is None:
            prop_names = self.properties
        else:
            prop_names = props

        if len(indices) == 0:
            return result_array

        for prop_name in prop_names:
            src_arr = self._data[prop_name]
            dst_arr = DeviceArray(src_arr.dtype, n=len(indices))
            src_arr.copy_values(indices, dst_arr.array)

            prop_type = cl.tools.dtype_to_ctype(src_arr.dtype)
            prop_default = self._particle_array.default_values[prop_name]
            result_array.add_property(name=prop_name,
                                      type=prop_type,
                                      default=prop_default)

        result_array.gpu.extend(len(indices))

        for prop in prop_names:
            src_arr = self._data[prop].array
            dst_arr = result_array.gpu.get_device_array(prop)
            dst_arr.set_data(cl.array.take(src_arr, indices))

        for const in self.constants:
            result_array.gpu.update_const(const, self._data[const].copy())

        result_array.gpu.align_particles()
        result_array.set_name(self._particle_array.name)

        if props is None:
            output_arrays = list(self._particle_array.output_property_arrays)
        else:
            output_arrays = list(
                set(props).intersection(
                    self._particle_array.output_property_arrays
                )
            )

        result_array.set_output_arrays(output_arrays)
        return result_array
