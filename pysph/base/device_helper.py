import logging
import numpy as np
from pytools import memoize_method

from pysph.cpy.config import get_config
from pysph.cpy.array import get_backend, wrap_array, Array
from pysph.cpy.parallel import Elementwise, Scan
from pysph.cpy.api import declare, annotate
from pysph.cpy.types import dtype_to_ctype
import pysph.cpy.array as array
import pysph.base.particle_array


logger = logging.getLogger()


class DeviceHelper(object):
    """Manages the arrays contained in a particle array on the device.

    Note that it converts the data to a suitable type depending on the value of
    get_config().use_double. Further, note that it assumes that the names of
    constants and properties do not clash.

    """

    def __init__(self, particle_array, backend=None):
        self.backend = get_backend(backend)
        self._particle_array = pa = particle_array
        use_double = get_config().use_double
        self._dtype = np.float64 if use_double else np.float32
        self.num_real_particles = pa.num_real_particles
        self._data = {}
        self.properties = []
        self.constants = []

        for prop, ary in pa.properties.items():
            self.add_prop(prop, ary)
        for prop, ary in pa.constants.items():
            self.add_const(prop, ary)

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
        g_ary = Array(np_array.dtype, n=carray.length,
                      backend=self.backend)
        g_ary.set(np_array)
        self._data[name] = g_ary
        setattr(self, name, g_ary)

    def get_number_of_particles(self, real=False):
        if real:
            return self.num_real_particles
        else:
            if len(self.properties) > 0:
                pname = self.properties[0]
                stride = self._particle_array.stride.get(pname, 1)
                prop0 = self._data[pname]
                return len(prop0.dev) // stride
            else:
                return 0

    def align(self, indices):
        '''Note that the indices passed here is a dictionary
        keyed on the stride.
        '''
        if not isinstance(indices, dict):
            indices = {1: indices}
        for prop in self.properties:
            stride = self._particle_array.stride.get(prop, 1)
            self._data[prop].align(indices.get(stride))
            setattr(self, prop, self._data[prop])

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
        setattr(self, name, dev_array)
        if name not in self.properties:
            self.properties.append(name)

    def update_const(self, name, dev_array):
        """Add a new constant to DeviceHelper. Note that this property
        is not added to the particle array itself"""
        self._data[name] = dev_array
        setattr(self, name, dev_array)
        if name not in self.constants:
            self.constants.append(name)

    def get_device_array(self, prop):
        if prop in self.properties or prop in self.constants:
            return self._data[prop]

    def max(self, arg):
        return float(array.maximum(getattr(self, arg),
                                   backend=self.backend))

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
            dev_arr = array.to_device(
                self._get_array(self._get_prop_or_const(arg)),
                backend=self.backend)
            self._data[arg].set_data(dev_arr)
            setattr(self, arg, self._data[arg])

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
            stride = self._particle_array.stride.get(prop, 1)
            self._data[prop].resize(new_size * stride)
            setattr(self, prop, self._data[prop])

    @memoize_method
    def _get_align_kernel_without_strides(self):
        @annotate(i='int', tag_arr='gintp', return_='int')
        def align_input_expr(i, tag_arr):
            return tag_arr[i] == 0

        @annotate(int='i, item, prev_item, last_item, num_particles',
                  gintp='tag_arr, new_indices, num_real_particles')
        def align_output_expr(i, item, prev_item, last_item, tag_arr,
                              new_indices, num_particles, num_real_particles):
            t, idx = declare('int', 2)
            t = last_item + i - prev_item
            idx = t if tag_arr[i] else prev_item
            new_indices[idx] = i
            if i == num_particles - 1:
                num_real_particles[0] = last_item

        align_particles_knl = Scan(align_input_expr, align_output_expr,
                                   'a+b', dtype=np.int32, backend=self.backend)

        return align_particles_knl

    @memoize_method
    def _get_align_kernel_with_strides(self):
        @annotate(i='int', tag_arr='gintp', return_='int')
        def align_input_expr(i, tag_arr):
            return tag_arr[i] == 0

        @annotate(int='i, item, prev_item, last_item, stride, num_particles',
                  gintp='tag_arr, new_indices',
                  return_='int')
        def align_output_expr(i, item, prev_item, last_item, tag_arr,
                              new_indices, num_particles, stride):
            t, idx, j_s = declare('int', 3)
            t = last_item + i - prev_item
            idx = t if tag_arr[i] else prev_item
            for j_s in range(stride):
                new_indices[stride * idx + j_s] = stride * i + j_s

        align_particles_knl = Scan(align_input_expr, align_output_expr,
                                   'a+b', dtype=np.int32, backend=self.backend)

        return align_particles_knl

    def align_particles(self):
        tag_arr = self._data['tag']

        if len(tag_arr) == 0:
            self.num_real_particles = 0
            return

        num_particles = len(tag_arr)

        new_indices = array.empty(num_particles, dtype=np.int32,
                                  backend=self.backend)
        num_real_particles = array.empty(1, dtype=np.int32,
                                         backend=self.backend)

        align_particles_knl = self._get_align_kernel_without_strides()

        align_particles_knl(tag_arr=tag_arr, new_indices=new_indices,
                            num_particles=num_particles,
                            num_real_particles=num_real_particles)

        indices = {1: new_indices}
        for stride in set(self._particle_array.stride.values()):
            if stride > 1:
                indices[stride] = self._build_indices_with_strides(
                    tag_arr, stride
                )

        self.align(indices)

        self.num_real_particles = int(num_real_particles.get())

    def _build_indices_with_strides(self, tag_arr, stride):
        num_particles = len(tag_arr.dev)

        new_indices = array.empty(num_particles * stride,
                                  dtype=np.int32,
                                  backend=self.backend)

        align_particles_knl = self._get_align_kernel_with_strides()

        align_particles_knl(tag_arr=tag_arr, new_indices=new_indices,
                            num_particles=num_particles,
                            stride=stride)

        return new_indices

    @memoize_method
    def _get_remove_particles_bool_kernels(self):
        @annotate(i='int', if_remove='gintp', return_='int')
        def remove_input_expr(i, if_remove):
            return if_remove[i]

        @annotate(int='i, item, last_item, num_particles',
                  gintp='if_remove, num_removed_particles',
                  new_indices='guintp')
        def remove_output_expr(i, item, last_item, if_remove, new_indices,
                               num_removed_particles, num_particles):
            if not if_remove[i]:
                new_indices[i - item] = i
            if i == num_particles - 1:
                num_removed_particles[0] = last_item

        remove_knl = Scan(remove_input_expr, remove_output_expr,
                          'a+b', dtype=np.int32, backend=self.backend)

        @annotate(int='i, size, stride', guintp='indices, new_indices')
        def stride_knl_elwise(i, indices, new_indices, size, stride):
            tmp_idx, j_s = declare('unsigned int', 2)
            for j_s in range(stride):
                tmp_idx = i * stride + j_s
                if tmp_idx < size:
                    new_indices[tmp_idx] = indices[i] * stride + j_s

        stride_knl = Elementwise(stride_knl_elwise, backend=self.backend)

        return remove_knl, stride_knl

    def _remove_particles_bool(self, if_remove):
        """ Remove particle i if if_remove[i] is True
        """
        num_indices = int(array.sum(if_remove, backend=self.backend))

        if num_indices == 0:
            return

        num_particles = self.get_number_of_particles()
        new_indices = Array(np.uint32, n=(num_particles - num_indices),
                            backend=self.backend)
        num_removed_particles = array.empty(1, dtype=np.int32,
                                            backend=self.backend)

        remove_knl, stride_knl = self._get_remove_particles_bool_kernels()

        remove_knl(if_remove=if_remove, new_indices=new_indices,
                   num_removed_particles=num_removed_particles,
                   num_particles=num_particles)

        new_num_particles = num_particles - int(num_removed_particles.get())

        strides = set(self._particle_array.stride.values())
        s_indices = {1: new_indices}
        for stride in strides:
            if stride == 1:
                continue
            size = new_num_particles * stride
            s_index = Array(np.uint32, n=size, backend=self.backend)
            stride_knl(new_indices, s_index, size, stride)
            s_indices[stride] = s_index

        for prop in self.properties:
            stride = self._particle_array.stride.get(prop, 1)
            s_index = s_indices[stride]
            self._data[prop].align(s_index)
            setattr(self, prop, self._data[prop])

        self.align_particles()

    @memoize_method
    def _get_remove_particles_kernel(self):
        @annotate(int='i, size', indices='guintp', if_remove='gintp')
        def fill_if_remove_elwise(i, indices, if_remove, size):
            if indices[i] < size:
                if_remove[indices[i]] = 1

        fill_if_remove_knl = Elementwise(fill_if_remove_elwise,
                                         backend=self.backend)

        return fill_if_remove_knl

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
        if_remove = Array(np.int32, n=num_particles, backend=self.backend)
        if_remove.fill(0)

        fill_if_remove_knl = self._get_remove_particles_kernel()
        fill_if_remove_knl(indices, if_remove, num_particles)

        self._remove_particles_bool(if_remove)

    def remove_tagged_particles(self, tag):
        """ Remove particles that have the given tag.

        Parameters
        ----------

        tag : int
            the type of particles that need to be removed.

        """
        tag_array = self.tag
        if_remove = wrap_array((tag_array.dev == tag).astype(np.int32),
                               self.backend)
        self._remove_particles_bool(if_remove)

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
            stride = self._particle_array.stride.get(prop, 1)
            if prop in particle_props.keys():
                s_arr = particle_props[prop]
                arr.extend(s_arr)
            else:
                arr.resize(new_num_particles * stride)
                # set the properties of the new particles to the default ones.
                arr[old_num_particles * stride:] = pa.default_values[prop]

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
            stride = self._particle_array.stride.get(prop, 1)
            arr.resize(new_size * stride)
            arr[old_size * stride:] = \
                self._particle_array.default_values[prop]
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

        my_stride = self._particle_array.stride
        for prop_name in parray.gpu.properties:
            stride = parray.stride.get(prop_name, 1)
            if stride > 1 and prop_name not in my_stride:
                my_stride[prop_name] = stride
            if prop_name in self.properties:
                arr = self._data[prop_name]
                source = parray.gpu.get_device_array(prop_name)
                arr.dev[old_num_particles * stride:] = source.dev
            else:
                # meaning this property is not there in self.
                dtype = parray.gpu.get_device_array(prop_name).dtype
                arr = Array(dtype, n=new_num_particles * stride,
                            backend=self.backend)
                arr.fill(parray.default_values[prop_name])
                self.update_prop(prop_name, arr)

                # now add the values to the end of the created array
                dest = self._data[prop_name]
                source = parray.gpu.get_device_array(prop_name)
                dest.dev[old_num_particles * stride:] = source.dev

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

        indices : Array
            indices of particles to be extracted.

        props : list
            the list of properties to extract, if None all properties
            are extracted.

        """
        result_array = pysph.base.particle_array.ParticleArray()
        result_array.set_name(self._particle_array.name)
        result_array.set_device_helper(DeviceHelper(result_array,
                                                    backend=self.backend))

        if props is None:
            prop_names = self.properties
        else:
            prop_names = props

        if len(indices) == 0:
            return result_array

        for prop_name in prop_names:
            src_arr = self._data[prop_name]
            stride = self._particle_array.stride.get(prop_name, 1)

            prop_type = dtype_to_ctype(src_arr.dtype)
            prop_default = self._particle_array.default_values[prop_name]
            result_array.add_property(name=prop_name,
                                      type=prop_type,
                                      default=prop_default, stride=stride)

        s_indices = self._build_copy_indices_with_strides(indices)

        result_array.gpu.extend(len(indices))

        for prop in prop_names:
            stride = self._particle_array.stride.get(prop, 1)
            s_index = s_indices[stride]
            src_arr = self._data[prop]
            dst_arr = result_array.gpu.get_device_array(prop)
            dst_arr.set_data(array.take(src_arr, s_index,
                                        backend=self.backend))

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

    @memoize_method
    def _get_copy_indices_kernel(self):
        @annotate(int='i, size, stride', guintp='indices, new_indices')
        def fill_indices_elwise(i, indices, new_indices, size, stride):
            tmp_idx, j_s = declare('unsigned int', 2)
            for j_s in range(stride):
                tmp_idx = i * stride + j_s
                if tmp_idx < size:
                    new_indices[tmp_idx] = indices[i] * stride + j_s

        fill_indices_knl = Elementwise(fill_indices_elwise,
                                       backend=self.backend)

        return fill_indices_knl

    def _build_copy_indices_with_strides(self, indices):
        result = {1: indices}
        n_indices = len(indices)
        strides = set(self._particle_array.stride.values())
        fill_indices_knl = self._get_copy_indices_kernel()

        for stride in strides:
            sz = n_indices * stride
            new_indices = array.empty(sz, dtype=np.uint32,
                                      backend=self.backend)
            fill_indices_knl(indices, new_indices, sz, stride)
            result[stride] = new_indices

        return result
