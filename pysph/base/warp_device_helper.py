from __future__ import print_function

import numpy as np
import warp as wp

try:
    from compyle.config import get_config
except ImportError:  # pragma: no cover - used only for isolated helper probes.
    get_config = None


_NUMPY_TO_WARP = {
    np.dtype(np.float64): wp.float64,
    np.dtype(np.float32): wp.float32,
    np.dtype(np.int64): wp.int64,
    np.dtype(np.int32): wp.int32,
    np.dtype(np.uint32): wp.uint32,
}


@wp.kernel
def _gather_f64(src: wp.array(dtype=wp.float64), indices: wp.array(dtype=wp.int32),
                stride: int, dst: wp.array(dtype=wp.float64)):
    i = wp.tid()
    p = i // stride
    c = i - p * stride
    dst[i] = src[indices[p] * stride + c]


@wp.kernel
def _gather_f32(src: wp.array(dtype=wp.float32), indices: wp.array(dtype=wp.int32),
                stride: int, dst: wp.array(dtype=wp.float32)):
    i = wp.tid()
    p = i // stride
    c = i - p * stride
    dst[i] = src[indices[p] * stride + c]


@wp.kernel
def _gather_i64(src: wp.array(dtype=wp.int64), indices: wp.array(dtype=wp.int32),
                stride: int, dst: wp.array(dtype=wp.int64)):
    i = wp.tid()
    p = i // stride
    c = i - p * stride
    dst[i] = src[indices[p] * stride + c]


@wp.kernel
def _gather_i32(src: wp.array(dtype=wp.int32), indices: wp.array(dtype=wp.int32),
                stride: int, dst: wp.array(dtype=wp.int32)):
    i = wp.tid()
    p = i // stride
    c = i - p * stride
    dst[i] = src[indices[p] * stride + c]


@wp.kernel
def _gather_u32(src: wp.array(dtype=wp.uint32), indices: wp.array(dtype=wp.int32),
                stride: int, dst: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    p = i // stride
    c = i - p * stride
    dst[i] = src[indices[p] * stride + c]


_GATHER_KERNELS = {
    np.dtype(np.float64): _gather_f64,
    np.dtype(np.float32): _gather_f32,
    np.dtype(np.int64): _gather_i64,
    np.dtype(np.int32): _gather_i32,
    np.dtype(np.uint32): _gather_u32,
}


_DTYPE_TO_CTYPE = {
    np.dtype(np.float64): 'double',
    np.dtype(np.float32): 'float',
    np.dtype(np.int64): 'long',
    np.dtype(np.int32): 'int',
    np.dtype(np.uint32): 'unsigned int',
}


def _warp_dtype(dtype):
    dtype = np.dtype(dtype)
    try:
        return _NUMPY_TO_WARP[dtype]
    except KeyError:
        raise TypeError('Unsupported Warp array dtype: %s' % dtype)


def _as_numpy_array(value, dtype=None):
    if isinstance(value, WarpArray):
        value = value.get()
    elif hasattr(value, 'get'):
        value = value.get()
    elif hasattr(value, 'get_npy_array'):
        value = value.get_npy_array()
    return np.asarray(value, dtype=dtype)


def _as_indices(indices):
    return _as_numpy_array(indices, dtype=np.int64).ravel()


def _dtype_to_ctype(dtype):
    dtype = np.dtype(dtype)
    try:
        return _DTYPE_TO_CTYPE[dtype]
    except KeyError:
        raise TypeError('Unsupported ParticleArray dtype: %s' % dtype)


class WarpArray(object):
    """Small compatibility wrapper around a one-dimensional Warp array."""

    def __init__(self, data=None, dtype=None, n=None, device=None):
        self.device = device
        if data is None:
            dtype = np.dtype(np.float64 if dtype is None else dtype)
            self._array = wp.zeros(n or 0, dtype=_warp_dtype(dtype),
                                   device=device)
            self.dtype = dtype
        else:
            arr = np.asarray(data, dtype=dtype)
            self.dtype = arr.dtype
            self._array = wp.from_numpy(arr, dtype=_warp_dtype(self.dtype),
                                        device=device)

    @property
    def data(self):
        return self._array

    @property
    def dev(self):
        return self._array

    def __len__(self):
        return self._array.shape[0]

    def __getitem__(self, index):
        return self.get()[index]

    def __setitem__(self, index, value):
        arr = self.get()
        arr[index] = value
        self.set(arr)

    def get(self):
        return self._array.numpy()

    def set(self, data):
        arr = np.asarray(data, dtype=self.dtype)
        self._array = wp.from_numpy(arr, dtype=_warp_dtype(self.dtype),
                                    device=self.device)

    def fill(self, value):
        self.set(np.full(len(self), value, dtype=self.dtype))

    def extend(self, data):
        data = _as_numpy_array(data, dtype=self.dtype)
        self.set(np.concatenate([self.get(), data]))

    def resize(self, size):
        old = self.get()
        new = np.zeros(size, dtype=self.dtype)
        new[:min(size, old.size)] = old[:min(size, old.size)]
        self.set(new)

    def copy(self):
        return WarpArray(self.get().copy(), device=self.device)

    def aligned(self, indices, stride=1):
        indices = np.asarray(indices, dtype=np.int32)
        if len(indices) == 0:
            return WarpArray(np.array([], dtype=self.dtype),
                             device=self.device)
        dst = wp.empty(len(indices) * stride, dtype=_warp_dtype(self.dtype),
                       device=self.device)
        wp_indices = wp.from_numpy(indices, dtype=wp.int32,
                                   device=self.device)
        kernel = _GATHER_KERNELS[np.dtype(self.dtype)]
        wp.launch(kernel, dim=dst.shape[0],
                  inputs=[self._array, wp_indices, stride, dst],
                  device=self.device)
        wp.synchronize_device(self.device)
        return WarpArray(dst.numpy(), dtype=self.dtype, device=self.device)


class WarpDeviceHelper(object):
    """Manage ParticleArray properties/constants with NVIDIA Warp arrays."""

    def __init__(self, particle_array, backend='warp', device=None):
        self.backend = backend
        self.device = wp.get_device(device)
        self._particle_array = pa = particle_array
        self.use_double = True if get_config is None else get_config().use_double
        self._dtype = np.float64 if self.use_double else np.float32
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
        arr = WarpArray(self._get_array(carray), device=self.device)
        self._data[name] = arr
        setattr(self, name, arr)

    def get_number_of_particles(self, real=False):
        if real:
            return self.num_real_particles
        elif len(self.properties) > 0:
            pname = self.properties[0]
            stride = self._particle_array.stride.get(pname, 1)
            return len(self._data[pname]) // stride
        else:
            return 0

    def get_device_array(self, name):
        return self._data[name]

    def add_prop(self, prop, carray):
        if prop not in self.properties:
            self.properties.append(prop)
        self._add_prop_or_const(prop, carray)

    def add_const(self, prop, carray):
        if prop not in self.constants:
            self.constants.append(prop)
        self._add_prop_or_const(prop, carray)

    def update_prop(self, prop, array):
        if not isinstance(array, WarpArray):
            array = WarpArray(array, device=self.device)
        if prop not in self.properties:
            self.properties.append(prop)
        self._data[prop] = array
        setattr(self, prop, array)

    def update_const(self, prop, array):
        if not isinstance(array, WarpArray):
            array = WarpArray(array, device=self.device)
        if prop not in self.constants:
            self.constants.append(prop)
        self._data[prop] = array
        setattr(self, prop, array)

    def remove_prop(self, prop):
        if prop in self.properties:
            self.properties.remove(prop)
        self._data.pop(prop, None)
        if hasattr(self, prop):
            delattr(self, prop)

    def push(self, *props):
        if len(props) == 0:
            props = list(self.properties) + list(self.constants)
        for prop in props:
            ary = self._get_prop_or_const(prop)
            self._data[prop].set(self._get_array(ary))

    def pull(self, *props):
        pa = self._particle_array
        if len(props) == 0:
            props = list(self.properties) + list(self.constants)
        for prop in props:
            data = self._data[prop].get()
            if prop in pa.properties:
                ary = pa.properties[prop]
                if ary.length != data.size:
                    ary.resize(data.size)
                ary.set_data(data)
            elif prop in pa.constants:
                ary = pa.constants[prop]
                if ary.length != data.size:
                    ary.resize(data.size)
                ary.set_data(data)
        pa.set_num_real_particles(self.num_real_particles)

    def max(self, prop):
        return self._data[prop].get().max()

    def resize(self, size):
        for prop in self.properties:
            stride = self._particle_array.stride.get(prop, 1)
            self._data[prop].resize(size * stride)

    def extend(self, num_particles):
        if num_particles <= 0:
            return

        old_size = self.get_number_of_particles()
        new_size = old_size + num_particles

        for prop in self.properties:
            arr = self._data[prop]
            stride = self._particle_array.stride.get(prop, 1)
            data = arr.get()
            new_data = np.empty(new_size * stride, dtype=arr.dtype)
            new_data[:old_size * stride] = data
            new_data[old_size * stride:] = \
                self._particle_array.default_values[prop]
            arr.set(new_data)
            self.update_prop(prop, arr)

    def align(self, indices):
        for prop in self.properties:
            stride = self._particle_array.stride.get(prop, 1)
            self._data[prop] = self._data[prop].aligned(indices, stride)
            setattr(self, prop, self._data[prop])

    def align_particles(self):
        tags = self._data['tag'].get()
        local = np.flatnonzero(tags == 0).astype(np.int32)
        other = np.flatnonzero(tags != 0).astype(np.int32)
        indices = np.concatenate([local, other])
        self.num_real_particles = local.size
        if indices.size > 0:
            self.align(indices)

    def remove_particles(self, indices, align=True):
        indices = _as_indices(indices)
        num_particles = self.get_number_of_particles()
        if len(indices) > num_particles:
            msg = 'Number of particles to be removed is greater than'
            msg += 'number of particles in array'
            raise ValueError(msg)

        indices = np.unique(indices[(indices >= 0) & (indices < num_particles)])
        if indices.size == 0:
            return

        keep = np.ones(num_particles, dtype=bool)
        keep[indices] = False
        self.align(np.flatnonzero(keep).astype(np.int32))

        if align:
            self.align_particles()

    def remove_tagged_particles(self, tag, align=True):
        indices = np.flatnonzero(self._data['tag'].get() == tag)
        self.remove_particles(indices, align=align)

    def add_particles(self, align=True, **particle_props):
        if len(particle_props) == 0:
            return 0

        for prop in particle_props:
            if prop not in self._particle_array.properties and \
                    prop not in self._particle_array.constants:
                raise AttributeError('property %s not present' % prop)

        first_prop = next(iter(particle_props))
        stride = self._particle_array.stride.get(first_prop, 1)
        num_extra_particles = len(_as_numpy_array(particle_props[first_prop])) // stride
        old_num_particles = self.get_number_of_particles()
        new_num_particles = old_num_particles + num_extra_particles

        for prop in self.properties:
            arr = self._data[prop]
            stride = self._particle_array.stride.get(prop, 1)
            if prop in particle_props:
                extra = _as_numpy_array(particle_props[prop], dtype=arr.dtype)
                arr.set(np.concatenate([arr.get(), extra]))
            else:
                data = np.empty(new_num_particles * stride, dtype=arr.dtype)
                data[:old_num_particles * stride] = arr.get()
                data[old_num_particles * stride:] = \
                    self._particle_array.default_values[prop]
                arr.set(data)
            self.update_prop(prop, arr)

        if num_extra_particles > 0 and align:
            self.align_particles()

        return 0

    def empty_clone(self, props=None):
        import pysph.base.particle_array

        prop_names = self.properties if props is None else props
        result_array = pysph.base.particle_array.ParticleArray(
            backend=self._particle_array.backend
        )
        result_array.set_name(self._particle_array.name)

        for prop_name in prop_names:
            src_arr = self._data[prop_name]
            stride = self._particle_array.stride.get(prop_name, 1)
            prop_type = _dtype_to_ctype(src_arr.dtype)
            prop_default = self._particle_array.default_values[prop_name]
            result_array.add_property(
                name=prop_name, type=prop_type,
                default=prop_default, stride=stride
            )

        for const in self.constants:
            result_array.gpu.update_const(const, self._data[const].copy())

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

    def append_parray(self, parray, align=True, update_constants=False):
        if parray.get_number_of_particles() == 0:
            return

        if parray.gpu is not None and parray.backend == 'warp':
            source_props = {
                prop: parray.gpu.get_device_array(prop).get()
                for prop in parray.gpu.properties
            }
        else:
            source_props = parray.get_property_arrays(all=True,
                                                      only_real=False)

        old_num_particles = self.get_number_of_particles()
        num_extra_particles = parray.get_number_of_particles()
        new_num_particles = old_num_particles + num_extra_particles
        pa = self._particle_array

        for prop_name in parray.properties:
            stride = parray.stride.get(prop_name, 1)
            if prop_name not in pa.properties:
                pa.add_property(
                    name=prop_name,
                    type=parray.properties[prop_name].get_c_type(),
                    default=parray.default_values[prop_name],
                    stride=stride
                )

            arr = self._data[prop_name]
            current = arr.get()
            data = np.empty(new_num_particles * stride, dtype=arr.dtype)
            data[:old_num_particles * stride] = \
                current[:old_num_particles * stride]
            data[old_num_particles * stride:] = \
                _as_numpy_array(source_props[prop_name], dtype=arr.dtype)
            arr.set(data)
            self.update_prop(prop_name, arr)

        if update_constants:
            for const in parray.constants:
                if const not in pa.constants:
                    pa.add_constant(
                        const, parray.constants[const].get_npy_array()
                    )

        if num_extra_particles > 0 and align:
            self.align_particles()

    def extract_particles(self, indices, dest_array=None, align=True,
                          props=None):
        if not dest_array:
            dest_array = self.empty_clone(props=props)

        indices = _as_indices(indices).astype(np.int32)
        if props is None:
            prop_names = list(self.properties)
        else:
            prop_names = props

        if len(indices) == 0:
            return dest_array

        start_idx = dest_array.gpu.get_number_of_particles()
        dest_array.gpu.extend(len(indices))

        for prop in prop_names:
            stride = self._particle_array.stride.get(prop, 1)
            extracted = self._data[prop].aligned(indices, stride).get()
            dest = dest_array.gpu.get_device_array(prop)
            data = dest.get()
            data[start_idx * stride:(start_idx + len(indices)) * stride] = extracted
            dest.set(data)
            dest_array.gpu.update_prop(prop, dest)

        if align:
            dest_array.gpu.align_particles()

        return dest_array
