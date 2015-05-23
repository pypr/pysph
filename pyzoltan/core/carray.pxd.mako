<%
type_info = [
    ('int', 'IntArray', 'NPY_INT'),
    ('unsigned int', 'UIntArray', 'NPY_UINT'),
    ('long', 'LongArray', 'NPY_LONG'),
    ('float', 'FloatArray', 'NPY_FLOAT'),
    ('double', 'DoubleArray', 'NPY_DOUBLE'),
]
%># This file (carray.pxd) has been generated automatically.
# DO NOT modify this file
# To make changes modify the source templates (carray.pxd.mako) and regenerate
"""
Implementation of resizeable arrays of different types in Cython.

Declaration File.

"""

# numpy import
cimport numpy as np

cdef long aligned(long n, int item_size) nogil
cdef void* aligned_malloc(size_t bytes) nogil
cdef void* aligned_realloc(void* existing, size_t bytes, size_t old_size) nogil
cdef void aligned_free(void* p) nogil

# forward declaration
cdef class BaseArray
cdef class LongArray(BaseArray)

cdef class BaseArrayIter:
    cdef BaseArray arr
    cdef long i

cdef class BaseArray:
    """Base class for managed C-arrays."""
    cdef public long length, alloc
    cdef np.ndarray _npy_array

    cdef void c_align_array(self, LongArray new_indices) nogil
    cdef void c_reserve(self, long size) nogil
    cdef void c_reset(self) nogil
    cdef void c_resize(self, long size) nogil
    cdef void c_squeeze(self) nogil

    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)

    cpdef align_array(self, LongArray new_indices)
    cpdef str get_c_type(self)
    cpdef copy_values(self, LongArray indices, BaseArray dest)
    cpdef copy_subset(self, BaseArray source, long start_index=*, long end_index=*)
    cpdef update_min_max(self)

% for ARRAY_TYPE, CLASSNAME, NUMPY_TYPENAME in type_info:
# ###########################################################################
# `${CLASSNAME}` class.
# ###########################################################################
cdef class ${CLASSNAME}(BaseArray):
    """This class defines a managed array of ${ARRAY_TYPE}s. """
    cdef ${ARRAY_TYPE} *data
    cdef ${ARRAY_TYPE} *_old_data
    cdef public ${ARRAY_TYPE} minimum, maximum
    cdef ${CLASSNAME} _parent

    cdef _setup_npy_array(self)
    cdef void c_align_array(self, LongArray new_indices) nogil
    cdef void c_append(self, ${ARRAY_TYPE} value) nogil
    cdef void c_set_view(self, ${ARRAY_TYPE} *array, long length) nogil
    cdef ${ARRAY_TYPE}* get_data_ptr(self)

    cpdef ${ARRAY_TYPE} get(self, long idx)
    cpdef set(self, long idx, ${ARRAY_TYPE} value)
    cpdef append(self, ${ARRAY_TYPE} value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef set_view(self, ${CLASSNAME}, long start, long end)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, ${ARRAY_TYPE} value)

% endfor
