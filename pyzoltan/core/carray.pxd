# This file (carray.pxd) has been generated automatically.
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

# ###########################################################################
# `IntArray` class.
# ###########################################################################
cdef class IntArray(BaseArray):
    """This class defines a managed array of ints. """
    cdef int *data
    cdef int *_old_data
    cdef public int minimum, maximum
    cdef IntArray _parent

    cdef _setup_npy_array(self)
    cdef void c_align_array(self, LongArray new_indices) nogil
    cdef void c_append(self, int value) nogil
    cdef void c_set_view(self, int *array, long length) nogil
    cdef int* get_data_ptr(self)

    cpdef int get(self, long idx)
    cpdef set(self, long idx, int value)
    cpdef append(self, int value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef set_view(self, IntArray, long start, long end)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, int value)

# ###########################################################################
# `UIntArray` class.
# ###########################################################################
cdef class UIntArray(BaseArray):
    """This class defines a managed array of unsigned ints. """
    cdef unsigned int *data
    cdef unsigned int *_old_data
    cdef public unsigned int minimum, maximum
    cdef UIntArray _parent

    cdef _setup_npy_array(self)
    cdef void c_align_array(self, LongArray new_indices) nogil
    cdef void c_append(self, unsigned int value) nogil
    cdef void c_set_view(self, unsigned int *array, long length) nogil
    cdef unsigned int* get_data_ptr(self)

    cpdef unsigned int get(self, long idx)
    cpdef set(self, long idx, unsigned int value)
    cpdef append(self, unsigned int value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef set_view(self, UIntArray, long start, long end)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, unsigned int value)

# ###########################################################################
# `LongArray` class.
# ###########################################################################
cdef class LongArray(BaseArray):
    """This class defines a managed array of longs. """
    cdef long *data
    cdef long *_old_data
    cdef public long minimum, maximum
    cdef LongArray _parent

    cdef _setup_npy_array(self)
    cdef void c_align_array(self, LongArray new_indices) nogil
    cdef void c_append(self, long value) nogil
    cdef void c_set_view(self, long *array, long length) nogil
    cdef long* get_data_ptr(self)

    cpdef long get(self, long idx)
    cpdef set(self, long idx, long value)
    cpdef append(self, long value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef set_view(self, LongArray, long start, long end)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, long value)

# ###########################################################################
# `FloatArray` class.
# ###########################################################################
cdef class FloatArray(BaseArray):
    """This class defines a managed array of floats. """
    cdef float *data
    cdef float *_old_data
    cdef public float minimum, maximum
    cdef FloatArray _parent

    cdef _setup_npy_array(self)
    cdef void c_align_array(self, LongArray new_indices) nogil
    cdef void c_append(self, float value) nogil
    cdef void c_set_view(self, float *array, long length) nogil
    cdef float* get_data_ptr(self)

    cpdef float get(self, long idx)
    cpdef set(self, long idx, float value)
    cpdef append(self, float value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef set_view(self, FloatArray, long start, long end)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, float value)

# ###########################################################################
# `DoubleArray` class.
# ###########################################################################
cdef class DoubleArray(BaseArray):
    """This class defines a managed array of doubles. """
    cdef double *data
    cdef double *_old_data
    cdef public double minimum, maximum
    cdef DoubleArray _parent

    cdef _setup_npy_array(self)
    cdef void c_align_array(self, LongArray new_indices) nogil
    cdef void c_append(self, double value) nogil
    cdef void c_set_view(self, double *array, long length) nogil
    cdef double* get_data_ptr(self)

    cpdef double get(self, long idx)
    cpdef set(self, long idx, double value)
    cpdef append(self, double value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef set_view(self, DoubleArray, long start, long end)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, double value)

