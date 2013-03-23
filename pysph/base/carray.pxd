# This file (carray.pxd) has been generated automatically on
# Sat Mar 23 20:03:12 2013
# DO NOT modify this file
# To make changes modify the source templates (carray_pxd.src) and regenerate
"""
Implementation of arrays of different types in Cython.

Declaration File.

"""

# numpy import
cimport numpy as np

# forward declaration
cdef class BaseArray
cdef class LongArray(BaseArray)

cdef class BaseArrayIter:
    cdef BaseArray arr
    cdef long i

cdef class BaseArray:
    """Base class for managed C-arrays."""
    cdef public long length, alloc, _length
    cdef np.ndarray _npy_array

    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)

    cpdef align_array(self, LongArray new_indices)
    cdef void _align_array(self, LongArray new_indices)
    cpdef str get_c_type(self)
    cpdef copy_values(self, LongArray indices, BaseArray dest)
    cpdef copy_subset(self, BaseArray source, long start_index=*, long end_index=*)
    cpdef update_min_max(self)


################################################################################
# `IntArray` class.
################################################################################
cdef class IntArray(BaseArray):
    """This class defines a managed array of ints. """
    cdef int *data
    cdef public int minimum, maximum

    cdef _setup_npy_array(self)
    cdef int* get_data_ptr(self)

    cpdef int get(self, long idx)
    cpdef set(self, long idx, int value)
    cpdef append(self, int value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, int value)

    cdef void _align_array(self, LongArray new_indices)


################################################################################
# `DoubleArray` class.
################################################################################
cdef class DoubleArray(BaseArray):
    """This class defines a managed array of doubles. """
    cdef double *data
    cdef public double minimum, maximum

    cdef _setup_npy_array(self)
    cdef double* get_data_ptr(self)

    cpdef double get(self, long idx)
    cpdef set(self, long idx, double value)
    cpdef append(self, double value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, double value)

    cdef void _align_array(self, LongArray new_indices)


################################################################################
# `FloatArray` class.
################################################################################
cdef class FloatArray(BaseArray):
    """This class defines a managed array of floats. """
    cdef float *data
    cdef public float minimum, maximum

    cdef _setup_npy_array(self)
    cdef float* get_data_ptr(self)

    cpdef float get(self, long idx)
    cpdef set(self, long idx, float value)
    cpdef append(self, float value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, float value)

    cdef void _align_array(self, LongArray new_indices)


################################################################################
# `LongArray` class.
################################################################################
cdef class LongArray(BaseArray):
    """This class defines a managed array of longs. """
    cdef long *data
    cdef public long minimum, maximum

    cdef _setup_npy_array(self)
    cdef long* get_data_ptr(self)

    cpdef long get(self, long idx)
    cpdef set(self, long idx, long value)
    cpdef append(self, long value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, long value)

    cdef void _align_array(self, LongArray new_indices)


################################################################################
# `UIntArray` class.
################################################################################
cdef class UIntArray(BaseArray):
    """This class defines a managed array of unsigned ints. """
    cdef unsigned int *data
    cdef public unsigned int minimum, maximum

    cdef _setup_npy_array(self)
    cdef unsigned int* get_data_ptr(self)

    cpdef unsigned int get(self, long idx)
    cpdef set(self, long idx, unsigned int value)
    cpdef append(self, unsigned int value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef long index(self, unsigned int value)

    cdef void _align_array(self, LongArray new_indices)

