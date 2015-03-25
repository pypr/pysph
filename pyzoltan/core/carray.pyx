# This file (carray.pxd) has been generated automatically on
# Wed Mar 25 18:45:06 2015
# DO NOT modify this file
# To make changes modify the source templates (carray_pxd.src) and regenerate
"""
Implementation of resizeable arrays of different types in Cython.

All arrays provide for the following operations:

 - access by indexing.
 - access through get/set function.
 - appending values at the end of the array.
 - reserving space for future appends.
 - access to internal data through a numpy array.

** Numpy array access **
Each array also provides an interface to its data through a numpy array. This
is done through the get_npy_array function. The returned numpy array can be
used just like any other numpy array but for the following restrictions:

 - the array may not be resized.
 - references of this array should not be kept.
 - slices of this array may not be made.

The numpy array may however be copied and used in any manner.

** Examples **

"""
# For malloc etc.
from libc.stdlib cimport *

cimport numpy as np

import numpy as np

# logging imports
import logging
logger = logging.getLogger()

# 'importing' some Numpy C-api functions.
cdef extern from "numpy/arrayobject.h":
    cdef void  import_array()

    ctypedef struct PyArrayObject:
        char  *data
        np.npy_intp *dimensions

    cdef enum NPY_TYPES:
        NPY_INT,
        NPY_UINT,
        NPY_LONG,
        NPY_FLOAT,
        NPY_DOUBLE

    np.ndarray PyArray_SimpleNewFromData(int, np.npy_intp*, int, void*)


# memcpy
cdef extern from "stdlib.h":
     void *memcpy(void *dst, void *src, long n)

# numpy module initialization call
import_array()

cdef class BaseArray:
    """ Base class for managed C-arrays. """
    ########################################################################
    #Public interface
    ########################################################################
    cpdef str get_c_type(self):
        """ Return the c data type of this array. """
        raise NotImplementedError, 'BaseArray::get_c_type'

    cpdef reserve(self, long size):
        """ Resizes the internal data to required size """
        raise NotImplementedError, 'BaseArray::reserve'

    cpdef resize(self, long size):
        """ reserve()s and sets the length to the new size. """
        raise NotImplementedError, 'BaseArray::resize'

    cpdef np.ndarray get_npy_array(self):
        """ returns a numpy array of the data: do not keep its reference """
        return self._npy_array

    cpdef set_data(self, np.ndarray nparr):
        """ Set data from the given numpy array.

        If the numpy array is a reference to the numpy array maintained
        internally by this class, nothing is done.
        Otherwise, if the size of nparr matches this array, values are
        copied into the array maintained.

        """
        cdef PyArrayObject* sarr = <PyArrayObject*>nparr
        cdef PyArrayObject* darr = <PyArrayObject*>self._npy_array

        if sarr.data == darr.data:
            return
        elif sarr.dimensions[0] <= darr.dimensions[0]:
            self._npy_array[:sarr.dimensions[0]] = nparr
        else:
            raise ValueError, 'array size mismatch'

    cpdef squeeze(self):
        """ Release any unused memory. """
        raise NotImplementedError, 'BaseArray::squeeze'

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
        """ Remove the particles with indices in index_list. """
        raise NotImplementedError, 'BaseArray::remove'

    cpdef extend(self, np.ndarray in_array):
        """ Extend the array with data from in_array. """
        raise NotImplementedError, 'BaseArray::extend'

    cpdef align_array(self, LongArray new_indices):
        """ Rearrange the array contents according to the new indices. """
        self._align_array(new_indices)

    cdef void _align_array(self, LongArray new_indices):
        """ Rearrange the array contents according to the new indices. """
        raise NotImplementedError, 'BaseArray::_align_array'

    cpdef reset(self):
        """ Reset the length of the array to 0. """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        self.length = 0
        arr.dimensions[0] = self.length

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """ Copy values of indexed particles from self to dest. """
        raise NotImplementedError, 'BaseArray::copy_values'

    cpdef copy_subset(self, BaseArray source,
                          long start_index=-1, long end_index=-1):
        """ Copy subset of values from source to self. """
        raise NotImplementedError, 'BaseArray::copy_subset'

    cpdef update_min_max(self):
        """ Update the min and max values of the array. """
        raise NotImplementedError, 'BaseArray::update_min_max'

    def __len__(self):
        return self.length

    def __iter__(self):
        """ Support the iteration protocol"""
        return BaseArrayIter(self)

cdef class BaseArrayIter:
    """ Iteration object to support iteration over BaseArray. """
    def __init__(self, BaseArray arr):
        self.arr = arr
        self.i = -1

    def __next__(self):
        self.i = self.i+1
        if self.i < self.arr.length:
            return self.arr[self.i]
        else:
            raise StopIteration

    def __iter__(self):
        return self


###############################################################################
# `IntArray` class.
###############################################################################
cdef class IntArray(BaseArray):
    """ Represents an array of `int`s """

    #cdef public long length, alloc
    #cdef int *data
    #cdef np.ndarray _npy_array

    def __cinit__(self, long n=0, *args, **kwargs):
        """ Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(int)) and sets up
        the numpy array.

        Parameters:
        -----------
        n -- Length of the array.

        Data attributes:
        ----------------
        data -- Pointer to an integer array.
        alloc -- Size of the data buffer allocated.

        """
        self.length = n
        self._length = 0
        self._parent = None
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <int*>malloc(n*sizeof(int))

        self._setup_npy_array()

    def __dealloc__(self):
        """ Frees the array. """
        if self._parent is None:
            free(<void*>self.data)
        else:
            free(<void*>self._old_data)

    def __getitem__(self, long idx):
        """ Get item at position idx. """
        return self.data[idx]

    def __setitem__(self, long idx, int value):
        """ Set location idx to value. """
        self.data[idx] = value

    cpdef long index(self, int value):
        """ Returns the index at which value is in self, else -1. """
        cdef long i
        for i in range(self.length):
            if self.data[i] == value:
                return i
        return -1

    def __contains__(self, int value):
        """ Returns True if value is in self. """
        return (self.index(value) >= 0)

    def __reduce__(self):
        """ Implemented to facilitate pickling. """
        d = {}
        d['data'] = self.get_npy_array()

        return (IntArray, (), d)

    def __setstate__(self, d):
        """ Load the carray from the dictionary d. """
        cdef np.ndarray arr = d['data']
        self.resize(arr.size)
        self.set_data(arr)

    cdef _setup_npy_array(self):
        """ Create the numpy array. """
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                                    NPY_INT, self.data)

    cpdef str get_c_type(self):
        """ Return the c data type for this array. """
        return 'int'

    cdef int* get_data_ptr(self):
        """ Return the internal data pointer. """
        return self.data

    cpdef int get(self, long idx):
        """ Gets value stored at position idx. """
        return self.data[idx]

    cpdef set(self, long idx, int value):
        """ Sets location idx to value. """
        self.data[idx] = value

    cpdef append(self, int value):
        """ Appends value to the end of the array. """
        cdef long l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, long size):
        """ Resizes the internal data to size*sizeof(int) bytes. """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <int*>realloc(self.data, size*sizeof(int))

            if data == NULL:
                free(<void*>self.data)
                raise MemoryError

            self.data = <int*>data
            self.alloc = size
            arr.data = <char *>self.data

    cpdef reset(self):
        """ Reset the length of the array to 0. """
        BaseArray.reset(self)
        if self._parent is not None:
            self.data = self._old_data
            self._npy_array.data = <char *>self.data
            self._parent = None

    cpdef resize(self, long size):
        """
         Resizes internal data to size*sizeof(int) bytes and sets the
        length to the new size.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot reize array which is a view.')
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef set_view(self, IntArray parent, long start, long end):
        """Create a view of a given a parent array starting at the
        given start index and ending at the end index (this does not
        include the end index).
        """
        if self._parent is None:
            self._old_data = self.data
        self._parent = parent
        self.data = parent.data + start
        self.length = end - start
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        arr.data = <char *>self.data
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """ Release any unused memory. """
        if self._parent is not None:
            raise RuntimeError('Cannot squeeze array which is a view.')

        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        data = <int*>realloc(self.data, self.length*sizeof(int))

        if data == NULL:
            # free original data
            free(<void*>self.data)
            raise MemoryError

        self.data = <int*>data
        self.alloc = self.length
        arr.data = <char *>self.data

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
        """
        Remove the particles with indices in index_list.

        **Parameters**

         - index_list - a list of indices which should be removed.
         - input_sorted - indicates if the input is sorted in ascending order.
           if not, the array will be sorted internally.

        **Algorithm**

         If the input indices are not sorted, sort them in ascending order.
         Starting with the last element in the index list, start replacing the
         element at the said index with the last element in the data and update
         the length of the array.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot remove elements from view array.')
        cdef long i
        cdef long inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long id
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in range(inlength):
            id = sorted_indices[inlength-(i+1)]
            if id < self.length:
                self.data[id] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.

        **Parameters**

         - in_array - a numpy array with data to be added to the current array.

        **Issues**

         - accessing the in_array using the indexing operation seems to be
           costly. Look at the annotated cython html file.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long len = in_array.size
        cdef long i
        for i in range(len):
            self.append(in_array[i])

    cdef void _align_array(self, LongArray new_indices):
        """ Rearrange the array contents according to the new indices. """
        if new_indices.length != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef long i
        cdef long length = self.length
        cdef long n_bytes
        cdef int *temp

        n_bytes = sizeof(int)*length
        temp = <int*>malloc(n_bytes)

        memcpy(<void*>temp, <void*>self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in range(length):
            if i != new_indices.data[i]:
                self.data[i] = temp[new_indices.data[i]]

        free(<void*>temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
        Copies values of indices in indices from self to dest.

        No size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef IntArray dest_array = <IntArray>dest
        cdef long i, num_values
        num_values = indices.length

        for i in range(num_values):
            dest_array.data[i] = self.data[indices.data[i]]

    cpdef copy_subset(self, BaseArray source, long start_index=-1,
                      long end_index=-1):
        """
        Copy a subset of values from src to self.

        **Parameters**

            - start_index - the first index in dest that corresponds to the 0th
            index in source
            - end_index   - the first index in dest from start_index that
            is not copied

        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long si, ei, s_length, d_length, i, j
        cdef IntArray src = <IntArray>source
        s_length = src.length
        d_length = self.length

        if end_index < 0:
            if start_index < 0:
                if s_length != d_length:
                    msg = 'Source length should be same as dest length'
                    logger.error(msg)
                    raise ValueError, msg
                si = 0
                ei = self.length
            else:
                # meaning we copy from the specified start index to the end of
                # self. make sure the sizes are consistent.
                si = start_index
                ei = d_length

                if start_index > (d_length-1):
                    msg = 'start_index beyond array length'
                    logger.error(msg)
                    raise ValueError, msg

                if (ei - si) > s_length:
                    msg = 'Not enough values in source'
                    logger.error(msg)
                    raise ValueError, msg
        else:
            if start_index < 0:
                msg = 'start_index : %d, end_index : %d'%(start_index,
                                                          end_index)
                logger.error(msg)
                raise ValueError, msg
            else:
                if (start_index > (d_length-1) or end_index > d_length or
                    start_index > end_index):
                    msg = 'start_index : %d, end_index : %d'%(start_index,
                                                              end_index)
                    logger.error(msg)
                    raise ValueError, msg

                si = start_index
                ei = end_index

        # we have valid start and end indices now. can start copying now.
        j = 0
        for i in range(si, ei):
            self.data[i] = src.data[j]
            j += 1

    cpdef update_min_max(self):
        """ Updates the min and max values of the array. """
        cdef long i = 0
        cdef int min_val, max_val

        if self.length == 0:
            self.minimum = <int>0
            self.maximum = <int>0
            return

        min_val = self.data[0]
        max_val = self.data[0]

        for i in range(self.length):
            if min_val > self.data[i]:
                min_val = self.data[i]
            if max_val < self.data[i]:
                max_val = self.data[i]

        self.minimum = min_val
        self.maximum = max_val


###############################################################################
# `DoubleArray` class.
###############################################################################
cdef class DoubleArray(BaseArray):
    """ Represents an array of `double`s """

    #cdef public long length, alloc
    #cdef double *data
    #cdef np.ndarray _npy_array

    def __cinit__(self, long n=0, *args, **kwargs):
        """ Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(double)) and sets up
        the numpy array.

        Parameters:
        -----------
        n -- Length of the array.

        Data attributes:
        ----------------
        data -- Pointer to an integer array.
        alloc -- Size of the data buffer allocated.

        """
        self.length = n
        self._length = 0
        self._parent = None
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <double*>malloc(n*sizeof(double))

        self._setup_npy_array()

    def __dealloc__(self):
        """ Frees the array. """
        if self._parent is None:
            free(<void*>self.data)
        else:
            free(<void*>self._old_data)

    def __getitem__(self, long idx):
        """ Get item at position idx. """
        return self.data[idx]

    def __setitem__(self, long idx, double value):
        """ Set location idx to value. """
        self.data[idx] = value

    cpdef long index(self, double value):
        """ Returns the index at which value is in self, else -1. """
        cdef long i
        for i in range(self.length):
            if self.data[i] == value:
                return i
        return -1

    def __contains__(self, double value):
        """ Returns True if value is in self. """
        return (self.index(value) >= 0)

    def __reduce__(self):
        """ Implemented to facilitate pickling. """
        d = {}
        d['data'] = self.get_npy_array()

        return (DoubleArray, (), d)

    def __setstate__(self, d):
        """ Load the carray from the dictionary d. """
        cdef np.ndarray arr = d['data']
        self.resize(arr.size)
        self.set_data(arr)

    cdef _setup_npy_array(self):
        """ Create the numpy array. """
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                                    NPY_DOUBLE, self.data)

    cpdef str get_c_type(self):
        """ Return the c data type for this array. """
        return 'double'

    cdef double* get_data_ptr(self):
        """ Return the internal data pointer. """
        return self.data

    cpdef double get(self, long idx):
        """ Gets value stored at position idx. """
        return self.data[idx]

    cpdef set(self, long idx, double value):
        """ Sets location idx to value. """
        self.data[idx] = value

    cpdef append(self, double value):
        """ Appends value to the end of the array. """
        cdef long l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, long size):
        """ Resizes the internal data to size*sizeof(double) bytes. """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <double*>realloc(self.data, size*sizeof(double))

            if data == NULL:
                free(<void*>self.data)
                raise MemoryError

            self.data = <double*>data
            self.alloc = size
            arr.data = <char *>self.data

    cpdef reset(self):
        """ Reset the length of the array to 0. """
        BaseArray.reset(self)
        if self._parent is not None:
            self.data = self._old_data
            self._npy_array.data = <char *>self.data
            self._parent = None

    cpdef resize(self, long size):
        """
         Resizes internal data to size*sizeof(double) bytes and sets the
        length to the new size.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot reize array which is a view.')
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef set_view(self, DoubleArray parent, long start, long end):
        """Create a view of a given a parent array starting at the
        given start index and ending at the end index (this does not
        include the end index).
        """
        if self._parent is None:
            self._old_data = self.data
        self._parent = parent
        self.data = parent.data + start
        self.length = end - start
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        arr.data = <char *>self.data
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """ Release any unused memory. """
        if self._parent is not None:
            raise RuntimeError('Cannot squeeze array which is a view.')

        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        data = <double*>realloc(self.data, self.length*sizeof(double))

        if data == NULL:
            # free original data
            free(<void*>self.data)
            raise MemoryError

        self.data = <double*>data
        self.alloc = self.length
        arr.data = <char *>self.data

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
        """
        Remove the particles with indices in index_list.

        **Parameters**

         - index_list - a list of indices which should be removed.
         - input_sorted - indicates if the input is sorted in ascending order.
           if not, the array will be sorted internally.

        **Algorithm**

         If the input indices are not sorted, sort them in ascending order.
         Starting with the last element in the index list, start replacing the
         element at the said index with the last element in the data and update
         the length of the array.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot remove elements from view array.')
        cdef long i
        cdef long inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long id
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in range(inlength):
            id = sorted_indices[inlength-(i+1)]
            if id < self.length:
                self.data[id] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.

        **Parameters**

         - in_array - a numpy array with data to be added to the current array.

        **Issues**

         - accessing the in_array using the indexing operation seems to be
           costly. Look at the annotated cython html file.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long len = in_array.size
        cdef long i
        for i in range(len):
            self.append(in_array[i])

    cdef void _align_array(self, LongArray new_indices):
        """ Rearrange the array contents according to the new indices. """
        if new_indices.length != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef long i
        cdef long length = self.length
        cdef long n_bytes
        cdef double *temp

        n_bytes = sizeof(double)*length
        temp = <double*>malloc(n_bytes)

        memcpy(<void*>temp, <void*>self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in range(length):
            if i != new_indices.data[i]:
                self.data[i] = temp[new_indices.data[i]]

        free(<void*>temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
        Copies values of indices in indices from self to dest.

        No size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef DoubleArray dest_array = <DoubleArray>dest
        cdef long i, num_values
        num_values = indices.length

        for i in range(num_values):
            dest_array.data[i] = self.data[indices.data[i]]

    cpdef copy_subset(self, BaseArray source, long start_index=-1,
                      long end_index=-1):
        """
        Copy a subset of values from src to self.

        **Parameters**

            - start_index - the first index in dest that corresponds to the 0th
            index in source
            - end_index   - the first index in dest from start_index that
            is not copied

        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long si, ei, s_length, d_length, i, j
        cdef DoubleArray src = <DoubleArray>source
        s_length = src.length
        d_length = self.length

        if end_index < 0:
            if start_index < 0:
                if s_length != d_length:
                    msg = 'Source length should be same as dest length'
                    logger.error(msg)
                    raise ValueError, msg
                si = 0
                ei = self.length
            else:
                # meaning we copy from the specified start index to the end of
                # self. make sure the sizes are consistent.
                si = start_index
                ei = d_length

                if start_index > (d_length-1):
                    msg = 'start_index beyond array length'
                    logger.error(msg)
                    raise ValueError, msg

                if (ei - si) > s_length:
                    msg = 'Not enough values in source'
                    logger.error(msg)
                    raise ValueError, msg
        else:
            if start_index < 0:
                msg = 'start_index : %d, end_index : %d'%(start_index,
                                                          end_index)
                logger.error(msg)
                raise ValueError, msg
            else:
                if (start_index > (d_length-1) or end_index > d_length or
                    start_index > end_index):
                    msg = 'start_index : %d, end_index : %d'%(start_index,
                                                              end_index)
                    logger.error(msg)
                    raise ValueError, msg

                si = start_index
                ei = end_index

        # we have valid start and end indices now. can start copying now.
        j = 0
        for i in range(si, ei):
            self.data[i] = src.data[j]
            j += 1

    cpdef update_min_max(self):
        """ Updates the min and max values of the array. """
        cdef long i = 0
        cdef double min_val, max_val

        if self.length == 0:
            self.minimum = <double>0
            self.maximum = <double>0
            return

        min_val = self.data[0]
        max_val = self.data[0]

        for i in range(self.length):
            if min_val > self.data[i]:
                min_val = self.data[i]
            if max_val < self.data[i]:
                max_val = self.data[i]

        self.minimum = min_val
        self.maximum = max_val


###############################################################################
# `FloatArray` class.
###############################################################################
cdef class FloatArray(BaseArray):
    """ Represents an array of `float`s """

    #cdef public long length, alloc
    #cdef float *data
    #cdef np.ndarray _npy_array

    def __cinit__(self, long n=0, *args, **kwargs):
        """ Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(float)) and sets up
        the numpy array.

        Parameters:
        -----------
        n -- Length of the array.

        Data attributes:
        ----------------
        data -- Pointer to an integer array.
        alloc -- Size of the data buffer allocated.

        """
        self.length = n
        self._length = 0
        self._parent = None
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <float*>malloc(n*sizeof(float))

        self._setup_npy_array()

    def __dealloc__(self):
        """ Frees the array. """
        if self._parent is None:
            free(<void*>self.data)
        else:
            free(<void*>self._old_data)

    def __getitem__(self, long idx):
        """ Get item at position idx. """
        return self.data[idx]

    def __setitem__(self, long idx, float value):
        """ Set location idx to value. """
        self.data[idx] = value

    cpdef long index(self, float value):
        """ Returns the index at which value is in self, else -1. """
        cdef long i
        for i in range(self.length):
            if self.data[i] == value:
                return i
        return -1

    def __contains__(self, float value):
        """ Returns True if value is in self. """
        return (self.index(value) >= 0)

    def __reduce__(self):
        """ Implemented to facilitate pickling. """
        d = {}
        d['data'] = self.get_npy_array()

        return (FloatArray, (), d)

    def __setstate__(self, d):
        """ Load the carray from the dictionary d. """
        cdef np.ndarray arr = d['data']
        self.resize(arr.size)
        self.set_data(arr)

    cdef _setup_npy_array(self):
        """ Create the numpy array. """
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                                    NPY_FLOAT, self.data)

    cpdef str get_c_type(self):
        """ Return the c data type for this array. """
        return 'float'

    cdef float* get_data_ptr(self):
        """ Return the internal data pointer. """
        return self.data

    cpdef float get(self, long idx):
        """ Gets value stored at position idx. """
        return self.data[idx]

    cpdef set(self, long idx, float value):
        """ Sets location idx to value. """
        self.data[idx] = value

    cpdef append(self, float value):
        """ Appends value to the end of the array. """
        cdef long l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, long size):
        """ Resizes the internal data to size*sizeof(float) bytes. """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <float*>realloc(self.data, size*sizeof(float))

            if data == NULL:
                free(<void*>self.data)
                raise MemoryError

            self.data = <float*>data
            self.alloc = size
            arr.data = <char *>self.data

    cpdef reset(self):
        """ Reset the length of the array to 0. """
        BaseArray.reset(self)
        if self._parent is not None:
            self.data = self._old_data
            self._npy_array.data = <char *>self.data
            self._parent = None

    cpdef resize(self, long size):
        """
         Resizes internal data to size*sizeof(float) bytes and sets the
        length to the new size.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot reize array which is a view.')
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef set_view(self, FloatArray parent, long start, long end):
        """Create a view of a given a parent array starting at the
        given start index and ending at the end index (this does not
        include the end index).
        """
        if self._parent is None:
            self._old_data = self.data
        self._parent = parent
        self.data = parent.data + start
        self.length = end - start
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        arr.data = <char *>self.data
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """ Release any unused memory. """
        if self._parent is not None:
            raise RuntimeError('Cannot squeeze array which is a view.')

        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        data = <float*>realloc(self.data, self.length*sizeof(float))

        if data == NULL:
            # free original data
            free(<void*>self.data)
            raise MemoryError

        self.data = <float*>data
        self.alloc = self.length
        arr.data = <char *>self.data

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
        """
        Remove the particles with indices in index_list.

        **Parameters**

         - index_list - a list of indices which should be removed.
         - input_sorted - indicates if the input is sorted in ascending order.
           if not, the array will be sorted internally.

        **Algorithm**

         If the input indices are not sorted, sort them in ascending order.
         Starting with the last element in the index list, start replacing the
         element at the said index with the last element in the data and update
         the length of the array.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot remove elements from view array.')
        cdef long i
        cdef long inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long id
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in range(inlength):
            id = sorted_indices[inlength-(i+1)]
            if id < self.length:
                self.data[id] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.

        **Parameters**

         - in_array - a numpy array with data to be added to the current array.

        **Issues**

         - accessing the in_array using the indexing operation seems to be
           costly. Look at the annotated cython html file.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long len = in_array.size
        cdef long i
        for i in range(len):
            self.append(in_array[i])

    cdef void _align_array(self, LongArray new_indices):
        """ Rearrange the array contents according to the new indices. """
        if new_indices.length != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef long i
        cdef long length = self.length
        cdef long n_bytes
        cdef float *temp

        n_bytes = sizeof(float)*length
        temp = <float*>malloc(n_bytes)

        memcpy(<void*>temp, <void*>self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in range(length):
            if i != new_indices.data[i]:
                self.data[i] = temp[new_indices.data[i]]

        free(<void*>temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
        Copies values of indices in indices from self to dest.

        No size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef FloatArray dest_array = <FloatArray>dest
        cdef long i, num_values
        num_values = indices.length

        for i in range(num_values):
            dest_array.data[i] = self.data[indices.data[i]]

    cpdef copy_subset(self, BaseArray source, long start_index=-1,
                      long end_index=-1):
        """
        Copy a subset of values from src to self.

        **Parameters**

            - start_index - the first index in dest that corresponds to the 0th
            index in source
            - end_index   - the first index in dest from start_index that
            is not copied

        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long si, ei, s_length, d_length, i, j
        cdef FloatArray src = <FloatArray>source
        s_length = src.length
        d_length = self.length

        if end_index < 0:
            if start_index < 0:
                if s_length != d_length:
                    msg = 'Source length should be same as dest length'
                    logger.error(msg)
                    raise ValueError, msg
                si = 0
                ei = self.length
            else:
                # meaning we copy from the specified start index to the end of
                # self. make sure the sizes are consistent.
                si = start_index
                ei = d_length

                if start_index > (d_length-1):
                    msg = 'start_index beyond array length'
                    logger.error(msg)
                    raise ValueError, msg

                if (ei - si) > s_length:
                    msg = 'Not enough values in source'
                    logger.error(msg)
                    raise ValueError, msg
        else:
            if start_index < 0:
                msg = 'start_index : %d, end_index : %d'%(start_index,
                                                          end_index)
                logger.error(msg)
                raise ValueError, msg
            else:
                if (start_index > (d_length-1) or end_index > d_length or
                    start_index > end_index):
                    msg = 'start_index : %d, end_index : %d'%(start_index,
                                                              end_index)
                    logger.error(msg)
                    raise ValueError, msg

                si = start_index
                ei = end_index

        # we have valid start and end indices now. can start copying now.
        j = 0
        for i in range(si, ei):
            self.data[i] = src.data[j]
            j += 1

    cpdef update_min_max(self):
        """ Updates the min and max values of the array. """
        cdef long i = 0
        cdef float min_val, max_val

        if self.length == 0:
            self.minimum = <float>0
            self.maximum = <float>0
            return

        min_val = self.data[0]
        max_val = self.data[0]

        for i in range(self.length):
            if min_val > self.data[i]:
                min_val = self.data[i]
            if max_val < self.data[i]:
                max_val = self.data[i]

        self.minimum = min_val
        self.maximum = max_val


###############################################################################
# `LongArray` class.
###############################################################################
cdef class LongArray(BaseArray):
    """ Represents an array of `long`s """

    #cdef public long length, alloc
    #cdef long *data
    #cdef np.ndarray _npy_array

    def __cinit__(self, long n=0, *args, **kwargs):
        """ Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(long)) and sets up
        the numpy array.

        Parameters:
        -----------
        n -- Length of the array.

        Data attributes:
        ----------------
        data -- Pointer to an integer array.
        alloc -- Size of the data buffer allocated.

        """
        self.length = n
        self._length = 0
        self._parent = None
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <long*>malloc(n*sizeof(long))

        self._setup_npy_array()

    def __dealloc__(self):
        """ Frees the array. """
        if self._parent is None:
            free(<void*>self.data)
        else:
            free(<void*>self._old_data)

    def __getitem__(self, long idx):
        """ Get item at position idx. """
        return self.data[idx]

    def __setitem__(self, long idx, long value):
        """ Set location idx to value. """
        self.data[idx] = value

    cpdef long index(self, long value):
        """ Returns the index at which value is in self, else -1. """
        cdef long i
        for i in range(self.length):
            if self.data[i] == value:
                return i
        return -1

    def __contains__(self, long value):
        """ Returns True if value is in self. """
        return (self.index(value) >= 0)

    def __reduce__(self):
        """ Implemented to facilitate pickling. """
        d = {}
        d['data'] = self.get_npy_array()

        return (LongArray, (), d)

    def __setstate__(self, d):
        """ Load the carray from the dictionary d. """
        cdef np.ndarray arr = d['data']
        self.resize(arr.size)
        self.set_data(arr)

    cdef _setup_npy_array(self):
        """ Create the numpy array. """
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                                    NPY_LONG, self.data)

    cpdef str get_c_type(self):
        """ Return the c data type for this array. """
        return 'long'

    cdef long* get_data_ptr(self):
        """ Return the internal data pointer. """
        return self.data

    cpdef long get(self, long idx):
        """ Gets value stored at position idx. """
        return self.data[idx]

    cpdef set(self, long idx, long value):
        """ Sets location idx to value. """
        self.data[idx] = value

    cpdef append(self, long value):
        """ Appends value to the end of the array. """
        cdef long l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, long size):
        """ Resizes the internal data to size*sizeof(long) bytes. """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <long*>realloc(self.data, size*sizeof(long))

            if data == NULL:
                free(<void*>self.data)
                raise MemoryError

            self.data = <long*>data
            self.alloc = size
            arr.data = <char *>self.data

    cpdef reset(self):
        """ Reset the length of the array to 0. """
        BaseArray.reset(self)
        if self._parent is not None:
            self.data = self._old_data
            self._npy_array.data = <char *>self.data
            self._parent = None

    cpdef resize(self, long size):
        """
         Resizes internal data to size*sizeof(long) bytes and sets the
        length to the new size.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot reize array which is a view.')
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef set_view(self, LongArray parent, long start, long end):
        """Create a view of a given a parent array starting at the
        given start index and ending at the end index (this does not
        include the end index).
        """
        if self._parent is None:
            self._old_data = self.data
        self._parent = parent
        self.data = parent.data + start
        self.length = end - start
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        arr.data = <char *>self.data
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """ Release any unused memory. """
        if self._parent is not None:
            raise RuntimeError('Cannot squeeze array which is a view.')

        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        data = <long*>realloc(self.data, self.length*sizeof(long))

        if data == NULL:
            # free original data
            free(<void*>self.data)
            raise MemoryError

        self.data = <long*>data
        self.alloc = self.length
        arr.data = <char *>self.data

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
        """
        Remove the particles with indices in index_list.

        **Parameters**

         - index_list - a list of indices which should be removed.
         - input_sorted - indicates if the input is sorted in ascending order.
           if not, the array will be sorted internally.

        **Algorithm**

         If the input indices are not sorted, sort them in ascending order.
         Starting with the last element in the index list, start replacing the
         element at the said index with the last element in the data and update
         the length of the array.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot remove elements from view array.')
        cdef long i
        cdef long inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long id
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in range(inlength):
            id = sorted_indices[inlength-(i+1)]
            if id < self.length:
                self.data[id] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.

        **Parameters**

         - in_array - a numpy array with data to be added to the current array.

        **Issues**

         - accessing the in_array using the indexing operation seems to be
           costly. Look at the annotated cython html file.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long len = in_array.size
        cdef long i
        for i in range(len):
            self.append(in_array[i])

    cdef void _align_array(self, LongArray new_indices):
        """ Rearrange the array contents according to the new indices. """
        if new_indices.length != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef long i
        cdef long length = self.length
        cdef long n_bytes
        cdef long *temp

        n_bytes = sizeof(long)*length
        temp = <long*>malloc(n_bytes)

        memcpy(<void*>temp, <void*>self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in range(length):
            if i != new_indices.data[i]:
                self.data[i] = temp[new_indices.data[i]]

        free(<void*>temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
        Copies values of indices in indices from self to dest.

        No size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef LongArray dest_array = <LongArray>dest
        cdef long i, num_values
        num_values = indices.length

        for i in range(num_values):
            dest_array.data[i] = self.data[indices.data[i]]

    cpdef copy_subset(self, BaseArray source, long start_index=-1,
                      long end_index=-1):
        """
        Copy a subset of values from src to self.

        **Parameters**

            - start_index - the first index in dest that corresponds to the 0th
            index in source
            - end_index   - the first index in dest from start_index that
            is not copied

        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long si, ei, s_length, d_length, i, j
        cdef LongArray src = <LongArray>source
        s_length = src.length
        d_length = self.length

        if end_index < 0:
            if start_index < 0:
                if s_length != d_length:
                    msg = 'Source length should be same as dest length'
                    logger.error(msg)
                    raise ValueError, msg
                si = 0
                ei = self.length
            else:
                # meaning we copy from the specified start index to the end of
                # self. make sure the sizes are consistent.
                si = start_index
                ei = d_length

                if start_index > (d_length-1):
                    msg = 'start_index beyond array length'
                    logger.error(msg)
                    raise ValueError, msg

                if (ei - si) > s_length:
                    msg = 'Not enough values in source'
                    logger.error(msg)
                    raise ValueError, msg
        else:
            if start_index < 0:
                msg = 'start_index : %d, end_index : %d'%(start_index,
                                                          end_index)
                logger.error(msg)
                raise ValueError, msg
            else:
                if (start_index > (d_length-1) or end_index > d_length or
                    start_index > end_index):
                    msg = 'start_index : %d, end_index : %d'%(start_index,
                                                              end_index)
                    logger.error(msg)
                    raise ValueError, msg

                si = start_index
                ei = end_index

        # we have valid start and end indices now. can start copying now.
        j = 0
        for i in range(si, ei):
            self.data[i] = src.data[j]
            j += 1

    cpdef update_min_max(self):
        """ Updates the min and max values of the array. """
        cdef long i = 0
        cdef long min_val, max_val

        if self.length == 0:
            self.minimum = <long>0
            self.maximum = <long>0
            return

        min_val = self.data[0]
        max_val = self.data[0]

        for i in range(self.length):
            if min_val > self.data[i]:
                min_val = self.data[i]
            if max_val < self.data[i]:
                max_val = self.data[i]

        self.minimum = min_val
        self.maximum = max_val


###############################################################################
# `UIntArray` class.
###############################################################################
cdef class UIntArray(BaseArray):
    """ Represents an array of `unsigned int`s """

    #cdef public long length, alloc
    #cdef unsigned int *data
    #cdef np.ndarray _npy_array

    def __cinit__(self, long n=0, *args, **kwargs):
        """ Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(unsigned int)) and sets up
        the numpy array.

        Parameters:
        -----------
        n -- Length of the array.

        Data attributes:
        ----------------
        data -- Pointer to an integer array.
        alloc -- Size of the data buffer allocated.

        """
        self.length = n
        self._length = 0
        self._parent = None
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <unsigned int*>malloc(n*sizeof(unsigned int))

        self._setup_npy_array()

    def __dealloc__(self):
        """ Frees the array. """
        if self._parent is None:
            free(<void*>self.data)
        else:
            free(<void*>self._old_data)

    def __getitem__(self, long idx):
        """ Get item at position idx. """
        return self.data[idx]

    def __setitem__(self, long idx, unsigned int value):
        """ Set location idx to value. """
        self.data[idx] = value

    cpdef long index(self, unsigned int value):
        """ Returns the index at which value is in self, else -1. """
        cdef long i
        for i in range(self.length):
            if self.data[i] == value:
                return i
        return -1

    def __contains__(self, unsigned int value):
        """ Returns True if value is in self. """
        return (self.index(value) >= 0)

    def __reduce__(self):
        """ Implemented to facilitate pickling. """
        d = {}
        d['data'] = self.get_npy_array()

        return (UIntArray, (), d)

    def __setstate__(self, d):
        """ Load the carray from the dictionary d. """
        cdef np.ndarray arr = d['data']
        self.resize(arr.size)
        self.set_data(arr)

    cdef _setup_npy_array(self):
        """ Create the numpy array. """
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                                    NPY_UINT, self.data)

    cpdef str get_c_type(self):
        """ Return the c data type for this array. """
        return 'unsigned int'

    cdef unsigned int* get_data_ptr(self):
        """ Return the internal data pointer. """
        return self.data

    cpdef unsigned int get(self, long idx):
        """ Gets value stored at position idx. """
        return self.data[idx]

    cpdef set(self, long idx, unsigned int value):
        """ Sets location idx to value. """
        self.data[idx] = value

    cpdef append(self, unsigned int value):
        """ Appends value to the end of the array. """
        cdef long l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, long size):
        """ Resizes the internal data to size*sizeof(unsigned int) bytes. """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <unsigned int*>realloc(self.data, size*sizeof(unsigned int))

            if data == NULL:
                free(<void*>self.data)
                raise MemoryError

            self.data = <unsigned int*>data
            self.alloc = size
            arr.data = <char *>self.data

    cpdef reset(self):
        """ Reset the length of the array to 0. """
        BaseArray.reset(self)
        if self._parent is not None:
            self.data = self._old_data
            self._npy_array.data = <char *>self.data
            self._parent = None

    cpdef resize(self, long size):
        """
         Resizes internal data to size*sizeof(unsigned int) bytes and sets the
        length to the new size.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot reize array which is a view.')
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef set_view(self, UIntArray parent, long start, long end):
        """Create a view of a given a parent array starting at the
        given start index and ending at the end index (this does not
        include the end index).
        """
        if self._parent is None:
            self._old_data = self.data
        self._parent = parent
        self.data = parent.data + start
        self.length = end - start
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        arr.data = <char *>self.data
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """ Release any unused memory. """
        if self._parent is not None:
            raise RuntimeError('Cannot squeeze array which is a view.')

        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        data = <unsigned int*>realloc(self.data, self.length*sizeof(unsigned int))

        if data == NULL:
            # free original data
            free(<void*>self.data)
            raise MemoryError

        self.data = <unsigned int*>data
        self.alloc = self.length
        arr.data = <char *>self.data

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
        """
        Remove the particles with indices in index_list.

        **Parameters**

         - index_list - a list of indices which should be removed.
         - input_sorted - indicates if the input is sorted in ascending order.
           if not, the array will be sorted internally.

        **Algorithm**

         If the input indices are not sorted, sort them in ascending order.
         Starting with the last element in the index list, start replacing the
         element at the said index with the last element in the data and update
         the length of the array.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot remove elements from view array.')
        cdef long i
        cdef long inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long id
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in range(inlength):
            id = sorted_indices[inlength-(i+1)]
            if id < self.length:
                self.data[id] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.

        **Parameters**

         - in_array - a numpy array with data to be added to the current array.

        **Issues**

         - accessing the in_array using the indexing operation seems to be
           costly. Look at the annotated cython html file.

        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long len = in_array.size
        cdef long i
        for i in range(len):
            self.append(in_array[i])

    cdef void _align_array(self, LongArray new_indices):
        """ Rearrange the array contents according to the new indices. """
        if new_indices.length != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef long i
        cdef long length = self.length
        cdef long n_bytes
        cdef unsigned int *temp

        n_bytes = sizeof(unsigned int)*length
        temp = <unsigned int*>malloc(n_bytes)

        memcpy(<void*>temp, <void*>self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in range(length):
            if i != new_indices.data[i]:
                self.data[i] = temp[new_indices.data[i]]

        free(<void*>temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
        Copies values of indices in indices from self to dest.

        No size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef UIntArray dest_array = <UIntArray>dest
        cdef long i, num_values
        num_values = indices.length

        for i in range(num_values):
            dest_array.data[i] = self.data[indices.data[i]]

    cpdef copy_subset(self, BaseArray source, long start_index=-1,
                      long end_index=-1):
        """
        Copy a subset of values from src to self.

        **Parameters**

            - start_index - the first index in dest that corresponds to the 0th
            index in source
            - end_index   - the first index in dest from start_index that
            is not copied

        """
        if self._parent is not None:
            raise RuntimeError('Cannot extend array which is a view.')
        cdef long si, ei, s_length, d_length, i, j
        cdef UIntArray src = <UIntArray>source
        s_length = src.length
        d_length = self.length

        if end_index < 0:
            if start_index < 0:
                if s_length != d_length:
                    msg = 'Source length should be same as dest length'
                    logger.error(msg)
                    raise ValueError, msg
                si = 0
                ei = self.length
            else:
                # meaning we copy from the specified start index to the end of
                # self. make sure the sizes are consistent.
                si = start_index
                ei = d_length

                if start_index > (d_length-1):
                    msg = 'start_index beyond array length'
                    logger.error(msg)
                    raise ValueError, msg

                if (ei - si) > s_length:
                    msg = 'Not enough values in source'
                    logger.error(msg)
                    raise ValueError, msg
        else:
            if start_index < 0:
                msg = 'start_index : %d, end_index : %d'%(start_index,
                                                          end_index)
                logger.error(msg)
                raise ValueError, msg
            else:
                if (start_index > (d_length-1) or end_index > d_length or
                    start_index > end_index):
                    msg = 'start_index : %d, end_index : %d'%(start_index,
                                                              end_index)
                    logger.error(msg)
                    raise ValueError, msg

                si = start_index
                ei = end_index

        # we have valid start and end indices now. can start copying now.
        j = 0
        for i in range(si, ei):
            self.data[i] = src.data[j]
            j += 1

    cpdef update_min_max(self):
        """ Updates the min and max values of the array. """
        cdef long i = 0
        cdef unsigned int min_val, max_val

        if self.length == 0:
            self.minimum = <unsigned int>0
            self.maximum = <unsigned int>0
            return

        min_val = self.data[0]
        max_val = self.data[0]

        for i in range(self.length):
            if min_val > self.data[i]:
                min_val = self.data[i]
            if max_val < self.data[i]:
                max_val = self.data[i]

        self.minimum = min_val
        self.maximum = max_val


