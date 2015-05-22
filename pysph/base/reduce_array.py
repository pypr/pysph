"""Functions to reduce array data in serial or parallel.
"""

import numpy as np

from pyzoltan.core.carray import BaseArray


def _check_operation(op):
    """Raise an exception if the wrong operation is given.
    """
    valid_ops = ('sum', 'max', 'min', 'prod')
    msg = "Unsupported operation %s, must be one of %s."%(op, valid_ops)
    if op not in valid_ops:
        raise RuntimeError(msg)

def _get_npy_array(array_or_carray):
    """Return a numpy array from given carray or numpy array.
    """
    if isinstance(array_or_carray, BaseArray):
        return array_or_carray.get_npy_array()
    else:
        return array_or_carray

def serial_reduce_array(array, op='sum'):
    """Reduce an array given an array and a suitable reduction operation.

    Currently, only 'sum', 'max', 'min' and 'prod' are supported.

    **Parameters**

     - array: numpy.ndarray: Any numpy array (1D).
     - op: str: reduction operation, one of ('sum', 'prod', 'min', 'max')

    """
    _check_operation(op)
    ops = {'sum': np.sum, 'prod': np.prod,
           'max': np.max, 'min': np.min}
    np_array = _get_npy_array(array)
    return ops[op](np_array)


def dummy_reduce_array(array, op='sum'):
    """Simply returns the array for the serial case.
    """
    return _get_npy_array(array)

def mpi_reduce_array(array, op='sum'):
    """Reduce an array given an array and a suitable reduction operation.

    Currently, only 'sum', 'max', 'min' and 'prod' are supported.

    **Parameters**

     - array: numpy.ndarray: Any numpy array (1D).
     - op: str: reduction operation, one of ('sum', 'prod', 'min', 'max')

    """
    np_array = _get_npy_array(array)
    from mpi4py import MPI
    ops = {'sum': MPI.SUM, 'prod': MPI.PROD,
           'max': MPI.MAX, 'min': MPI.MIN}
    return MPI.COMM_WORLD.allreduce(np_array, op=ops[op])

# This is just to keep syntax highlighters happy in editors while writing
# equations.
parallel_reduce_array = mpi_reduce_array
