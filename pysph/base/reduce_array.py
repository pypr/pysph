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
    if isinstance(array, BaseArray):
        np_array = array.get_npy_array()
    else:
        np_array = array
    return ops[op](np_array)


def mpi_reduce_array(array, op='sum'):
    """Reduce an array given an array and a suitable reduction operation.

    Currently, only 'sum', 'max', 'min' and 'prod' are supported.

    **Parameters**

     - array: numpy.ndarray: Any numpy array (1D).
     - op: str: reduction operation, one of ('sum', 'prod', 'min', 'max')

    """
    value = serial_reduce_array(array, op)
    from mpi4py import MPI
    ops = {'sum': MPI.SUM, 'prod': MPI.PROD,
           'max': MPI.MAX, 'min': MPI.MIN}
    return MPI.COMM_WORLD.allreduce(value, op=ops[op])


try:
    import mpi4py
except ImportError:
    reduce_array = serial_reduce_array
else:
    reduce_array = mpi_reduce_array
