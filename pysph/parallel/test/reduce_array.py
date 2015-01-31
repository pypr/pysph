"""Test if the mpi_reduce_array function works correctly.
"""

import mpi4py.MPI as mpi
import numpy as np

from pysph.base.reduce_array import serial_reduce_array, mpi_reduce_array

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n = 5
data = np.ones(n)*(rank + 1)

full_data = []
for i in range(size):
    full_data = np.concatenate([full_data, np.ones(n)*(i+1)])

for op in ('sum', 'prod', 'min', 'max'):
    serial_data = serial_reduce_array(data, op)
    result = mpi_reduce_array(serial_data, op)
    expect = getattr(np, op)(full_data)
    msg = "For op %s: Expected %s, got %s"%(op, expect, result)
    assert expect == result, msg
