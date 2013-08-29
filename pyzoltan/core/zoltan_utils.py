"""Utility functions to work with the Zoltan generated lists"""
import numpy
from mpi4py import MPI

def get_num_objects_per_proc(comm, num_objects):
    """Utility function to get number of objects on each processor.

    Parameters:
    -----------

    comm : mpi.Comm
        The communicator (COMM_WORLD)

    num_objects : int
        Number of objects per processor

    This function uses MPI.Allreduce to get, on each processor, an
    array of size 'comm.Get_size()' which stores the number of
    particles per processor. Using this array, we can compute the new
    unique global indices.

    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    send_data = numpy.zeros(shape=size, dtype=numpy.int32)
    send_data[rank] = num_objects

    num_objects_data = numpy.zeros(shape=size, dtype=numpy.int32)
    comm.Allreduce(send_data, num_objects_data, op=MPI.MAX)

    return num_objects_data
