"""Utility functions to work with the Zoltan generated lists"""
import numpy
from mpi4py import MPI

def count_recv_data(
    comm, recv, numImport, importProcs):
    """Count the data to be received from different processors.

    Parameters:
    -----------

    comm : mpi.Comm
        MPI communicator

    recv : dict
        Upon output, will contain keys corresponding to processors and
        values indicating number of objects to receive from that proc.

    numImport : int
        Zoltan generated total number of objects to be imported
        to the calling proc

    importProcs : DoubleArray
        Zoltan generated list for processors from where objects are
        to be received.

    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    recv.clear()
    for processor in range(size):
        recv[processor] = 0

    for i in range(numImport):
        processor = importProcs[i]
        recv[processor] += 1
        
    for processor in recv.keys():
        if recv[processor] == 0:
            del recv[processor]

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
