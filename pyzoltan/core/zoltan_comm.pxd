import numpy as np
cimport numpy as np

# Zoltan comm objects and headers
from pyzoltan.czoltan cimport czoltan_comm as zcomm

cdef class ZComm:
    # pointer to the zoltan comm object
    cdef zcomm.ZOLTAN_COMM_OBJ* _zoltan_comm_obj

    # mpi communicator, rank and size
    cdef public object comm
    cdef public int rank, size

    # tag for the communicator
    cdef public int tag

    # number of objects to send
    cdef public int nsend

    # processor list for particles to be exported
    cdef public np.ndarray proclist

    # number of objects to be received
    cdef public int nreturn

    # size of each element and dtype
    cdef public int nbytes
    cdef public object dtype
