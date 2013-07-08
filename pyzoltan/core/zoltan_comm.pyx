"""Wrapper for the Zoltan Unstructured Communication Package"""
cimport mpi4py.MPI as mpi
from mpi4py cimport mpi_c as mpic

# Cython for pure mode
cimport cython

# NUMPY
import numpy as np
cimport numpy as np

# Zoltan error checking
from zoltan cimport _check_error

cdef class ZComm:
    def __init__(self, object comm, int tag, int nsend, np.ndarray proclist):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        self.tag = tag
        self.nsend = nsend
        self.proclist = proclist.astype(np.int32)

        self.nreturn = 0
    
        # the size of each element to exchange
        self.nbytes = 8

        self.initialize()

    def __dealloc__(self):
        cdef zcomm.ZOLTAN_COMM_OBJ* _zoltan_comm_obj = self._zoltan_comm_obj
        zcomm.Zoltan_Comm_Destroy(&_zoltan_comm_obj)
        
    def initialize(self):
        cdef zcomm.ZOLTAN_COMM_OBJ** zoltan_comm_obj = &self._zoltan_comm_obj
        
        cdef np.ndarray[ndim=1, dtype=np.int32_t] _proclist = self.proclist
        cdef int* proclist = <int*>_proclist.data

        cdef mpi.Comm _comm = self.comm
        cdef mpic.MPI_Comm comm = _comm.ob_mpi

        cdef int nsend = self.nsend, tag = self.tag

        # cython declarations
        cdef int ierr, _nreturn = -1
        cdef int* nreturn = &_nreturn

        # create the communication plan
        ierr = zcomm.Zoltan_Comm_Create(zoltan_comm_obj,
                                        nsend,
                                        proclist,
                                        comm,
                                        tag,
                                        nreturn)

        # check for the error
        _check_error(ierr)

        # save the number of objects to be returned
        self.nreturn = _nreturn

    def Comm_Do(self, np.ndarray _sendbuf, np.ndarray _recvbuf):
        cdef zcomm.ZOLTAN_COMM_OBJ* _zoltan_comm_obj = self._zoltan_comm_obj
        cdef char* send_data = _sendbuf.data
        cdef char* recv_data = _recvbuf.data
        cdef int ierr, tag = self.tag, nbytes = self.nbytes

        ierr = zcomm.Zoltan_Comm_Do(
            _zoltan_comm_obj,
            tag, 
            send_data,
            nbytes,
            recv_data)

        _check_error(ierr)
        
    def set_nbytes(self, int nbytes):
        self.nbytes = nbytes

    def set_tag(self, int tag):
        self.tag = tag
