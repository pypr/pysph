"""Wrapper for the Zoltan Unstructured Communication Package

The Unstructured Communication utilities simplifies some common
point-to-point message passing paradigms required for dynamic
applications.

The main wrapper object for this package is the ZComm object which is
described below.

"""
cimport mpi4py.MPI as mpi
if MPI4PY_V2:
   from mpi4py cimport libmpi as mpic
else:
   from mpi4py cimport mpi_c as mpic

# Cython for pure mode
cimport cython

# NUMPY
import numpy as np
cimport numpy as np

# Zoltan error checking
from zoltan cimport _check_error

cdef class ZComm:
    """Wrapper for simplified unstructured point-to-point communication

    The ZComm wrapper is used when each processor knows a list of
    objects it must send to other processors but does not know of what
    objects it must receive from them. It is designed to work with
    NumPy arrays for convenience.

    Processor 0 sends data (uint32) [4, 9, 6] to processors [1, 2, 2]
    Processor 1 sends data (uint32) [5, 3] to processors [0, 0]
    Processor 2 sends data (uint32) [1, 9, 0, 2] to processors [0, 1, 1, 0]

    Note that the size of data to be sent is different on each
    processor. Each processor instantiates the data and the ZComm
    object like so:

    >>> sendbuf = numpy.array( [...], dtype=numpy.uint32 )
    >>> proclist = numpy.array( [...], dtype=numpy.int32 )
    >>> nsend = sendbuf.size

    >>> zcomm = ZComm(comm, tag=0, nsend=nsend, proclist=proclist)

    """
    def __init__(self, object comm, int tag, int nsend, np.ndarray proclist):
        """Constructor for the ZComm object

        Parameters:

        comm : MPI Comm
            MPI communicator (mpi4py.MPI.COMM_WORLD)

        tag : int
            Message tag for the unstructured communication

        nsend : int
            Number of objects this processor has to send

        proclist : numpy.ndarray
            Array of size 'nsend' indicating where each object in the
            sendbuf (to be defined) will be sent.

        Notes:

        In the general case, every processor has some data it knows it
        must share with remote processors.

        As an example, we may have that processor 2 sends 4 ojects of
        data to processors to [0, 3, 1, 0]. In this example, the first
        object is sent to processor 0, the second to processor 3 and
        so on and so forth. The ZComm for this processor can be
        constructed as

        >>> nsend = 4
        >>> proclist = numpy.array( [0, 3, 1, 0], dtype=nupmy.int32 )
        >>> zcomm = ZComm(mpi.COMM_WORLD, nsend=nsend, tag=0, proclist=proclist)

        Upon instantiation, the zcomm object will know of the number
        of objects it must receive from all remote processors:

        >>> nrecv = zcomm.nreturn

        Knowing the number of objects that must be received, we can
        allocate buffers of the appropriate size to collect this
        data. The actual transfer of data is effected with a call to
        the `ZComm.Comm_Do` method

        We can use the same ZComm object for multiple exchanges of
        data of different types as long as the number of objects to be
        exchanged are the same. For example, a processor may want to
        exchange solution data (doubles) along with indices
        (uints). The same plan can be used in this case with the
        provision to alter the number of bytes per object with the
        method `ZComm.set_nbytes`

        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.tag = tag
        self.nsend = nsend
        self.proclist = proclist.astype(np.int32)

        self.nreturn = 0

        # the size of each element to exchange
        self.nbytes = 8

        # internally call Zoltan_Comm_Create
        self.initialize()

    def __dealloc__(self):
        "Cleanup"
        cdef zcomm.ZOLTAN_COMM_OBJ* _zoltan_comm_obj = self._zoltan_comm_obj
        zcomm.Zoltan_Comm_Destroy(&_zoltan_comm_obj)

    def initialize(self):
        """Zoltan_Comm_Create

        This function calls the Zoltan_Comm_Create function to create
        the unstructured communication plan. This plan can
        subsequently be used to effect data transfers in an
        unstructured manner between processors.

        Upon return, each processor knows the number of objects that
        are to be received through the data attribute `nreturn`

        """
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
        """Perform an unstructured communication between processors

        Parameters:

        _sendbuf : np.ndarray
            The array of data to be sent by this processor

        _recvbuf : np.ndarray
            The array of data to be received by this processor

        Notes:

        Internally, Zoltan_Comm_Do accepts char* buffers to move the
        data between processors. The number of objects is determined
        by the `nbytes` argument.

        The `nsend` argument used to create the ZComm object and the
        `nbytes` argument should be consistent to avoid strange
        behaviour.

        """
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

    def Comm_Do_Reverse(self, np.ndarray _sendbuf, np.ndarray recvbuf):
        """Perform the reverse of the unstructured communication
        between processors

        Parameters:

        _sendbuf : np.ndarray
            The array of data to be sent by this processor

        Notes:

        Internally, Zoltan_Comm_Do accepts char* buffers to move the
        data between processors. The number of objects is determined
        by the `nbytes` argument.

        """
        cdef zcomm.ZOLTAN_COMM_OBJ* _zoltan_comm_obj = self._zoltan_comm_obj
        cdef char* send_data = _sendbuf.data
        cdef int ierr, tag = self.tag, nbytes = self.nbytes

        # the returned data will be updated. The number of objects to
        # send for Do_Reverse is therefore equal to nreturn
        cdef int nsend = self.nreturn

        # sizes pointer is null for equal sized objects
        cdef int* sizesp = NULL

        #cdef np.ndarray recvbuf = np.zeros( self.nsend, dtype=dtype )
        cdef char* _recvbuf = recvbuf.data

        # Zoltan interface function
        ierr = zcomm.Zoltan_Comm_Do_Reverse(
            _zoltan_comm_obj,
            tag,
            send_data,
            nbytes,
            sizesp,
            _recvbuf)

        _check_error(ierr)

        #return recvbuf

    def set_nbytes(self, int nbytes, object dtype=None):
        "Set the number of bytes for each object"
        self.nbytes = nbytes
        self.dtype = dtype

    def set_tag(self, int tag):
        "Set the message tag for this plan"
        self.tag = tag
