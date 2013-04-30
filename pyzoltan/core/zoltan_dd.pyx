"""Example for the Zoltan Distributed data directory"""
from zoltan cimport _check_error

cdef class Zoltan_DD:
    def __init__(self, mpi.Comm comm):

        self.comm = comm.ob_mpi
        ierr = Zoltan_DD_Create(&self.dd, self.comm, 1, 1, 0, 0, 0)

        _check_error( ierr )

    def Zoltan_DD_Update(self, UIntArray gid, IntArray part):
        cdef int nentries = gid.length
        ierr = Zoltan_DD_Update(self.dd, gid.data, NULL, NULL, part.data, nentries)

        _check_error( ierr )

    def Zoltan_DD_Print(self):
        Zoltan_DD_Print( self.dd )

    def Zoltan_DD_Find(self, UIntArray gid, IntArray part, IntArray own,
                       object lid=None, data=None):
        cdef int count = gid.length
        cdef int ierr

        # resize the own and part arrays
        own.resize( count )
        part.resize( count )

        ierr = Zoltan_DD_Find( self.dd, gid.data, NULL, NULL,
                               part.data, count, own.data )
        _check_error( ierr )
        
    def __dealloc__(self):
        Zoltan_DD_Destroy( &self.dd )
