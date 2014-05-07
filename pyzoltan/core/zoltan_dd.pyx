"""Example for the Zoltan Distributed data directory"""
from zoltan cimport _check_error

cdef class Zoltan_DD:
    """A Zoltan Distributed data directory is used as a parallel hash map.

    The hash map is implemented as keys of type ZOLTAN_ID_TYPE which
    can be global indices for the objects being partitioned for
    example. Each entry is owned by a processor and has associated
    with it, additional user data.

    """
    def __init__(self, mpi.Comm comm):
        """Initialize the Zoltan DD (ZDD)

        Parameters:
        -----------

        comm : mpi.Comm
            MPI communicator ( COMM_WORLD )

        All processes must instantiate a copy of the Zoltan DD. A
        pointer to the ZDD is created and stored for furhter
        calls. Upon destruction, Zoltan_Destroy is called to get rid
        of the allocated ZDD

        """
        self.comm = comm.ob_mpi
        ierr = Zoltan_DD_Create(&self.dd, self.comm, 1, 1, 0, 0, 0)

        _check_error( ierr )

    def Zoltan_DD_Update(self, UIntArray gid, IntArray part):
        """Populate the ZDD with some data.

        Parameters:
        ------------

        gid : UIntArray
            Global indices for the keys in the hash map

        part : UIntArray
            Partition/Processor which owns the data

        The ZDD can store additional user data and local indices with
        each hash entry. To skip these, we pass in Cython NULL
        pointers as recommended by the Zoltan user guide.

        """
        cdef int nentries = gid.length
        ierr = Zoltan_DD_Update(self.dd, gid.data, NULL, NULL, part.data, nentries)

        _check_error( ierr )

    def Zoltan_DD_Find(self, UIntArray gid, IntArray part, IntArray own,
                       object lid=None, data=None):
        """Look up ownership and partitions for a given entry.

        Parameters:
        -----------

        gid : UIntArray (in)
            Global indices for requested data

        part : IntArray (out)
            Partition/Processor to which the entry belongs

        own : IntArray (out)
            Partition/Processor to which the associated data belongs.

        lid, data : None (not used)

        """
        cdef int count = gid.length
        cdef int ierr

        # resize the own and part arrays
        own.resize( count )
        part.resize( count )

        ierr = Zoltan_DD_Find( self.dd, gid.data, NULL, NULL,
                               part.data, count, own.data )
        _check_error( ierr )

    def Zoltan_DD_Print(self):
        """Print the contents of the DD"""
        Zoltan_DD_Print( self.dd )

    def __dealloc__(self):
        """Boom!"""
        Zoltan_DD_Destroy( &self.dd )
