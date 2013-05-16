"""PyZoltan wrapper"""
cimport mpi4py.MPI as mpi
from mpi4py cimport mpi_c as mpic

# Cython for pure mode
cimport cython

# NUMPY
import numpy as np
cimport numpy as np

# malloc and friends
from libc.stdlib cimport malloc, free

# Zoltan config imports
ZOLTAN_UNSIGNED_INT=True
try:
    from pyzoltan.czoltan.czoltan_config cimport UNSIGNED_INT_GLOBAL_IDS
except ImportError:
    ZOLTAN_UNSIGNED_INT=False

# Python standard library imports
from warnings import warn

# Local imports
import zoltan_utils

def get_zoltan_id_type_max():
    if ZOLTAN_UNSIGNED_INT:
        return (1<<32) - 1

cdef extern from "limits.h":
    cdef unsigned int UINT_MAX
    cdef int INT_MAX
    cdef int INT_MIN

# Zoltan error function
cdef _check_error(int ierr):
    if ierr == ZOLTAN_WARN:
        warn("ZOTLAN WARNING")

    if ierr == ZOLTAN_FATAL:
        raise RuntimeError("Zoltan FATAL error!")

    if ierr == ZOLTAN_MEMERR:
        raise MemoryError("Zoltan MEMERR error!")

###############################################################
# ZOLTAN QUERY FUNCTIONS FOR GEOMETRIC PARTITIONING
###############################################################
cdef int get_number_of_objects(void* data, int* ierr):
    """Return the number of local objects on a processor.

    Methods: RCB, RIB, HSFC

    """
    cdef CoordinateData* _data = <CoordinateData *>data
    return _data.numMyPoints

cdef void get_obj_list(void* data, int sizeGID, int sizeLID,
                       ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                       int wgt_dim, float* obj_wts, int* ierr):
    """Return the local and global ids of the particles.

    Methods: RCB, RIB, HSFC

    """
    cdef CoordinateData* _data = <CoordinateData *>data
    cdef int numMyPoints = _data.numMyPoints
    cdef int i

    for i in range (numMyPoints):
        globalID[i] = _data.myGlobalIDs[i]
        localID[i] = <ZOLTAN_ID_TYPE>i

cdef int get_num_geom(void* data, int* ierr):
    """Return the dimensionality of the problem."""
    ierr[0] = 0
    return 2

cdef void get_geometry_list(void* data, int sizeGID, int sizeLID, int num_obj,
                            ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                            int num_dim, double* geom_vec, int* ierr):
    """Return the coordinate locations for Zoltan.

    Methods: RCB, RIB, HSFC

    """
    cdef CoordinateData* _data = <CoordinateData *>data
    cdef int i

    for i in range( num_obj ):
        geom_vec[2*i + 0] = _data.x[i]
        geom_vec[2*i + 1] = _data.y[i]

#########################################################################
# The actual Zoltan Wrapper
#########################################################################
cdef class PyZoltan:
    def __init__(self, object comm):
        """Initialize the Zoltan wrapper."""
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # initialize Zoltan
        self.version = self.Zoltan_Initialize()

        # Create the Zoltan struct
        self.Zoltan_Create(comm)

        # setup the required arrays
        self._setup_zoltan_arrays()

        # set default values
        self._set_default()

    #######################################################################
    # Public interface
    #######################################################################
    def set_num_local_objects(self, int num_local_objects):
        """Set the number of local objects"""
        self.num_local_objects = num_local_objects

    def set_num_global_objects(self, int num_global_objects):
        """Set the number of global objects"""
        self.num_global_objects = num_global_objects        

    def Zoltan_Initialize(self, int argc=0, args=''):
        """Initialize Zoltan"""
        cdef float version
        cdef char **c_argv
        
        args = [ bytes(x) for x in args ]
        c_argv = <char**>malloc( sizeof(char*) *len(args) )
        if c_argv is NULL:
            raise MemoryError()
        try:
            for idx, s in enumerate( args ):
                c_argv[idx] = s
        finally:
            free( c_argv )

        # call the Zoltan Init function
        error_code = cython.declare(cython.int)
        error_code = czoltan.Zoltan_Initialize(len(args), c_argv, &version)
        _check_error(error_code)
        return version        

    def Zoltan_Create(self, mpi.Comm comm):
        """Create the Zoltan struct"""
        cdef mpic.MPI_Comm _comm = comm.ob_mpi

        cdef czoltan.Zoltan_Struct* zz = czoltan.Zoltan_Create( _comm )
        self._zstruct.zz = zz

    def Zoltan_Set_Param(self, str _name, str _value):
        """Set a general Zoltan Parameter"""
        cdef bytes tmp_name = _name.encode()
        cdef bytes tmp_value = _value.encode()

        cdef char* name = tmp_name
        cdef char* value = tmp_value

        cdef czoltan.Zoltan_Struct* zz = self._zstruct.zz
        czoltan.Zoltan_Set_Param( zz, name, value )

    def set_lb_method(self, str value):
        """Set the Zoltan load balancing method"""
        cdef str name = "LB_METHOD"
        self.lb_method = value
        
        self.Zoltan_Set_Param(name, value)
        self.ZOLTAN_LB_METHOD = value

    def Zoltan_Destroy(self):
        """Destroy the Zoltan struct"""
        czoltan.Zoltan_Destroy( &self._zstruct.zz )

    def Zoltan_LB_Balance(self):
        """Call the Zoltan load balancing function.
        
        After a call to this function, we get the import/export lists
        required for load balancing.

        """
        cdef Zoltan_Struct* zz = self._zstruct.zz

        # set the object data. We must ensure that the global ids are
        # unique and properly set up before calling LB_Balance
        self._set_data()

        # initialize the data buffers for input to Zoltan
        cython.declare(changes=cython.int, numGidEntries=cython.int,
                       numLidEntries=cython.int, numImport=cython.int,
                       numExport=cython.int, ierr=cython.int)

        cython.declare(importGlobal=ZOLTAN_ID_PTR,importLocal=ZOLTAN_ID_PTR,
                       exportGlobal=ZOLTAN_ID_PTR,exportLocal=ZOLTAN_ID_PTR)
    
        cython.declare(importProcs=cython.p_int, exportProcs=cython.p_int)

        # call the load balance function
        ierr = czoltan.Zoltan_LB_Balance(
            zz,
            cython.address(changes),
            cython.address(numGidEntries),
            cython.address(numLidEntries),
            cython.address(numImport),
            &importGlobal,
            &importLocal,
            &importProcs,
            cython.address(numExport),
            &exportGlobal,
            &exportLocal,
            &exportProcs
            )

        _check_error(ierr)

        # Copy the Zoltan allocated lists locally
        self.reset_Zoltan_lists()
        self._set_Zoltan_lists(numExport,
                               exportGlobal,
                               exportLocal,
                               exportProcs,
                               numImport,
                               importGlobal,
                               importLocal,
                               importProcs)
        
        # free the Zoltan allocated data
        ierr = czoltan.Zoltan_LB_Free_Data(
            &importGlobal,
            &importLocal,
            &importProcs,
            &exportGlobal,
            &exportLocal,
            &exportProcs
            )

        _check_error(ierr)

        # return changes to determine if we need to do data movement
        return changes

    def reset_Zoltan_lists(self):
        """Reset all Zoltan interface lists"""
        self.exportGlobalids.reset()
        self.exportLocalids.reset()
        self.exportProcs.reset()

        self.importGlobalids.reset()
        self.importLocalids.reset()
        self.importProcs.reset()

        self.numExport = 0
        self.numImport = 0

    cdef _set_Zoltan_lists(self,
                           int numExport,
                           ZOLTAN_ID_PTR _exportGlobal,
                           ZOLTAN_ID_PTR _exportLocal,
                           int* _exportProcs,
                           int numImport,
                           ZOLTAN_ID_PTR _importGlobal,
                           ZOLTAN_ID_PTR _importLocal,
                           int* _importProcs):
        """Get the import/export lists returned by Zoltan."""
        cdef int i

        cdef UIntArray exportGlobalids = self.exportGlobalids
        cdef UIntArray exportLocalids = self.exportLocalids
        cdef IntArray exportProcs = self.exportProcs

        cdef UIntArray importGlobalids = self.importGlobalids
        cdef UIntArray importLocalids = self.importLocalids
        cdef IntArray importProcs = self.importProcs

        # set the values for the number of import and export objects
        self.numImport = numImport; self.numExport = numExport

        # resize the PyZoltan import lists
        importGlobalids.resize(numImport)
        importLocalids.resize(numImport)
        importProcs.resize(numImport)

        # resize the PyZoltan export lists
        exportGlobalids.resize(numExport)
        exportLocalids.resize(numExport)
        exportProcs.resize(numExport)

        # set the Import/Export lists
        for i in range(numExport):
            exportGlobalids.data[i] = _exportGlobal[i]
            exportLocalids.data[i] = _exportLocal[i]
            exportProcs.data[i] = _exportProcs[i]

        for i in range(numImport):
            importGlobalids.data[i] = _importGlobal[i]
            importLocalids.data[i] = _importLocal[i]
            importProcs.data[i] = _importProcs[i]

    cpdef Zoltan_Invert_Lists(self):
        """Invert export lists to get import lists

        At times, we know which particles to export without any
        information aobut import requirements. Two situations in which
        this arises is in computing neighbors for geometric
        partitioners and load balancing using cell lists.

        """
        cdef Zoltan_Struct* zz = self._zstruct.zz
        cdef UIntArray exportGlobalids = self.exportGlobalids
        cdef UIntArray exportLocalids = self.exportLocalids
        cdef IntArray exportProcs = self.exportProcs

        cdef UIntArray importGlobalids = self.importGlobalids
        cdef UIntArray importLocalids = self.importLocalids
        cdef IntArray importProcs = self.importProcs
        
        cdef int numExport = self.numExport
        cdef int i, ierr

        # declare the import arrays
        cython.declare(_importGlobalids=ZOLTAN_ID_PTR,
                       _importLocalids=ZOLTAN_ID_PTR,
                       _importProcs=cython.p_int,
                       numImport=cython.int)

        ierr = czoltan.Zoltan_Compute_Destinations(
            zz,
            numExport,
            exportGlobalids.data,
            exportLocalids.data,
            exportProcs.data,
            &numImport,
            &_importGlobalids,
            &_importLocalids,
            &_importProcs,
            )
        
        _check_error(ierr)

        # save the data in the local import lists
        importGlobalids.resize(numImport)
        importLocalids.resize(numImport)
        importProcs.resize(numImport)

        for i in range(numImport):
            importGlobalids.data[i] = _importGlobalids[i]
            importLocalids.data[i] = _importLocalids[i]
            importProcs.data[i] = _importProcs[i]

        self.numImport = numImport

        # free the Zoltan allocated lists
        ierr = czoltan.Zoltan_LB_Free_Part(
            &_importGlobalids,
            &_importLocalids,
            &_importProcs,
            NULL,
            )

        _check_error(ierr)

    def _update_gid(self, UIntArray gid):
        """Update the unique global indices.

        We call a utility function to get the new number of particles
        across the processors and then linearly assign indices to the
        objects.

        """
        cdef int num_global_objects, num_local_objects, _sum, i

        cdef np.ndarray[ndim=1, dtype=np.int32_t] num_objects_data

        cdef mpi.Comm comm = self.comm
        cdef int rank = self.rank
        cdef int size = self.size

        num_objects_data = zoltan_utils.get_num_objects_per_proc(
             comm, self.num_local_objects)
        
        num_local_objects = num_objects_data[ rank ]
        num_global_objects = np.sum( num_objects_data )

        _sum = np.sum( num_objects_data[:rank] )

        gid.resize( num_local_objects )
        for i in range( num_local_objects ):
            gid.data[i] = <ZOLTAN_ID_TYPE> ( _sum + i )

        self.num_global_objects = num_global_objects
        self.num_local_objects = num_local_objects        

    def _setup_zoltan_arrays(self):
        self.exportGlobalids = UIntArray()
        self.exportLocalids = UIntArray()
        self.exportProcs = IntArray()

        self.importGlobalids = UIntArray()
        self.importLocalids = UIntArray()
        self.importProcs = IntArray()

        self.procs = np.ones(shape=self.size, dtype=np.int32)
        self.parts = np.ones(shape=self.size, dtype=np.int32)

    def _set_default(self):
        self.ZOLTAN_DEBUG_LEVEL = "1"
        self.Zoltan_Set_Param("DEBUG_LEVEL", "0")
        
        self.ZOLTAN_OBJ_WEIGHT_DIM = "0"
        self.Zoltan_Set_Param("OBJ_WEIGHT_DIM", "0")
        
        self.ZOLTAN_EDGE_WEIGHT_DIM = "0"
        self.Zoltan_Set_Param("EDGE_WEIGHT_DIM", "0")
        
        self.ZOLTAN_RETURN_LISTS = "ALL"
        self.Zoltan_Set_Param("RETURN_LISTS", "ALL")

    def _set_data(self):
        raise NotImplementedError("PyZoltan::_set_data should not be called!")

    def __dealloc__(self):
        self.Zoltan_Destroy()

cdef class ZoltanGeometricPartitioner(PyZoltan):
    """Concrete implementation of PyZoltan using the geometric algorithms.

    Use the ZoltanGeometricPartitioner to load balance/partition a set
    of objects defined by their coordinates (x, y & z) and an array of
    unique global indices.

    """
    def __init__(self, int dim, object comm, DoubleArray x, DoubleArray y,
                 DoubleArray z, UIntArray gid):
        """Constructor

        Parameters:
        -----------

        dim : int
            Problem dimensionality

        comm : mpi4py.MPI.Comm
            MPI communicator (typically COMM_WORLD)

        x, y, z : DoubleArray
            Coordinate arrays for the objects to be partitioned

        gid : UIntArray
            Global indices for the objects to be partitioned
        """
        super(ZoltanGeometricPartitioner, self).__init__(comm)
        self.dim = dim

        self.x = x
        self.y = y
        self.z = z
        self.gid = gid

        self.num_local_objects = x.length

        # register the query functions with Zoltan
        self._Zoltan_register_query_functions()

    def Zoltan_Box_PP_Assign(self, double xmin, double ymin, double zmin,
                             double xmax, double ymax, double zmax):
        """Find the processors that intersect with a box"""
        cdef Zoltan_Struct* zz = self._zstruct.zz

        cdef np.ndarray[ndim=1, dtype=np.int32_t] procs = self.procs
        cdef np.ndarray[ndim=1, dtype=np.int32_t] parts = self.parts

        cdef int numprocs = 0
        cdef int numparts = 0

        cdef int ierr

        # initialize procs and parts
        procs[:] = -1
        parts[:] = -1

        ierr = czoltan.Zoltan_LB_Box_PP_Assign(
            zz,
            xmin, ymin, zmin,
            xmax, ymax, zmax,
            <int*>procs.data, &numprocs,
            <int*>parts.data, &numparts
            )

        _check_error(ierr)

        return numprocs

    #######################################################################
    # Private interface
    #######################################################################
    def _Zoltan_register_query_functions(self):
        """Query functions for the Geometric based partitioners."""
        cdef Zoltan_Struct* zz = self._zstruct.zz
        cdef int err

        # number of objects function
        err = czoltan.Zoltan_Set_Num_Obj_Fn(
            zz, &get_number_of_objects, <void*>&self._cdata)

        _check_error(err)

        # object list function
        err = czoltan.Zoltan_Set_Obj_List_Fn(
            zz, &get_obj_list, <void*>&self._cdata)

        _check_error(err)

        # geom num geom function
        err = czoltan.Zoltan_Set_Num_Geom_Fn(
            zz, &get_num_geom, <void*>&self._cdata)

        _check_error(err)

        # geom multi function
        err = czoltan.Zoltan_Set_Geom_Multi_Fn(
            zz, &get_geometry_list, <void*>&self._cdata)

        _check_error(err)

    def _set_data(self):
        """Set the user defined particle data structure for Zoltan.

        This is called just before load balancing to update the user
        defined particle data structure for Zoltan. The reason this
        needs to be called is because the particle information keeps
        changing for each time step.

        """
        self._cdata.numGlobalPoints = <int>self.num_global_objects
        self._cdata.numMyPoints = <int>self.num_local_objects
        
        self._cdata.myGlobalIDs = self.gid.data
        self._cdata.x = self.x.data
        self._cdata.y = self.y.data

    def _set_default(self):
        """Resonable defaults?"""
        PyZoltan._set_default(self)

        self.ZOLTAN_LB_METHOD = "RCB"
        self.Zoltan_Set_Param("LB_METHOD", "RCB")

        self.ZOLTAN_KEEP_CUTS = "1"
        self.Zoltan_Set_Param("KEEP_CUTS", "1")
