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

# The data structure to hold information about objects that are to be
# partitioned using the class of Geometric load balancing algorithms
# is defined in the struct `CoordinateData` in zoltan.pxd. This struct
# is populated with object data to be partitioned and passed as a
# void* to the query functions. 

# Refer to the Zoltan reference manual for a complete list of
# available query functions.
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
    """Return the local and global ids of the objects.

    Methods: RCB, RIB, HSFC

    """
    cdef CoordinateData* _data = <CoordinateData *>data
    cdef int numMyPoints = _data.numMyPoints
    cdef int i

    # check object weight dimensions
    if wgt_dim > 1:
        raise ValueError("Object weight %d not supported"%wgt_dim)

    for i in range (numMyPoints):
        globalID[i] = _data.myGlobalIDs[i]
        localID[i] = <ZOLTAN_ID_TYPE>i

        # set the object weights
        if _data.use_weights:
            obj_wts[i] = <float>_data.obj_wts[i]

cdef int get_num_geom(void* data, int* ierr):
    """Return the dimensionality of the problem."""
    cdef CoordinateData* _data = <CoordinateData *>data
    cdef int dim = _data.dim
    ierr[0] = 0
    return dim

cdef void get_geometry_list(void* data, int sizeGID, int sizeLID, int num_obj,
                            ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                            int num_dim, double* geom_vec, int* ierr):
    """Return the coordinate locations for Zoltan.

    Methods: RCB, RIB, HSFC

    """
    cdef CoordinateData* _data = <CoordinateData *>data
    cdef int i, dim = _data.dim

    if dim == 2:
        for i in range( num_obj ):
            geom_vec[2*i + 0] = _data.x[i]
            geom_vec[2*i + 1] = _data.y[i]

    elif dim == 3:
        for i in range( num_obj ):
            geom_vec[3*i + 0] = _data.x[i]
            geom_vec[3*i + 1] = _data.y[i]
            geom_vec[3*i + 2] = _data.z[i]

    else:
        raise ValueError("Dimension %d invalid for PyZoltan!"%dim)

#########################################################################
# Zoltan Wrapper
#########################################################################
cdef class PyZoltan:
    """Base class for the Python wrapper for Zoltan

    All Zoltan partitioners are derived from PyZoltan. This class sets
    up the basic arrays used (import/export lists) for load balancing,
    methods to set Zoltan parameters and the actual load balancing
    method itself. 

    The specific Zoltan interface functions that are exposed through
    this class are:

     - Zoltan_LB_Balance : The main load balancing routine. 
         Upon return, a list of indices (local, global) to 
         be imported and exported are available.

     - Zoltan_Invert_Lists : Invert import/export lists
         Upon return, a given set of lists is inverted.

     - Zoltan_Set_Param : Set the value of a Zoltan parameter
    
    """
    def __init__(self, object comm, str obj_weight_dim="0",
                 str edge_weight_dim="0", debug_level="0",
                 str return_lists="ALL"):
        """Initialize the Zoltan wrapper

        Parameters:
        
        comm : MPI communicator
            MPI communicator to be used

        obj_weight_dim : str, default "0"
            Weight dimensions for the object. The default value assumes 
            all objects have equal weight.            

        edge_weight_dim : str, default "0"
            Weight dimensions for the edge. The default value assumes all 
            edges have equal weight.

        debug_level : str, default "0"
            Zoltan debug level. Values in the range -1 < 0 < 11 are accepted

        return_lists : str, default "ALL"
            Kind of lists to be returned by Zoltan

        Notes:
        
        Instantiation of the PyZoltan object initializes the Zoltan
        library, creates the Zoltan struct ubiquitous in Zoltan calls
        and also sets up the import/export lists that will be used for
        the data exchange. It also sets up some reasonable default
        values. 

        In general though, any parameter can be set using the
        Zoltan_Set_Param function wrapper

        """
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
        self.edge_weight_dim = edge_weight_dim
        self.obj_weight_dim = obj_weight_dim
        self.return_lists = return_lists
        self.debug_level = debug_level
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
        """Reset all Zoltan Import/Export lists"""
        self.exportGlobalids.reset()
        self.exportLocalids.reset()
        self.exportProcs.reset()

        self.importGlobalids.reset()
        self.importLocalids.reset()
        self.importProcs.reset()

        self.numExport = 0
        self.numImport = 0

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

    #######################################################################
    # Private interface
    #######################################################################
    cdef _set_Zoltan_lists(self,
                           int numExport,
                           ZOLTAN_ID_PTR _exportGlobal,
                           ZOLTAN_ID_PTR _exportLocal,
                           int* _exportProcs,
                           int numImport,
                           ZOLTAN_ID_PTR _importGlobal,
                           ZOLTAN_ID_PTR _importLocal,
                           int* _importProcs):
        "Copy the import/export lists returned by Zoltan"
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
        """Import/Export lists used by Zoltan"""
        self.exportGlobalids = UIntArray()
        self.exportLocalids = UIntArray()
        self.exportProcs = IntArray()

        self.importGlobalids = UIntArray()
        self.importLocalids = UIntArray()
        self.importProcs = IntArray()

        self.procs = np.ones(shape=self.size, dtype=np.int32)
        self.parts = np.ones(shape=self.size, dtype=np.int32)

        # object weight arrays
        self.weights = DoubleArray()

    def _set_default(self):
        "Set reasonable default values"
        self.Zoltan_Set_Param("DEBUG_LEVEL", self.debug_level)

        self.Zoltan_Set_Param("OBJ_WEIGHT_DIM", self.obj_weight_dim)

        self.Zoltan_Set_Param("EDGE_WEIGHT_DIM", self.edge_weight_dim)

        self.Zoltan_Set_Param("RETURN_LISTS", self.return_lists)

    def _set_data(self):
        raise NotImplementedError("PyZoltan::_set_data should not be called!")

    def __dealloc__(self):
        self.Zoltan_Destroy()

cdef class ZoltanGeometricPartitioner(PyZoltan):
    """Concrete implementation of PyZoltan using the geometric algorithms.

    Use the ZoltanGeometricPartitioner to load balance/partition a set
    of objects defined by their coordinates (x, y & z) and an array of
    unique global indices. Additionally, each object can also have a
    weight associated with it.

    """
    def __init__(self, int dim, object comm, DoubleArray x, DoubleArray y,
                 DoubleArray z, UIntArray gid,
                 str obj_weight_dim="0",
                 str return_lists="ALL",
                 str lb_method="RCB",
                 str keep_cuts="1"
                 ):
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

        obj_weight_dim : str
            Optional weights for the objects (this should be 1 at most)

        return_lists : str
            Specify lists requested from Zoltan (Import/Export)

        lb_method : str
            String specifying the load balancing method to use

        keep_cuts : str
            Parameter used for adding items to a decomposition

        """
        # sanity check
        if not ( x.length == y.length == z.length ):
            raise ValueError('Coordinate data (x, y, z) lengths not equal!')

        # values needed for defaults
        self.lb_method = lb_method
        self.keep_cuts = keep_cuts

        # Base class initialization
        super(ZoltanGeometricPartitioner, self).__init__(
            comm, obj_weight_dim=obj_weight_dim, return_lists=return_lists)

        # set the problem dimensionality
        self.dim = dim

        # set the data arrays
        self.x = x; self.y = y; self.z = z; self.gid = gid

        # number of local objects. This is taken equal to the length
        # of the data supplied initially
        self.num_local_objects = num_local_objects = x.length

        # object weights. If obj_weight_dim == "0" this array should be 0
        self.weights.resize( num_local_objects )

        # register the query functions with Zoltan
        self._Zoltan_register_query_functions()

    #######################################################################
    # Private interface
    #######################################################################
    def Zoltan_Box_PP_Assign(self, double xmin, double ymin, double zmin,
                             double xmax, double ymax, double zmax):
        """Find the processors that intersect with a box

        For Zoltan, given a domain decomposition using a geometric
        algorithm, we can use Zoltan_Box_PP_Assign to find processors
        that intersect with a rectilinear box defined by the values
        xmin, .... zmax

        """
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

    def Zoltan_Point_PP_Assign(self, double x, double y, double z):
        """Find to which processor a given point must be sent to

        For Zoltan, given a domain decomposition using a geometric
        algorithm, we can use Zoltan_Point_PP_Assign to find a
        processor in the decomposition to which a given point (x, y,
        z) belongs to

        """
        cdef Zoltan_Struct* zz = self._zstruct.zz

        cdef int ierr, proc = -1, part = -1
        cdef double[3] coords

        coords[0] = x; coords[1] = y; coords[2] = z

        ierr = czoltan.Zoltan_LB_Point_PP_Assign(
            zz, coords, &proc, &part)

        _check_error(ierr)

        return proc

    #######################################################################
    # Private interface
    #######################################################################
    def _Zoltan_register_query_functions(self):
        """Register query functions for the Geometric based partitioners

        The Geometric based partitioners are the simplest kind of
        dynamic load balancing algorithms provided by Zoltan. These
        require four callbakcs to be registered, two geometry based
        and two object base callbakcs respectively:

        Num_Obj_Fn : Returns the number of objects assigned locally

        Obj_List_Fn : Populates Zoltan allocated arrays with local and
        global indices for objects assigned locally

        Num_Geom_Fn : Returns the number of objects used to represent
        the geometry of an object (2 for 2D applications, 3 for 3D
        applications etc)

        Geom_Multi_Fn : Populates Zoltan allocated arrays with
        geometry information (x, y, z, weights) of the objects
        assigned locally.

        """
        cdef Zoltan_Struct* zz = self._zstruct.zz
        cdef int err

        # Num_Obj_Fn
        err = czoltan.Zoltan_Set_Num_Obj_Fn(
            zz, &get_number_of_objects, <void*>&self._cdata)

        _check_error(err)

        # Obj_List_Fn
        err = czoltan.Zoltan_Set_Obj_List_Fn(
            zz, &get_obj_list, <void*>&self._cdata)

        _check_error(err)

        # Num_Geom_Fn
        err = czoltan.Zoltan_Set_Num_Geom_Fn(
            zz, &get_num_geom, <void*>&self._cdata)

        _check_error(err)

        # Geom_Multi_Fn
        err = czoltan.Zoltan_Set_Geom_Multi_Fn(
            zz, &get_geometry_list, <void*>&self._cdata)

        _check_error(err)

    def _set_data(self):
        """Set the user defined particle data structure for Zoltan.

        This is called just before load balancing to update the user
        defined data structure (CoordinateData) for Zoltan.

        """
        self._cdata.dim = <int>self.dim
        self._cdata.numGlobalPoints = <int>self.num_global_objects
        self._cdata.numMyPoints = <int>self.num_local_objects

        self._cdata.myGlobalIDs = self.gid.data
        self._cdata.x = self.x.data
        self._cdata.y = self.y.data
        self._cdata.z = self.z.data

        cdef int i
        cdef DoubleArray weights = self.weights

        # set the weights
        self._cdata.obj_wts = weights.data

        self._cdata.use_weights = True
        if self.obj_weight_dim == "0":
            self._cdata.use_weights = False

    def _set_default(self):
        """Resonable defaults?"""
        PyZoltan._set_default(self)

        self.Zoltan_Set_Param("LB_METHOD", self.lb_method)

        self.Zoltan_Set_Param("KEEP_CUTS", self.keep_cuts)
