cimport mpi4py.MPI as mpi
if MPI4PY_V2:
   from mpi4py cimport libmpi as mpic
else:
   from mpi4py cimport mpi_c as mpic

# Zoltan imports
from pyzoltan.czoltan cimport czoltan
from pyzoltan.czoltan.czoltan cimport Zoltan_Struct

# Zoltan type imports
from pyzoltan.czoltan.czoltan_types cimport ZOLTAN_ID_PTR, ZOLTAN_ID_TYPE, \
     ZOLTAN_OK, ZOLTAN_WARN, ZOLTAN_FATAL, ZOLTAN_MEMERR

# NUMPY
import numpy as np
cimport numpy as np

# Carrays
from carray cimport UIntArray, IntArray, LongArray, DoubleArray

# Compatibility for older MPI versions and later
# mpi4py releases (Ubuntu 14.04 is one such).
cdef extern from 'mpi-compat.h': pass

# Error checking for Zoltan
cdef _check_error(int ierr)

# Pointer to the Zoltan struct
cdef struct _Zoltan_Struct:
    czoltan.Zoltan_Struct* zz

cdef class PyZoltan:
    # problem dimensionsionality
    cdef public int dim

    # version number
    cdef public double version

    # mpi.Comm object and associated rank and size
    cdef public object comm
    cdef public int rank, size

    # Pointer to the Zoltan structure upon creation
    cdef _Zoltan_Struct _zstruct

    # string to store the current load balancing method
    cdef public str lb_method

    # Arrays returned by Zoltan
    cdef public UIntArray exportGlobalids
    cdef public UIntArray exportLocalids
    cdef public IntArray exportProcs

    cdef public UIntArray importGlobalids
    cdef public UIntArray importLocalids
    cdef public IntArray importProcs

    # the number of objects to import/export
    cdef public int numImport, numExport

    cdef public np.ndarray procs             # processors of range size
    cdef public np.ndarray parts             # partitions of range size

    # data array for the object weights
    cdef public DoubleArray weights

    # General Zoltan parameters (refer the user guide)
    cdef public str debug_level
    cdef public str obj_weight_dim
    cdef public str edge_weight_dim
    cdef public str return_lists

    ###############################################################
    # Member functions
    ###############################################################
    # after a load balance, copy the Zoltan allocated lists to local
    # numpy arrays. The Zoltan lists are subsequently deallocated
    cdef _set_zoltan_lists(
        self,
        int numExport,                          # number of objects to export
        ZOLTAN_ID_PTR _exportGlobal,            # global indices of export objects
        ZOLTAN_ID_PTR _exportLocal,             # local indices of export objects
        int* _exportProcs,                      # target processors to export
        int numImport,                          # number of objects to import
        ZOLTAN_ID_PTR _importGlobal,            # global indices of import objects
        ZOLTAN_ID_PTR _importLocal,             # local indices of import objects
        int* _importProcs                       # target processors to import
        )

    # Invert the export lists. Given a situation where every processor
    # knows which objects must be exported to remote processors, a
    # call to invert lists will return a list of objects that must be
    # imported from remote processors.
    cpdef Zoltan_Invert_Lists(self)

# User defined data for the RCB, RIB and HSFC methods
cdef struct CoordinateData:
    # flag for using weights
    bint use_weights

    # dimensionality of the problem
    int dim

    # number of local/global points
    int numGlobalPoints
    int numMyPoints

    # pointers to the object data
    ZOLTAN_ID_PTR myGlobalIDs
    double* obj_wts
    double* x
    double* y
    double *z

cdef class ZoltanGeometricPartitioner(PyZoltan):
    # data arrays for the coordinates
    cdef public DoubleArray x, y, z

    # data array for the global indices
    cdef public UIntArray gid

    # User defined structure to hold the coordinate data for the
    # Zoltan interface
    cdef CoordinateData _cdata

    # number of global and local objects
    cdef public int num_global_objects, num_local_objects

    # ZOLTAN parameters for Geometric partitioners
    cdef public str keep_cuts
