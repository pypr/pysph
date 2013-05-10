# Numpy
cimport numpy as np

# PyZoltan Imports
from pyzoltan.core.zoltan cimport PyZoltan, ZoltanGeometricPartitioner
from pyzoltan.core.carray cimport UIntArray, IntArray, DoubleArray, LongArray
from pyzoltan.czoltan.czoltan_types cimport ZOLTAN_ID_TYPE, ZOLTAN_ID_PTR, ZOLTAN_OK

# PySPH imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport *

cdef inline int real_to_int(double val, double step)
cdef inline cIntPoint find_cell_id(cPoint pnt, double cell_size)
cpdef UIntArray arange_uint(int start, int stop=*)

# Simple ParticleArray wrapper class to hold particle data. For NNPS,
# we hold only essential data like x, y, z, h and gid
cdef class ParticleArrayWrapper:
    cdef ParticleArray pa
    cdef public str name
    cdef public DoubleArray x, y, z, h
    cdef public UIntArray gid
    cdef public IntArray tag

# Domain limits
cdef class DomainLimits:
    cdef public double xmin, xmax
    cdef public double ymin, ymax
    cdef public double zmin, zmax

    cdef public double xtranslate
    cdef public double ytranslate
    cdef public double ztranslate

    cdef public int dim
    cdef public bint is_periodic

# basic cell data structure
cdef class Cell:
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef cIntPoint _cid                 # Spatial index for the cell
    cdef public bint is_boundary        # Flag to indicate boundary cells
    cdef int narrays                    # number of arrays
    cdef public list lindices           # Local indices for particles
    cdef public list gindices           # Global indices for binned particles
    cdef list nparticles                # Number of particles in the cell
    cdef double cell_size               # bin size
    cdef public cPoint centroid         # Centroid computed from indices
    cdef cPoint boxmin                  # Bounding box min for the cell
    cdef cPoint boxmax                  # Bounding box max for the cell
    cdef int layers                     # Layers to compute bounding box
    cdef IntArray nbrprocs              # List of neighboring processors

    ############################################################################
    # Member functions
    ############################################################################
    # set the indices for the cell
    cpdef set_indices(self, int index, UIntArray lindices, UIntArray gindices)

    # compute the bounding box for a cell. Layers is used to determine
    # the factor times the cell size the bounding box is offset from
    # the cell.
    cdef _compute_bounding_box(self, double cell_size,
                               int layers)

cdef class ParticleArrayExchange:
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef public ParticleArray pa                   # Particle data
    cdef public ParticleArrayWrapper pa_wrapper    # wrapper for data access

    cdef public size_t num_local         # Total number of particles
    cdef public size_t num_global        # Global number of particles
    cdef public size_t num_remote        # Number of remote particles
    cdef public size_t num_ghost         # Number of ghost particles

    # mpi.Comm object and associated rank and size
    cdef public object comm
    cdef public int rank, size    

    # Zoltan interface definitions
    cdef public list lb_props                      # list of load balancing props

    # Zoltan Import/Export lists for particles
    cdef public UIntArray exportParticleGlobalids
    cdef public UIntArray exportParticleLocalids
    cdef public IntArray exportParticleProcs
    cdef public int numParticleExport

    cdef public UIntArray importParticleGlobalids
    cdef public UIntArray importParticleLocalids
    cdef public IntArray importParticleProcs
    cdef public int numParticleImport

    cdef public DoubleArray doublebuf        # temp buffer to import data
    cdef public UIntArray uintbuf            # temp buffer to import id info
    cdef public IntArray intbuf              # temp buffer for integer arrays
    cdef public LongArray longbuf            # temp buffer for long arrays

    ############################################################################
    # Member functions
    ############################################################################
    # exchange data given send and receive lists
    cdef _exchange_data(self, int count, dict send, dict recv)

# base class for all parallel managers
cdef class ParallelManager:
    ############################################################################
    # Data Attributes
    ############################################################################
    # mpi comm, rank and size
    cdef public object comm
    cdef public int rank
    cdef public int size

    cdef public dict cells               # index structure
    cdef public int ghost_layers         # BOunding box size
    cdef public double cell_size         # cell size used for binning

    # list of particle arrays, wrappers, exchange and nnps instances
    cdef public list particles
    cdef public list pa_wrappers
    cdef public list pa_exchanges

    # number of local and remote particles
    cdef public list num_local
    cdef public list num_remote
    cdef public list num_global

    cdef public double radius_scale      # Radius scale for kernel

    # number of arrays
    cdef int narrays

    # boolean for parallel
    cdef bint in_parallel

    # Min and max across all processors
    cdef np.ndarray minx, miny, minz     # min and max arrays used 
    cdef np.ndarray maxx, maxy, maxz     # for MPI Allreduce 
    cdef np.ndarray maxh                 # operations

    cdef public double mx, my, mz        # global min and max values
    cdef public double Mx, My, Mz, Mh

    # global indices for the cells
    cdef UIntArray cell_gid

    # cell coordinate values
    cdef DoubleArray cx, cy, cz

    # Import/Export lists for cells
    cdef public UIntArray exportCellGlobalids
    cdef public UIntArray exportCellLocalids
    cdef public IntArray exportCellProcs
    cdef public int numCellExport

    cdef public UIntArray importCellGlobalids
    cdef public UIntArray importCellLocalids
    cdef public IntArray importCellProcs
    cdef public int numCellImport    

    ############################################################################
    # Member functions
    ############################################################################
    # Index particles given by a list of indices. The indices are
    # assumed to be of type unsigned int and local to the NNPS object
    cdef _bin(self, int index, UIntArray indices)
    
    # Compute the cell size across processors. The cell size is taken
    # as max(h)*radius_scale
    cpdef compute_cell_size(self)

    # compute global bounds for the particle distribution. The bounds
    # are the min and max coordinate values across all processors and
    # the maximum smoothing length needed for parallel binning.
    cdef _compute_bounds(self)

    # nearest neighbor search routines taking into account multiple
    # particle arrays
    cpdef get_nearest_particles(self, int src_index, int dst_index,
                                size_t i, UIntArray nbrs)    

# Zoltan based parallel cell manager for SPH simulations
cdef class ZoltanParallelManager(ParallelManager):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef public PyZoltan pz              # the PyZoltan wrapper for lb etc

# Class of geometric load balancers
cdef class ZoltanParallelManagerGeometric(ZoltanParallelManager):
    pass
