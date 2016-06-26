# Numpy
cimport numpy as np

from cpython cimport dict
from cpython cimport list

# PyZoltan
from pyzoltan.core.carray cimport UIntArray, IntArray, DoubleArray, LongArray
from pyzoltan.core.zoltan cimport PyZoltan, ZoltanGeometricPartitioner
from pyzoltan.core.zoltan_comm cimport ZComm
from pyzoltan.czoltan.czoltan_types cimport ZOLTAN_ID_TYPE, ZOLTAN_ID_PTR, ZOLTAN_OK

# PySPH imports
from pysph.base.nnps_base cimport NNPSParticleArrayWrapper
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport *

cdef class ParticleArrayExchange:
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef public int msglength_tag_remote           # msg length tag for remote_exchange
    cdef public int data_tag_remote                # data tag for remote_exchange

    cdef public int msglength_tag_lb               # msg length tag for lb_exchange
    cdef public int data_tag_lb                    # data tag for lb_exchange

    cdef public int pa_index                       # Particle index
    cdef public ParticleArray pa                   # Particle data
    cdef public NNPSParticleArrayWrapper pa_wrapper    # wrapper to exchange data

    # flags to indicate whether data needs to be exchanged
    cdef public bint lb_exchange
    cdef public bint remote_exchange

    cdef public size_t num_local         # Total number of particles
    cdef public size_t num_global        # Global number of particles
    cdef public size_t num_remote        # Number of remote particles
    cdef public size_t num_ghost         # Number of ghost particles


    # mpi.Comm object and associated rank and size
    cdef public object comm
    cdef public int rank, size

    # list of load balancing props
    cdef public list lb_props
    cdef public int nprops

    # Import/Export lists for particles
    cdef public UIntArray exportParticleGlobalids
    cdef public UIntArray exportParticleLocalids
    cdef public IntArray exportParticleProcs
    cdef public int numParticleExport

    cdef public UIntArray importParticleGlobalids
    cdef public UIntArray importParticleLocalids
    cdef public IntArray importParticleProcs
    cdef public int numParticleImport

    ############################################################################
    # Member functions
    ############################################################################
    # exchange data given send and receive lists
    cdef exchange_data(self, ZComm zcomm, dict sendbufs, int count)

# base class for all parallel managers
cdef class ParallelManager:
    ############################################################################
    # Data Attributes
    ############################################################################
    # mpi comm, rank and size
    cdef public object comm
    cdef public int rank
    cdef public int size

    cdef public np.ndarray dt_sendbuf    # array for global reduction
    cdef public np.ndarray dt_recvbuf    # for time step calculation

    cdef public int lb_freq              # load balancing frequency
    cdef public int lb_count             # counter for current lb step

    cdef public int ncells_local         # number of local cells
    cdef public int ncells_remote        # number of remote cells
    cdef public int ncells_total         # total number of cells
    cdef public list cell_list           # list of cells

    cdef public dict cell_map            # index structure
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

    cdef public bint initial_update
    cdef public bint update_cell_sizes

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
    cdef _bin(self, int pa_index, UIntArray indices)

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
                                size_t d_idx, UIntArray nbrs)

# Zoltan based parallel cell manager for SPH simulations
cdef class ZoltanParallelManager(ParallelManager):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef public int changes              # logical (0,1) if the partition changes
    cdef public PyZoltan pz              # the PyZoltan wrapper for lb etc

# Class of geometric load balancers
cdef class ZoltanParallelManagerGeometric(ZoltanParallelManager):
    pass
