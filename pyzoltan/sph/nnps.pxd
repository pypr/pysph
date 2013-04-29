# Numpy
cimport numpy as np

# PyZoltan
from pyzoltan.core.carray cimport UIntArray, IntArray, DoubleArray, LongArray
from pyzoltan.sph.particle_array cimport ParticleArray
from pyzoltan.core.point cimport *
from pyzoltan.czoltan.zoltan_types cimport ZOLTAN_ID_TYPE, ZOLTAN_ID_PTR, ZOLTAN_OK
from pyzoltan.czoltan cimport czoltan
from pyzoltan.core.zoltan cimport ZoltanGeometricPartitioner

cdef inline int real_to_int(double val, double step)
cdef inline cIntPoint find_cell_id(cPoint pnt, double cell_size)
cpdef UIntArray arange_uint(int start, int stop=*)

cdef class DomainLimits:
    cdef public double xmin, xmax
    cdef public double ymin, ymax
    cdef public double zmin, zmax

    cdef public double xtranslate
    cdef public double ytranslate
    cdef public double ztranslate

    cdef public int dim
    cdef public bint is_periodic

cdef class Cell:
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef cIntPoint _cid                 # Spatial index for the cell
    cdef public bint is_boundary        # Flag to indicate boundary cells
    cdef long nparticles                # Number of particles in the cell         
    cdef public UIntArray lindices      # Local indices for particles
    cdef public UIntArray gindices      # Global indices for binned particles
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
    cpdef set_indices(self, UIntArray lindices, UIntArray gindices)

    # compute the bounding box for a cell. Layers is used to determine
    # the factor times the cell size the bounding box is offset from
    # the cell.
    cdef _compute_bounding_box(self, double cell_size,
                               int layers)

cdef class NNPSParticleGeometric(ZoltanGeometricPartitioner):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef public ParticleArray pa         # particle array data
    cdef DoubleArray h                   # Data arrays required for binning
    cdef IntArray tag                    # tag array

    cdef bint in_parallel                # Flag to determine if in parallel

    cdef public size_t num_particles     # Total number of particles 
    cdef public size_t num_remote        # Number of remote particles
    cdef public size_t num_ghost         # Number of ghost particles
    cdef public size_t num_global        # Global number of particles
    cdef public DomainLimits domain      # Domain limits for the geometry

    cdef public int ghost_layers         # BOunding box size
    cdef public dict cells
    cdef public bint is_periodic         # Flag to indicate periodicity
    cdef public double cell_size         # Cell size for binning
    cdef public double radius_scale      # Radius scale for kernel

    cdef np.ndarray hmax_recvbuf         # Array of maximum local h's
    cdef np.ndarray hmax_sendbuf         # send buffer of max h's

    # Zoltan interface definitions
    cdef public list lb_props                # list of load balancing props

    ############################################################################
    # Member functions
    ############################################################################
    # Index particles given by a list of indices. The indices are
    # assumed to be of type unsigned int and local to the NNPS object
    cdef _bin(self, UIntArray indices)

    # Index additional ghost (remote) particle data after we have
    # communicated this information. The indices for this call is
    # taken to be the difference in lengths between the current size
    # of the arrays and the old (num_particles) size
    cpdef remote_bin(self)

    # Main binning routine for NNPS for local particles. This clears
    # the current cell data, re-computes the cell size and bins all
    # particles locally.
    cpdef local_bin(self)

    # Compute the cell size across processors. The cell size is taken
    # as max(h)*radius_scale
    cpdef compute_cell_size(self)

    # Re-compute the bin structure after a load balancing step.
    cpdef update_data(self)

    # Add remote particles to the local bins after they have
    # been exchanged. 
    cpdef update_remote_particle_data(self)

    # Compute the particles that need to be exported to remote
    # processors to satisfy their neighbor requirements for binning.
    cpdef compute_remote_particles(self)

    # exchange data given send and receive lists
    cdef _exchange_data(self, int count, dict send, dict recv)

    #######################################################
    # Functions for periodicity
    #######################################################
    # adjust particles for periodicity to keep them in the domain.
    cpdef adjust_particles(self)

    # create ghost particles.
    cpdef _create_ghosts(self)

    #######################################################
    # Functions for neighbor searching
    #######################################################
    # Neighbor query function. Returns the list of neighbors for a
    # requested particle. The returned list is assumed to be of type
    # unsigned int to follow the type of the local and global ids.
    cpdef get_nearest_particles(self, size_t i,
                                UIntArray nbrs)    

    # Testing function for brute force neighbor search. The return
    # list is of the same type of the local and global ids (uint)
    cpdef brute_force_neighbors(self, size_t i,
                                UIntArray nbrs)
