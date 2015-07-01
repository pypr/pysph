# numpy
cimport numpy as np

from libcpp.map cimport map

# PyZoltan CArrays
from pyzoltan.core.carray cimport UIntArray, IntArray, DoubleArray, LongArray

# local imports
from particle_array cimport ParticleArray
from point cimport *

cdef inline int real_to_int(double val, double step) nogil
cdef cIntPoint find_cell_id(cPoint pnt, double cell_size)

cpdef UIntArray arange_uint(int start, int stop=*)

# Basic particle array wrapper used for NNPS
cdef class NNPSParticleArrayWrapper:
    cdef public DoubleArray x,y,z,h
    cdef public UIntArray gid
    cdef public IntArray tag
    cdef public ParticleArray pa
    cdef str name
    cdef int np

    # get the number of particles
    cdef int get_number_of_particles(self)

# Domain limits for the simulation
cdef class DomainManager:
    cdef public double xmin, xmax
    cdef public double ymin, ymax
    cdef public double zmin, zmax

    cdef public double xtranslate
    cdef public double ytranslate
    cdef public double ztranslate

    cdef public int dim
    cdef public bint periodic_in_x, periodic_in_y, periodic_in_z
    cdef public bint is_periodic

    cdef public list pa_wrappers        # NNPS particle array wrappers
    cdef public int narrays             # number of arrays
    cdef public double cell_size        # distance to create ghosts
    cdef bint in_parallel               # Flag to determine if in parallel
    cdef public double radius_scale     # Radius scale for kernel

    # remove ghost particles from a previous iteration
    cdef _remove_ghosts(self)

    # box-wrap particles within the physical domain
    cdef _box_wrap_periodic(self)

    # Convenience function to add a value to a carray
    cdef _add_to_array(self, DoubleArray arr, double disp)

    # create new ghosts
    cdef _create_ghosts_periodic(self)

    # Compute the cell size across processors. The cell size is taken
    # as max(h)*radius_scale
    cdef _compute_cell_size_for_binning(self)


# Cell to hold particle indices
cdef class Cell:
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef cIntPoint _cid                 # Spatial index for the cell
    cdef public bint is_boundary        # Flag to indicate boundary cells
    cdef int narrays                    # Number of particle arrays
    cdef public list lindices           # Local indices for particles
    cdef public list gindices           # Global indices for binned particles
    cdef list nparticles                # Number of particles in the cell
    cdef double cell_size               # bin size
    cdef public cPoint centroid         # Centroid computed from indices
    cdef cPoint boxmin                  # Bounding box min for the cell
    cdef cPoint boxmax                  # Bounding box max for the cell
    cdef int layers                     # Layers to compute bounding box
    cdef IntArray nbrprocs              # List of neighboring processors
    cdef public int size                # total number of particles in this cell

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


cdef class NeighborCache:

    cdef int _dst_index
    cdef int _src_index
    cdef int _n_threads
    cdef NNPS _nnps
    cdef UIntArray _pid_to_tid
    cdef UIntArray _start_stop
    cdef IntArray _cached
    cdef void **_neighbors
    cdef list _neighbor_arrays
    cdef int _narrays
    cdef list _particles
    cdef int _last_avg_nbr_size

    cdef void get_neighbors_raw(self, size_t d_idx, UIntArray nbrs) nogil
    cpdef get_neighbors(self, int src_index, size_t d_idx, UIntArray nbrs)
    cpdef find_all_neighbors(self)
    cpdef update(self)

    cdef void _update_last_avg_nbr_size(self)
    cdef void _find_neighbors(self, long d_idx) nogil

# Nearest neighbor locator
cdef class NNPS:
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef public list particles        # list of particle arrays
    cdef public list pa_wrappers      # list of particle array wrappers
    cdef public int narrays           # Number of particle arrays
    cdef public bint use_cache        # Use cache or not.
    cdef list cache                   # The neighbor cache.
    cdef public NeighborCache current_cache  # The current cache
    cdef int src_index, dst_index     # The current source and dest indices

    cdef public DomainManager domain  # Domain manager
    cdef public bint is_periodic      # flag for periodicity

    cdef public int dim               # Dimensionality of the problem
    cdef public DoubleArray xmin      # co-ordinate min values
    cdef public DoubleArray xmax      # co-ordinate max values
    cdef public double cell_size      # Cell size for binning
    cdef public double radius_scale   # Radius scale for kernel
    cdef IntArray cell_shifts         # cell shifts
    cdef public int n_cells           # number of cells
    cdef public bint sort_gids        # Sort neighbors by their gids.

    ##########################################################################
    # Member functions
    ##########################################################################
    # Main binning routine for NNPS for local particles. This clears
    # the current cell data, re-computes the cell size and bins all
    # particles locally.
    cpdef update(self)

    # Index particles given by a list of indices. The indices are
    # assumed to be of type unsigned int and local to the NNPS object
    cpdef _bin(self, int pa_index, UIntArray indices)

    cdef void _sort_neighbors(self, unsigned int* nbrs, size_t length,
                              unsigned int *gids) nogil

    # compute the min and max for the particle coordinates
    cdef _compute_bounds(self)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cdef void get_nearest_neighbors(self, size_t d_idx, 
                                      UIntArray nbrs) nogil

    # Neighbor query function. Returns the list of neighbors for a
    # requested particle. The returned list is assumed to be of type
    # unsigned int to follow the type of the local and global ids.
    # This method will never use the cached values.  If prealloc is set
    # to True it will assume that the neighbor array has enough space for
    # all the new neighbors and directly set the values in the array.
    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
                            size_t d_idx, UIntArray nbrs, bint prealloc)

    # Neighbor query function. Returns the list of neighbors for a
    # requested particle. The returned list is assumed to be of type
    # unsigned int to follow the type of the local and global ids.
    cpdef get_nearest_particles(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs)

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices)

    # Testing function for brute force neighbor search. The return
    # list is of the same type of the local and global ids (uint)
    cpdef brute_force_neighbors(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs)

    cpdef set_context(self, int src_index, int dst_index)

    cpdef spatially_order_particles(self, int pa_index)

    # refresh any data structures needed for binning
    cpdef _refresh(self)


# NNPS using the original gridding algorithm
cdef class DictBoxSortNNPS(NNPS):
    cdef public dict cells               # lookup table for the cells
    cdef list _cell_keys


# NNPS using the linked list approach
cdef class LinkedListNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef public IntArray ncells_per_dim  # number of cells in each direction
    cdef public int ncells_tot           # total number of cells
    cdef public bint fixed_h             # Constant cell sizes
    cdef public list heads               # Head arrays for the cells
    cdef public list nexts               # Next arrays for the particles

    cdef NNPSParticleArrayWrapper src, dst # Current source and destination.
    cdef UIntArray next, head              # Current next and head arrays.

    cpdef long _count_occupied_cells(self, long n_cells) except -1
    cpdef long _get_number_of_cells(self) except -1
    cdef long _get_flattened_cell_index(self, cPoint pnt, double cell_size)
    cdef long _get_valid_cell_index(self, int cid_x, int cid_y, int cid_z,
            int* ncells_per_dim, int dim, int n_cells) nogil
    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil


# NNPS using the linked list approach
cdef class BoxSortNNPS(LinkedListNNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef public map[long, int] cell_to_index  # Maps cell ID to an index

