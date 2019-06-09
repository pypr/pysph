# numpy
cimport numpy as np
cimport cython

from libcpp.map cimport map
from libcpp.vector cimport vector

# PyZoltan CArrays
from cyarray.carray cimport UIntArray, IntArray, DoubleArray, LongArray

# local imports
from particle_array cimport ParticleArray
from point cimport *

cdef extern from 'math.h':
    int abs(int) nogil
    double ceil(double) nogil
    double floor(double) nogil
    double fabs(double) nogil
    double fmax(double, double) nogil
    double fmin(double, double) nogil

cdef extern from 'limits.h':
    cdef unsigned int UINT_MAX
    cdef int INT_MAX

# ZOLTAN ID TYPE AND PTR
ctypedef unsigned int ZOLTAN_ID_TYPE
ctypedef unsigned int* ZOLTAN_ID_PTR

cdef inline double norm2(double x, double y, double z) nogil:
    return x*x + y*y + z*z

@cython.cdivision(True)
cdef inline int real_to_int(double real_val, double step) nogil:
    """ Return the bin index to which the given position belongs.

    Parameters
    ----------
    val -- The coordinate location to bin
    step -- the bin size

    Examples
    --------
    >>> real_to_int(1.5, 1.0)
    1
    >>> real_to_int(-0.5, 1.0)
    -1
    """
    cdef int ret_val = <int>floor( real_val/step )

    return ret_val

cdef inline void find_cell_id_raw(double x, double y, double z, double
                           cell_size, int *ix, int *iy, int *iz) nogil:
    """ Find the cell index for the corresponding point

    Parameters
    ----------
    x, y, z: double
        the point for which the index is sought
    cell_size : double
        the cell size to use
    ix, iy, iz : int*
        output parameter holding the cell index

    Notes
    ------
    Performs a box sort based on the point and cell size

    Uses the function  `real_to_int`

    """
    ix[0] = real_to_int(x, cell_size)
    iy[0] = real_to_int(y, cell_size)
    iz[0] = real_to_int(z, cell_size)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline long flatten_raw(int x, int y, int z, int* ncells_per_dim,
        int dim) nogil:
    """Return a flattened index for a cell

    The flattening is determined using the row-order indexing commonly
    employed in SPH. This would need to be changed for hash functions
    based on alternate orderings.

    """
    cdef long ncx = ncells_per_dim[0]
    cdef long ncy = ncells_per_dim[1]

    return <long>( x + ncx * y + ncx*ncy * z )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline long flatten(cIntPoint cid, IntArray ncells_per_dim, int dim) nogil:
    """Return a flattened index for a cell

    The flattening is determined using the row-order indexing commonly
    employed in SPH. This would need to be changed for hash functions
    based on alternate orderings.

    """
    return flatten_raw(cid.x, cid.y, cid.z, ncells_per_dim.data, dim)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline long get_valid_cell_index(int cid_x, int cid_y, int cid_z,
        int* ncells_per_dim, int dim, int n_cells) nogil:
    """Return the flattened index for a valid cell"""
    cdef long ncx = ncells_per_dim[0]
    cdef long ncy = ncells_per_dim[1]
    cdef long ncz = ncells_per_dim[2]

    cdef long cell_index = -1

    # basic test for valid indices. Since we bin the particles with
    # respect to the origin, negative indices can never occur.
    cdef bint is_valid = (
        (ncx > cid_x > -1) and (ncy > cid_y > -1) and (ncz > cid_z > -1)
    )

    # Given the validity of the cells, return the flattened cell index
    if is_valid:
        cell_index = flatten_raw(cid_x, cid_y, cid_z, ncells_per_dim, dim)

        if not (-1 < cell_index < n_cells):
            cell_index = -1

    return cell_index

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


cdef class DomainManager:
    cdef public object backend
    cdef public object manager


cdef class DomainManagerBase:
    cdef public double xmin, xmax
    cdef public double ymin, ymax
    cdef public double zmin, zmax

    cdef public double xtranslate
    cdef public double ytranslate
    cdef public double ztranslate

    cdef public int dim
    cdef public bint periodic_in_x, periodic_in_y, periodic_in_z
    cdef public bint is_periodic
    cdef public bint mirror_in_x, mirror_in_y, mirror_in_z
    cdef public bint is_mirror

    cdef public object props
    cdef public list copy_props
    cdef public list pa_wrappers     # NNPS particle array wrappers
    cdef public int narrays          # number of arrays
    cdef public double cell_size     # distance to create ghosts
    cdef public double hmin          # minimum h
    cdef public bint in_parallel     # Flag to determine if in parallel
    cdef public double radius_scale  # Radius scale for kernel
    cdef public double n_layers      # Number of layers of ghost particles

    #cdef double dbl_max              # Maximum value of double

    # remove ghost particles from a previous iteration
    cpdef _remove_ghosts(self)


# Domain limits for the simulation
cdef class CPUDomainManager(DomainManagerBase):
    cdef public bint use_double
    cdef public object dtype
    cdef public double dtype_max
    cdef public list ghosts

    # box-wrap particles within the physical domain
    cdef _box_wrap_periodic(self)

    # Convenience function to add a value to a carray
    cdef _add_to_array(self, DoubleArray arr, double disp, int start=*)

    # Convenience function to multiply a value to a carray
    cdef _mul_to_array(self, DoubleArray arr, double val)

    # Convenience function to add a carray to a carray elementwise
    cdef _add_array_to_array(self, DoubleArray arr, DoubleArray translate)

    # Convenience function to add a value to a carray
    cdef _change_velocity(self, DoubleArray arr, double disp)

    # create new periodic ghosts
    cdef _create_ghosts_periodic(self)

    # create new mirror ghosts
    cdef _create_ghosts_mirror(self)

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
    cdef int _narrays
    cdef list _particles

    cdef int _n_threads
    cdef NNPS _nnps
    cdef UIntArray _pid_to_tid
    cdef UIntArray _start_stop
    cdef IntArray _cached
    cdef void **_neighbors
    # This is made public purely for testing!
    cdef public list _neighbor_arrays
    cdef int _last_avg_nbr_size

    cdef void get_neighbors_raw(self, size_t d_idx, UIntArray nbrs) nogil
    cpdef get_neighbors(self, int src_index, size_t d_idx, UIntArray nbrs)
    cpdef find_all_neighbors(self)
    cpdef update(self)

    cdef void _update_last_avg_nbr_size(self)
    cdef void _find_neighbors(self, long d_idx) nogil

cdef class NNPSBase:
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef public list particles        # list of particle arrays
    cdef public list pa_wrappers      # list of particle array wrappers
    cdef public int narrays           # Number of particle arrays
    cdef bint use_cache               # Use cache or not.
    cdef public list cache            # The neighbor cache.
    cdef int src_index, dst_index     # The current source and dest indices

    cdef public DomainManager domain  # Domain manager
    cdef public bint is_periodic      # flag for periodicity

    cdef public int dim               # Dimensionality of the problem
    cdef public double cell_size      # Cell size for binning
    cdef public double hmin           # Minimum h
    cdef public double radius_scale   # Radius scale for kernel
    cdef IntArray cell_shifts         # cell shifts
    cdef public int n_cells           # number of cells

    # Testing function for brute force neighbor search. The return
    # list is of the same type of the local and global ids (uint)
    cpdef brute_force_neighbors(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs)

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cpdef get_nearest_particles(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs)
    cpdef set_context(self, int src_index, int dst_index)

# Nearest neighbor locator
cdef class NNPS(NNPSBase):
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef public DoubleArray xmin      # co-ordinate min values
    cdef public DoubleArray xmax      # co-ordinate max values
    cdef public NeighborCache current_cache  # The current cache
    cdef public double _last_domain_size # last size of domain.

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

    cpdef set_context(self, int src_index, int dst_index)

    cpdef spatially_order_particles(self, int pa_index)

    # refresh any data structures needed for binning
    cpdef _refresh(self)
