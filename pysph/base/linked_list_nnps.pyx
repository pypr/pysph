#cython: embedsignature=True
# Library imports.
import numpy as np
cimport numpy as np

# Cython imports
from cython.operator cimport dereference as deref, preincrement as inc
from cython.parallel import parallel, prange, threadid

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.vector cimport vector

# cpython
from cpython.dict cimport PyDict_Clear, PyDict_Contains, PyDict_GetItem
from cpython.list cimport PyList_GetItem, PyList_SetItem, PyList_GET_ITEM

# Cython for compiler directives
cimport cython


#############################################################################
cdef class LinkedListNNPS(NNPS):
    """Nearest neighbor query class using the linked list method.
    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None,
                 bint fixed_h=False, bint cache=False, bint sort_gids=False):
        """Constructor for NNPS

        Parameters
        ----------

        dim : int
            Number of dimension.

        particles : list
            The list of particle arrays we are working on

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        ghost_layers : int
            Optional number of layers to share in parallel

        domain : DomainManager, default (None)
            Optional limits for the domain

        fixed_h : bint
            Optional flag to use constant cell sizes throughout.

        cache : bint
            Flag to set if we want to cache neighbor calls. This costs
            storage but speeds up neighbor calculations.

        sort_gids : bint, default (False)
            Flag to sort neighbors using gids (if they are available).
            This is useful when comparing parallel results with those
            from a serial run.
        """
        # initialize the base class
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        # initialize the head and next for each particle array
        self.heads = [UIntArray() for i in range(self.narrays)]
        self.nexts = [UIntArray() for i in range(self.narrays)]

        # flag for constant smoothing lengths
        self.fixed_h = fixed_h

        # defaults
        self.ncells_per_dim = IntArray(3)
        self.n_cells = 0
        self.sort_gids = sort_gids

        # compute the intial box sort for all local particles. The
        # DomainManager.setup_domain method is called to compute the
        # cell size.
        self.domain.update()
        self.update()

    #### Public protocol ################################################

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil:
        """Low level, high-performance non-gil method to find neighbors.
        This requires that `set_context()` be called beforehand.  This method
        does not reset the neighbors array before it appends the
        neighbors to it.

        """
        # Number of cells
        cdef int n_cells = self.n_cells
        cdef int dim = self.dim

        # cell shifts
        cdef int* shifts = self.cell_shifts.data

        # Source data arrays
        cdef double* s_x = self.src.x.data
        cdef double* s_y = self.src.y.data
        cdef double* s_z = self.src.z.data
        cdef double* s_h = self.src.h.data
        cdef unsigned int* s_gid = self.src.gid.data

        # Destination particle arrays
        cdef double* d_x = self.dst.x.data
        cdef double* d_y = self.dst.y.data
        cdef double* d_z = self.dst.z.data
        cdef double* d_h = self.dst.h.data
        cdef unsigned int* d_gid = self.dst.gid.data

        cdef unsigned int* head = self.head.data
        cdef unsigned int* next = self.next.data

        # minimum values for the particle distribution
        cdef double* xmin = self.xmin.data

        # cell size and radius
        cdef double radius_scale = self.radius_scale
        cdef double cell_size = self.cell_size

        # locals
        cdef size_t indexj
        cdef double xij2
        cdef double hi2, hj2
        cdef int ierr, nnbrs
        cdef unsigned int _next
        cdef int ix, iy, iz

        # this is the physical position of the particle that will be
        # used in pairwise searching
        cdef double x = d_x[d_idx]
        cdef double y = d_y[d_idx]
        cdef double z = d_z[d_idx]

        # get the un-flattened index for the destination particle with
        # respect to the minimum
        cdef int _cid_x, _cid_y, _cid_z
        find_cell_id_raw(
            x - xmin[0], y - xmin[1], z - xmin[2],
            cell_size, &_cid_x, &_cid_y, &_cid_z
        )

        cdef int cid_x, cid_y, cid_z
        cdef long cell_index, orig_length
        cid_x = cid_y = cid_z = 0

        # gather search radius
        hi2 = radius_scale * d_h[d_idx]
        hi2 *= hi2

        orig_length = nbrs.length

        # Begin search through neighboring cells
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    cid_x = _cid_x + shifts[ix]
                    cid_y = _cid_y + shifts[iy]
                    cid_z = _cid_z + shifts[iz]

                    # Only consider valid cell indices
                    cell_index = self._get_valid_cell_index(
                        cid_x, cid_y, cid_z,
                        self.ncells_per_dim.data, dim, n_cells
                    )
                    if cell_index > -1:

                        # get the first particle and begin iteration
                        _next = head[ cell_index ]
                        while( _next != UINT_MAX ):
                            hj2 = radius_scale * s_h[_next]
                            hj2 *= hj2

                            xij2 = norm2( s_x[_next]-x,
                                          s_y[_next]-y,
                                          s_z[_next]-z )

                            # select neighbor
                            if ( (xij2 < hi2) or (xij2 < hj2) ):
                                nbrs.c_append(_next)

                            # get the 'next' particle in this cell
                            _next = next[_next]
        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
                            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Utility function to get near-neighbors for a particle.

        Parameters
        ----------

        src_index : int
            Index of the source particle array in the particles list

        dst_index : int
            Index of the destination particle array in the particles list

        d_idx : int (input)
            Destination particle for which neighbors are sought.

        nbrs : UIntArray (output)
            Neighbors for the requested particle are stored here.

        prealloc : bool
            Specifies if the neighbor array already has pre-allocated space
            for the neighbor list.  In this case the neighbors are directly set
            in the given array without resetting or appending to the array.
            This improves performance when the neighbors are cached.
        """
        self.set_context(src_index, dst_index)

        # reset the length of the nbr array
        if prealloc:
            nbrs.length = 0
        else:
            nbrs.reset()

        self.find_nearest_neighbors(d_idx, nbrs)

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        cdef UIntArray head = self.heads[pa_index]
        cdef UIntArray next = self.nexts[pa_index]
        cdef ZOLTAN_ID_TYPE _next
        indices.reset()
        cdef long i

        for i in range(self.n_cells):
            _next = head.data[i]
            while (_next != UINT_MAX):
                indices.append(<long>_next)
                _next = next.data[_next]

    cpdef set_context(self, int src_index, int dst_index):
        """Setup the context before asking for neighbors.  The `dst_index`
        represents the particles for whom the neighbors are to be determined
        from the particle array with index `src_index`.

        Parameters
        ----------

         src_index: int: the source index of the particle array.
         dst_index: int: the destination index of the particle array.
        """
        NNPS.set_context(self, src_index, dst_index)

        # Set the current context.
        self.src = self.pa_wrappers[ src_index ]
        self.dst = self.pa_wrappers[ dst_index ]

        # next and head linked lists
        self.next = self.nexts[ src_index ]
        self.head = self.heads[ src_index ]


    #### Private protocol ################################################

    cpdef _bin(self, int pa_index, UIntArray indices):
        """Bin a given particle array with indices.

        Parameters
        ----------

        pa_index : int
            Index of the particle array corresponding to the particles list

        indices : UIntArray
            Subset of particles to bin

        """
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[ pa_index ]
        cdef DoubleArray x = pa_wrapper.x
        cdef DoubleArray y = pa_wrapper.y
        cdef DoubleArray z = pa_wrapper.z

        cdef DoubleArray xmin = self.xmin
        cdef DoubleArray xmax = self.xmax
        cdef IntArray ncells_per_dim = self.ncells_per_dim
        cdef int dim = self.dim

        # the head and next arrays for this particle array
        cdef UIntArray head = self.heads[ pa_index ]
        cdef UIntArray next = self.nexts[ pa_index ]
        cdef double cell_size = self.cell_size

        cdef UIntArray lindices, gindices
        cdef size_t num_particles, indexi, i

        # point and flattened index
        cdef cPoint pnt = cPoint_new(0, 0, 0)
        cdef int _cid

        # now bin the particles
        num_particles = indices.length
        for indexi in range(num_particles):
            i = indices.data[indexi]

            # the flattened index is considered relative to the
            # minimum along each co-ordinate direction
            pnt.x = x.data[i] - xmin.data[0]
            pnt.y = y.data[i] - xmin.data[1]
            pnt.z = z.data[i] - xmin.data[2]

            # flattened cell index
            _cid = self._get_flattened_cell_index(pnt, cell_size)

            # insert this particle
            next.data[ i ] = head.data[ _cid ]
            head.data[_cid] = i

    cdef long _get_flattened_cell_index(self, cPoint pnt, double cell_size):
        return flatten(
            find_cell_id(pnt, cell_size), self.ncells_per_dim, self.dim
        )

    cpdef long _get_number_of_cells(self) except -1:
        cdef double cell_size = self.cell_size
        cdef double cell_size1 = 1./cell_size
        cdef DoubleArray xmin = self.xmin
        cdef DoubleArray xmax = self.xmax
        cdef int ncx, ncy, ncz
        cdef long _ncells
        cdef int dim = self.dim

        # calculate the number of cells.
        ncx = <int>ceil( cell_size1*(xmax.data[0] - xmin.data[0]) )
        ncy = <int>ceil( cell_size1*(xmax.data[1] - xmin.data[1]) )
        ncz = <int>ceil( cell_size1*(xmax.data[2] - xmin.data[2]) )

        if ncx < 0 or ncy < 0 or ncz < 0:
            msg = 'LinkedListNNPS: Number of cells is negative '\
                   '(%s, %s, %s).'%(ncx, ncy, ncz)
            raise RuntimeError(msg)
            return -1

        # number of cells along each coordinate direction
        self.ncells_per_dim.data[0] = ncx
        self.ncells_per_dim.data[1] = ncy
        self.ncells_per_dim.data[2] = ncz

        # total number of cells
        _ncells = ncx
        if dim == 2: _ncells = ncx * ncy
        if dim == 3: _ncells = ncx * ncy * ncz
        return _ncells

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long _get_valid_cell_index(self, int cid_x, int cid_y, int cid_z,
            int* ncells_per_dim, int dim, int n_cells) nogil:
        return get_valid_cell_index(
            cid_x, cid_y, cid_z, ncells_per_dim, dim, n_cells
        )

    cpdef long _count_occupied_cells(self, long n_cells) except -1:
        if n_cells < 0 or n_cells > 2**28:
            # unsigned ints are 4 bytes, which means 2**28 cells requires 1GB.
            msg = "ERROR: LinkedListNNPS requires too many cells (%s)."%n_cells
            raise RuntimeError(msg)
            return -1

        return n_cells

    cpdef _refresh(self):
        """Refresh the head and next arrays locally"""
        cdef DomainManager domain = self.domain
        cdef int narrays = self.narrays
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef list heads = self.heads
        cdef list nexts = self.nexts

        # locals
        cdef int i, j, np
        cdef long _ncells
        cdef UIntArray head, next

        _ncells = self._get_number_of_cells()

        _ncells = self._count_occupied_cells(_ncells)

        self.n_cells = <int>_ncells

        # initialize the head and next arrays
        for i in range(narrays):
            pa_wrapper = pa_wrappers[i]
            np = pa_wrapper.get_number_of_particles()

            # re-size the head and next arrays
            head = <UIntArray>PyList_GetItem(heads, i)
            next = <UIntArray>PyList_GetItem(nexts, i )

            head.resize( _ncells )
            next.resize( np )

            # UINT_MAX is used to indicate an invalid index
            for j in range(_ncells):
                head.data[j] = UINT_MAX

            for j in range(np):
                next.data[j] = UINT_MAX


