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
cdef class BoxSortNNPS(LinkedListNNPS):

    """Nearest neighbor query class using the box sort method but which
    uses the LinkedList algorithm.  This makes this very fast but
    perhaps not as safe as the DictBoxSortNNPS.  All this class does
    is to use a std::map to obtain a linear cell index from the actual
    flattened cell index.
    """

    #### Private protocol ################################################

    cdef long _get_flattened_cell_index(self, cPoint pnt, double cell_size):
        cdef long cell_id = flatten(
            find_cell_id(pnt, cell_size), self.ncells_per_dim, self.dim
        )
        return self.cell_to_index[cell_id]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline long _get_valid_cell_index(self, int cid_x, int cid_y, int cid_z,
            int* ncells_per_dim, int dim, int n_cells) nogil:
        """Return the flattened index for a valid cell"""
        cdef long ncy = ncells_per_dim[1]
        cdef long ncz = ncells_per_dim[2]

        cdef long cell_id
        cdef long cell_index = -1

        # basic test for valid indices. Since we bin the particles with
        # respect to the origin, negative indices can never occur.
        cdef bint is_valid = (cid_x > -1) and (cid_y > -1) and (cid_z > -1)

        # additional check for 1D. This is because we search in all 26
        # neighboring cells for neighbors. In 1D this can be problematic
        # since (ncy = ncz = 0) which means (ncy=1 or ncz=1) will also
        # result in a valid cell with a flattened index < ncells
        if dim == 1:
            if ( (cid_y > ncy) or (cid_z > ncz) ):
                is_valid = False

        # Given the validity of the cells, return the flattened cell index
        cdef map[long, int].iterator it
        if is_valid:
            cell_id = flatten_raw(cid_x, cid_y, cid_z, ncells_per_dim, dim)
            if cell_id > -1:
                it = self.cell_to_index.find(cell_id)
                if it != self.cell_to_index.end():
                    cell_index = deref(it).second

        return cell_index

    cpdef long _count_occupied_cells(self, long n_cells) except -1:
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int np
        cdef DoubleArray x, y, z
        cdef DoubleArray xmin = self.xmin
        cdef DoubleArray xmax = self.xmax
        cdef IntArray ncells_per_dim = self.ncells_per_dim
        cdef int narrays = self.narrays
        cdef int dim = self.dim
        cdef double cell_size = self.cell_size

        cdef cPoint pnt = cPoint_new(0, 0, 0)
        cdef long _cid
        cdef map[long, int] _cid_to_index
        cdef pair[long, int] _entry

        # flattened cell index
        cdef int i, j
        cdef long cell_index

        for j in range(narrays):
            pa_wrapper = pa_wrappers[j]
            np = pa_wrapper.get_number_of_particles()
            x = pa_wrapper.x
            y = pa_wrapper.y
            z = pa_wrapper.z

            for i in range(np):
                # the flattened index is considered relative to the
                # minimum along each co-ordinate direction
                pnt.x = x.data[i] - xmin.data[0]
                pnt.y = y.data[i] - xmin.data[1]
                pnt.z = z.data[i] - xmin.data[2]

                # flattened cell index
                _cid = flatten( find_cell_id( pnt, cell_size ), ncells_per_dim, dim )
                _entry.first = _cid
                _entry.second = 1
                _cid_to_index.insert(_entry)

        cdef map[long, int].iterator it = _cid_to_index.begin()
        cdef int count = 0
        while it != _cid_to_index.end():
            _entry = deref(it)
            _cid_to_index[_entry.first] = count
            count += 1
            inc(it)

        self.cell_to_index = _cid_to_index
        return count

##############################################################################
cdef class DictBoxSortNNPS(NNPS):
    """Nearest neighbor query class using the box-sort algorithm using a
    dictionary.

    NNPS bins all local particles using the box sort algorithm in
    Cells. The cells are stored in a dictionary 'cells' which is keyed
    on the spatial index (IntPoint) of the cell.

    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, cache=False,
                 sort_gids=False):
        """Constructor for NNPS

        Parameters
        ----------

        dim : int
            Number of dimensions.

        particles : list
            The list of particle arrays we are working on.

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        domain : DomainManager, default (None)
            Optional limits for the domain

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

        # initialize the cells dict
        self.cells = {}

        # compute the intial box sort. First, the Domain Manager's
        # update method is called to comptue the maximum smoothing
        # length for particle binning.
        self.domain.update()
        self.update()

        msg = """WARNING: The cache will currently work only if
        find_nearest_neighbors works and can be used without requiring
        the GIL.  The DictBoxSort does not work with OpenMP since it uses
        Python objects and cannot release the GIL.  Until this is fixed,
        we cannot use caching, or parallel neighbor finding using this
        DictBoxSortNNPS, use the more efficient LinkedListNNPS instead.
        Disabling caching for now.
        """
        if cache:
            print(msg)
        self.use_cache = False


    #### Public protocol ################################################

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
        cdef dict cells = self.cells
        cdef Cell cell

        cdef NNPSParticleArrayWrapper src = self.pa_wrappers[ src_index ]
        cdef NNPSParticleArrayWrapper dst = self.pa_wrappers[ dst_index ]

        # Source data arrays
        cdef DoubleArray s_x = src.x
        cdef DoubleArray s_y = src.y
        cdef DoubleArray s_z = src.z
        cdef DoubleArray s_h = src.h
        cdef UIntArray s_gid = src.gid

        # Destination particle arrays
        cdef DoubleArray d_x = dst.x
        cdef DoubleArray d_y = dst.y
        cdef DoubleArray d_z = dst.z
        cdef DoubleArray d_h = dst.h
        cdef UIntArray d_gid = dst.gid

        cdef double radius_scale = self.radius_scale
        cdef double cell_size = self.cell_size
        cdef UIntArray lindices
        cdef size_t indexj, count
        cdef ZOLTAN_ID_TYPE j

        cdef cPoint xi = cPoint_new(d_x.data[d_idx], d_y.data[d_idx], d_z.data[d_idx])
        cdef cIntPoint _cid = find_cell_id( xi, cell_size )
        cdef IntPoint cid = IntPoint_from_cIntPoint( _cid )
        cdef IntPoint cellid = IntPoint(0, 0, 0)

        cdef double xij2

        cdef double hi2, hj2

        hi2 = radius_scale * d_h.data[d_idx]
        hi2 *= hi2

        cdef int ierr

        # reset the nbr array length. This should avoid a realloc
        if prealloc:
            nbrs.length = 0
        else:
            nbrs.reset()
        count = 0

        cdef int ix, iy, iz
        for ix in [cid.data.x -1, cid.data.x, cid.data.x + 1]:
            for iy in [cid.data.y - 1, cid.data.y, cid.data.y + 1]:
                for iz in [cid.data.z - 1, cid.data.z, cid.data.z + 1]:
                    cellid.data.x = ix; cellid.data.y = iy; cellid.data.z = iz

                    ierr = PyDict_Contains( cells, cellid )
                    if ierr == 1:

                        cell = <Cell>PyDict_GetItem( cells, cellid )
                        lindices = <UIntArray>PyList_GetItem( cell.lindices, src_index )

                        for indexj in range( lindices.length ):
                            j = lindices.data[indexj]

                            xij2 = norm2( s_x.data[j]-xi.x,
                                          s_y.data[j]-xi.y,
                                          s_z.data[j]-xi.z )

                            hj2 = radius_scale * s_h.data[j]
                            hj2 *= hj2

                            # select neighbor
                            if ( (xij2 < hi2) or (xij2 < hj2) ):
                                if prealloc:
                                    nbrs.data[count] = j
                                    count += 1
                                else:
                                    nbrs.append( j )
        if prealloc:
            nbrs.length = count

        if self.sort_gids:
            self._sort_neighbors(nbrs.data, count, s_gid.data)

    #### Private protocol ################################################

    cpdef _refresh(self):
        """Clear the cells dict"""
        cdef dict cells = self.cells
        PyDict_Clear( cells )

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

        cdef dict cells = self.cells
        cdef double cell_size = self.cell_size

        cdef UIntArray lindices, gindices
        cdef size_t num_particles, indexi, i

        cdef cIntPoint _cid
        cdef IntPoint cid

        cdef Cell cell
        cdef int ierr, narrays = self.narrays

        # now begin binning the particles
        num_particles = indices.length
        for indexi in range(num_particles):
            i = indices.data[indexi]

            pnt = cPoint_new( x.data[i], y.data[i], z.data[i] )
            _cid = find_cell_id( pnt, cell_size )

            cid = IntPoint_from_cIntPoint(_cid)

            ierr = PyDict_Contains(cells, cid)
            if ierr == 0:
                cell = Cell(cid, cell_size, narrays)
                cells[ cid ] = cell

            # add this particle to the list of indicies
            cell = <Cell>PyDict_GetItem( cells, cid )
            lindices = <UIntArray>PyList_GetItem( cell.lindices, pa_index )

            lindices.append( <ZOLTAN_ID_TYPE> i )
            #gindices.append( gid.data[i] )

        self.n_cells = <int>len(cells)
        self._cell_keys = list(cells.keys())

