#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

# Cython for compiler directives
cimport cython

#############################################################################
cdef class SpatialHashNNPS(NNPS):

    """Nearest neighbor particle search using Spatial Hashing algorithm

    Uses a hashtable to store particles according to cell it belongs to.

    Ref. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.105.6732&rep=rep1&type=pdf
    """

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None,
            bint fixed_h = False, bint cache = False,
            bint sort_gids = False, long long int table_size = 131072):
        #Initialize base class
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None,
            bint fixed_h = False, bint cache = False,
            bint sort_gids = False, long long int table_size = 131072):

        cdef int narrays = len(particles)

        self.table_size = table_size
        self.radius_scale2 = radius_scale*radius_scale

        self.hashtable = <HashTable**> malloc(narrays*sizeof(HashTable*))

        cdef int i
        for i from 0<=i<narrays:
            self.hashtable[i] = new HashTable(table_size)

        self.current_hash = NULL

    def __dealloc__(self):
        cdef int i
        for i from 0<=i<self.narrays:
            del self.hashtable[i]
        free(self.hashtable)


    #### Public protocol ################################################

    cpdef set_context(self, int src_index, int dst_index):
        """Set context for nearest neighbor searches.

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        """
        NNPS.set_context(self, src_index, dst_index)
        self.current_hash = self.hashtable[src_index]

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil:
        """Low level, high-performance non-gil method to find neighbors.
        This requires that `set_context()` be called beforehand.  This method
        does not reset the neighbors array before it appends the
        neighbors to it.

        """
        cdef double* dst_x_ptr = self.dst.x.data
        cdef double* dst_y_ptr = self.dst.y.data
        cdef double* dst_z_ptr = self.dst.z.data
        cdef double* dst_h_ptr = self.dst.h.data

        cdef double* src_x_ptr = self.src.x.data
        cdef double* src_y_ptr = self.src.y.data
        cdef double* src_z_ptr = self.src.z.data
        cdef double* src_h_ptr = self.src.h.data

        cdef double x = dst_x_ptr[d_idx]
        cdef double y = dst_y_ptr[d_idx]
        cdef double z = dst_z_ptr[d_idx]
        cdef double h = dst_h_ptr[d_idx]

        cdef unsigned int* s_gid = self.src.gid.data
        cdef int orig_length = nbrs.length

        cdef int c_x, c_y, c_z
        cdef double* xmin = self.xmin.data
        cdef unsigned int i, j, k
        cdef vector[unsigned int] *candidates
        find_cell_id_raw(
                x - xmin[0],
                y - xmin[1],
                z - xmin[2],
                self.cell_size,
                &c_x, &c_y, &c_z
                )
        cdef int candidate_size = 0

        cdef int x_boxes[27]
        cdef int y_boxes[27]
        cdef int z_boxes[27]
        cdef int num_boxes = self._neighbor_boxes(c_x, c_y, c_z,
                x_boxes, y_boxes, z_boxes)

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2*h*h
        cdef double hj2 = 0

        for i from 0<=i<num_boxes:
            candidates = self.current_hash.get(x_boxes[i], y_boxes[i], z_boxes[i])
            if candidates == NULL:
                continue
            candidate_size = candidates.size()
            for j from 0<=j<candidate_size:
                k = (candidates[0])[j]
                hj2 = self.radius_scale2*src_h_ptr[k]*src_h_ptr[k]
                xij2 = norm2(
                        src_x_ptr[k] - x,
                        src_y_ptr[k] - y,
                        src_z_ptr[k] - z
                        )
                if (xij2 < hi2) or (xij2 < hj2):
                    nbrs.c_append(k)

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Find nearest neighbors for particle id 'd_idx' without cache

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        d_idx: size_t
            Index of the query point in the destination particle array

        nbrs: UIntArray
            Array to be populated by nearest neighbors of 'd_idx'

        """
        self.set_context(src_index, dst_index)

        if prealloc:
            nbrs.length = 0
        else:
            nbrs.c_reset()

        self.find_nearest_neighbors(d_idx, nbrs)


    #### Private protocol ################################################

    cdef inline void _add_to_hashtable(self, int hash_id, unsigned int pid,
            int i, int j, int k) nogil:
        self.hashtable[hash_id].add(i,j,k,pid)

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z) nogil:
        cdef int length = 0
        cdef int p, q, r
        for p from -1<=p<2:
            for q from -1<=q<2:
                for r from -1<=r<2:
                    if i+p>=0 and j+q>=0 and k+r>=0:
                        x[length] = i+p
                        y[length] = j+q
                        z[length] = k+r
                        length += 1
        return length

    cpdef _refresh(self):
        cdef int i
        for i from 0<=i<self.narrays:
            del self.hashtable[i]
            self.hashtable[i] = new HashTable(self.table_size)
        self.current_hash = self.hashtable[self.src_index]

    cpdef _bin(self, int pa_index, UIntArray indices):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        cdef double* src_x_ptr = pa_wrapper.x.data
        cdef double* src_y_ptr = pa_wrapper.y.data
        cdef double* src_z_ptr = pa_wrapper.z.data

        cdef int num_indices = indices.length

        cdef double* xmin = self.xmin.data
        cdef int c_x, c_y, c_z
        cdef unsigned int i
        cdef unsigned int idx

        for i from 0<=i<num_indices:
            idx = indices.data[i]
            find_cell_id_raw(
                    src_x_ptr[idx] - xmin[0],
                    src_y_ptr[idx] - xmin[1],
                    src_z_ptr[idx] - xmin[2],
                    self.cell_size,
                    &c_x, &c_y, &c_z
                    )
            self._add_to_hashtable(pa_index, idx, c_x, c_y, c_z)


#############################################################################
cdef class ExtendedSpatialHashNNPS(NNPS):

    """Finds nearest neighbors using Extended Spatial Hashing algorithm

    Sub-divides each cell into smaller ones. Useful when particles cluster
    in a cell.

    For approximate Extended Spatial Hash, if the distance between a cell and
    the cell of the query particle is greater than search radius, the entire cell
    is ignored.

    Ref. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.105.6732&rep=rep1&type=pdf
    """

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int H = 3, int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False,
            long long int table_size = 131072, bint approximate = False):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.H = H
        self.approximate = approximate

        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int H = 3, int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False,
            long long int table_size = 131072, bint approximate = False):

        cdef int narrays = len(particles)

        self.table_size = table_size
        self.radius_scale2 = radius_scale*radius_scale

        self.hashtable = <HashTable**> malloc(narrays*sizeof(HashTable*))

        cdef int i
        for i from 0<=i<narrays:
            self.hashtable[i] = new HashTable(table_size)

        self.current_hash = NULL

    def __dealloc__(self):
        cdef int i
        for i from 0<=i<self.narrays:
            del self.hashtable[i]
        free(self.hashtable)


    #### Public protocol ################################################

    cpdef set_context(self, int src_index, int dst_index):
        """Set context for nearest neighbor searches.

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        """
        NNPS.set_context(self, src_index, dst_index)
        self.current_hash = self.hashtable[src_index]

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil:
        """Low level, high-performance non-gil method to find neighbors.
        This requires that `set_context()` be called beforehand.  This method
        does not reset the neighbors array before it appends the
        neighbors to it.

        """
        cdef double* dst_x_ptr = self.dst.x.data
        cdef double* dst_y_ptr = self.dst.y.data
        cdef double* dst_z_ptr = self.dst.z.data
        cdef double* dst_h_ptr = self.dst.h.data

        cdef double* src_x_ptr = self.src.x.data
        cdef double* src_y_ptr = self.src.y.data
        cdef double* src_z_ptr = self.src.z.data
        cdef double* src_h_ptr = self.src.h.data

        cdef double x = dst_x_ptr[d_idx]
        cdef double y = dst_y_ptr[d_idx]
        cdef double z = dst_z_ptr[d_idx]
        cdef double h = dst_h_ptr[d_idx]

        cdef unsigned int* s_gid = self.src.gid.data
        cdef int orig_length = nbrs.length

        cdef int c_x, c_y, c_z
        cdef double* xmin = self.xmin.data
        cdef unsigned int i, j, k
        cdef vector[unsigned int] *candidates

        find_cell_id_raw(
                x - xmin[0],
                y - xmin[1],
                z - xmin[2],
                self.h_sub,
                &c_x, &c_y, &c_z
                )

        cdef int candidate_size = 0
        cdef int mask_len = (2*self.H+1)*(2*self.H+1)*(2*self.H+1)

        cdef int* x_boxes = <int*> malloc(mask_len*sizeof(int))
        cdef int* y_boxes = <int*> malloc(mask_len*sizeof(int))
        cdef int* z_boxes = <int*> malloc(mask_len*sizeof(int))

        cdef int num_boxes = self._neighbor_boxes(c_x, c_y, c_z,
                x_boxes, y_boxes, z_boxes)

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2*h*h
        cdef double hj2 = 0

        for i from 0<=i<num_boxes:
            candidates = self.current_hash.get(x_boxes[i], y_boxes[i], z_boxes[i])
            if candidates == NULL:
                continue
            candidate_size = candidates.size()
            for j from 0<=j<candidate_size:
                k = (candidates[0])[j]
                hj2 = self.radius_scale2*src_h_ptr[k]*src_h_ptr[k]
                xij2 = norm2(
                        src_x_ptr[k] - x,
                        src_y_ptr[k] - y,
                        src_z_ptr[k] - z
                        )
                if (xij2 < hi2) or (xij2 < hj2):
                    nbrs.c_append(k)

        free(x_boxes)
        free(y_boxes)
        free(z_boxes)

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Find nearest neighbors for particle id 'd_idx' without cache

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        d_idx: size_t
            Index of the query point in the destination particle array

        nbrs: UIntArray
            Array to be populated by nearest neighbors of 'd_idx'

        """
        self.set_context(src_index, dst_index)

        if prealloc:
            nbrs.length = 0
        else:
            nbrs.c_reset()

        self.find_nearest_neighbors(d_idx, nbrs)


    #### Private protocol ################################################

    cdef inline void _add_to_hashtable(self, int hash_id, unsigned int pid,
            int i, int j, int k) nogil:
        self.hashtable[hash_id].add(i,j,k,pid)

    cdef inline int _h_mask_approx(self, int* x, int* y, int* z) nogil:
        cdef int length = 0
        cdef int s, t, u

        for s from -self.H<=s<=self.H:
            for t from -self.H<=t<=self.H:
                for u from -self.H<=u<=self.H:
                    if norm2(self.h_sub*s, self.h_sub*t, self.h_sub*u) \
                        <= self.cell_size*self.cell_size:
                            x[length] = s
                            y[length] = t
                            z[length] = u
                            length += 1

        return length

    cdef inline int _h_mask_exact(self, int* x, int* y, int* z) nogil:
        cdef int length = 0
        cdef int s, t, u

        for s from -self.H<=s<=self.H:
            for t from -self.H<=t<=self.H:
                for u from -self.H<=u<=self.H:
                    x[length] = s
                    y[length] = t
                    z[length] = u
                    length += 1

        return length

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z) nogil:
        cdef int length = 0
        cdef int p

        cdef int mask_len = (2*self.H+1)*(2*self.H+1)*(2*self.H+1)

        cdef int* x_mask = <int*> malloc(mask_len*sizeof(int))
        cdef int* y_mask = <int*> malloc(mask_len*sizeof(int))
        cdef int* z_mask = <int*> malloc(mask_len*sizeof(int))

        if self.approximate:
            mask_len = self._h_mask_approx(x_mask, y_mask, z_mask)
        else:
            mask_len = self._h_mask_exact(x_mask, y_mask, z_mask)

        for p from 0<=p<mask_len:
            if (i + x_mask[p] >= 0 and
                j + y_mask[p] >= 0 and
                k + z_mask[p] >= 0):
                    x[length] = i + x_mask[p]
                    y[length] = j + y_mask[p]
                    z[length] = k + z_mask[p]
                    length += 1

        free(x_mask)
        free(y_mask)
        free(z_mask)

        return length

    cpdef _refresh(self):
        cdef int i
        for i from 0<=i<self.narrays:
            del self.hashtable[i]
            self.hashtable[i] = new HashTable(self.table_size)
        self.current_hash = self.hashtable[self.src_index]

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        cdef double* src_x_ptr = pa_wrapper.x.data
        cdef double* src_y_ptr = pa_wrapper.y.data
        cdef double* src_z_ptr = pa_wrapper.z.data

        cdef int num_indices = indices.length

        cdef double* xmin = self.xmin.data
        cdef int c_x, c_y, c_z
        cdef unsigned int i
        cdef unsigned int idx

        self.h_sub = self.cell_size/self.H

        for i from 0<=i<num_indices:
            idx = indices.data[i]
            find_cell_id_raw(
                    src_x_ptr[idx] - xmin[0],
                    src_y_ptr[idx] - xmin[1],
                    src_z_ptr[idx] - xmin[2],
                    self.h_sub,
                    &c_x, &c_y, &c_z
                    )
            self._add_to_hashtable(pa_index, idx, c_x, c_y, c_z)

