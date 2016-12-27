#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libcpp.vector cimport vector

from nnps_base cimport *

# Cython for compiler directives
cimport cython

DEF EPS = 1e-6

IF UNAME_SYSNAME == "Windows":
    cdef inline double fmin(double x, double y) nogil:
        return x if x < y else y
    cdef inline double fmax(double x, double y) nogil:
        return x if x > y else y

#############################################################################
cdef class StratifiedHashNNPS(NNPS):

    """Finds nearest neighbors using Spatial Hashing with particles classified according
    to their support radii.
    """

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int H = 1,
            int num_levels = 1, long long int table_size = 131072):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.table_size = table_size
        self.radius_scale2 = radius_scale*radius_scale
        self.interval_size = 0
        self.H = H

        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int H = 1,
            int num_levels = 1, long long int table_size = 131072):

        cdef int narrays = len(particles)
        cdef HashTable** current_hash

        if fixed_h:
            self.num_levels = 1
        else:
            self.num_levels = num_levels

        self.hashtable = <HashTable***> malloc(narrays*sizeof(HashTable**))
        self.cell_sizes = <double**> malloc(narrays*sizeof(double*))

        cdef int i, j
        for i from 0<=i<narrays:
            self.hashtable[i] = <HashTable**> malloc(self.num_levels*sizeof(HashTable*))
            self.cell_sizes[i] = <double*> malloc(self.num_levels*sizeof(double))
            current_hash = self.hashtable[i]
            for j from 0<=j<self.num_levels:
                current_hash[j] = NULL

        self.current_hash = NULL
        self.current_cells = NULL

    def __dealloc__(self):
        cdef HashTable** current_hash
        cdef int i, j
        for i from 0<=i<self.narrays:
            current_hash = self.hashtable[i]
            for j from 0<=j<self.num_levels:
                if current_hash[j] != NULL:
                    del current_hash[j]
            free(self.hashtable[i])
            free(self.cell_sizes[i])
        free(self.hashtable)
        free(self.cell_sizes)

    #### Public protocol ################################################

    cpdef int count_particles(self, int interval):
        """Count number of particles in at a level"""
        cdef int i
        cdef int num_particles = 0
        return self.current_hash[interval].number_of_particles()

    cpdef double get_binning_size(self, int interval):
        """Get bin size at a level"""
        return self._get_h_max(self.current_cells, interval)

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
        self.current_cells = self.cell_sizes[src_index]

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]

    @cython.cdivision(True)
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
        cdef unsigned int i, j, k, n
        cdef vector[unsigned int] *candidates
        cdef int candidate_size = 0

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2*h*h
        cdef double hj2 = 0

        cdef double h_max
        cdef int H, mask_len, num_boxes

        cdef int* x_boxes
        cdef int* y_boxes
        cdef int* z_boxes

        cdef HashTable* hash_level = NULL
        cdef HashEntry* candidate_cell = NULL

        for i from 0<=i<self.num_levels:

            h_max = fmax(self.radius_scale*h, self._get_h_max(self.current_cells, i))
            H = <int> ceil(h_max*self.H/self._get_h_max(self.current_cells, i))

            mask_len = (2*H+1)*(2*H+1)*(2*H+1)

            x_boxes = <int*> malloc(mask_len*sizeof(int))
            y_boxes = <int*> malloc(mask_len*sizeof(int))
            z_boxes = <int*> malloc(mask_len*sizeof(int))

            hash_level = self.current_hash[i]
            find_cell_id_raw(
                    x - xmin[0],
                    y - xmin[1],
                    z - xmin[2],
                    self._get_h_max(self.current_cells, i)/self.H,
                    &c_x, &c_y, &c_z
                    )

            num_boxes = self._neighbor_boxes(c_x, c_y, c_z,
                    x_boxes, y_boxes, z_boxes, H)

            for j from 0<=j<num_boxes:
                candidate_cell = hash_level.get(x_boxes[j], y_boxes[j], z_boxes[j])
                if candidate_cell == NULL:
                    continue
                candidates = candidate_cell.get_indices()
                candidate_size = candidates.size()
                for k from 0<=k<candidate_size:
                    n = (candidates[0])[k]
                    hj2 = self.radius_scale2*src_h_ptr[n]*src_h_ptr[n]
                    xij2 = norm2(
                            src_x_ptr[n] - x,
                            src_y_ptr[n] - y,
                            src_z_ptr[n] - z
                            )
                    if (xij2 < hi2) or (xij2 < hj2):
                        nbrs.c_append(n)

            free(x_boxes)
            free(y_boxes)
            free(z_boxes)

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )


    #### Private protocol ################################################

    @cython.cdivision(True)
    cdef inline int _get_hash_id(self, double h) nogil:
        return <int> floor((self.radius_scale*h - self.hmin)/self.interval_size)

    cdef inline double _get_h_max(self, double* current_cells, int hash_id) nogil:
        return self.radius_scale*current_cells[hash_id]

    @cython.cdivision(True)
    cdef inline int _h_mask_exact(self, int* x, int* y, int* z,
            int H) nogil:
        cdef int length = 0
        cdef int s, t, u

        for s from -H<=s<=H:
            for t from -H<=t<=H:
                for u from -H<=u<=H:
                    x[length] = s
                    y[length] = t
                    z[length] = u
                    length += 1

        return length

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z, int H) nogil:
        cdef int length = 0
        cdef int p

        cdef int mask_len = (2*H+1)*(2*H+1)*(2*H+1)

        cdef int* x_mask = <int*> malloc(mask_len*sizeof(int))
        cdef int* y_mask = <int*> malloc(mask_len*sizeof(int))
        cdef int* z_mask = <int*> malloc(mask_len*sizeof(int))

        mask_len = self._h_mask_exact(x_mask, y_mask, z_mask, H)

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

    cdef inline void _set_h_max(self, double* current_cells, double* src_h_ptr,
            int num_particles) nogil:
        cdef double h
        cdef int i, idx
        for i from 0<=i<num_particles:
            h = src_h_ptr[i]
            idx = self._get_hash_id(h)
            current_cells[idx] = fmax(h, current_cells[idx])

    @cython.cdivision(True)
    cpdef _refresh(self):
        self.interval_size = (self.cell_size - self.hmin)/self.num_levels + EPS

        cdef HashTable** current_hash
        cdef int i, j
        for i from 0<=i<self.narrays:
            current_hash = self.hashtable[i]
            current_cells = self.cell_sizes[i]
            for j from 0<=j<self.num_levels:
                if current_hash[j] != NULL:
                    del current_hash[j]
                current_hash[j] = new HashTable(self.table_size)
                current_cells[j] = 0
        self.current_hash = self.hashtable[self.src_index]
        self.current_cells = self.cell_sizes[self.src_index]

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        cdef double* src_x_ptr = pa_wrapper.x.data
        cdef double* src_y_ptr = pa_wrapper.y.data
        cdef double* src_z_ptr = pa_wrapper.z.data
        cdef double* src_h_ptr = pa_wrapper.h.data

        cdef double* xmin = self.xmin.data

        cdef u_int i, idx
        cdef int hash_id
        cdef int c_x, c_y, c_z
        cdef double cell_size
        cdef double* current_cells = self.cell_sizes[pa_index]

        self._set_h_max(current_cells, src_h_ptr, pa_wrapper.get_number_of_particles())
        cdef HashTable** current_hash = self.hashtable[pa_index]

        for i from 0<=i<indices.length:
            idx = indices.data[i]
            hash_id = self._get_hash_id(src_h_ptr[idx])
            cell_size = self._get_h_max(current_cells, hash_id)/self.H
            find_cell_id_raw(
                    src_x_ptr[idx] - xmin[0],
                    src_y_ptr[idx] - xmin[1],
                    src_z_ptr[idx] - xmin[2],
                    cell_size,
                    &c_x, &c_y, &c_z
                    )
            current_hash[hash_id].add(c_x, c_y, c_z, idx, src_h_ptr[idx])

