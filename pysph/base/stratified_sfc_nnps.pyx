#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libc.stdio cimport printf

from cython.operator cimport dereference as deref, preincrement as inc

from .nnps_base cimport *
# Cython for compiler directives
cimport cython

import numpy as np
cimport numpy as np

DEF EPS = 1e-13

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)
    void sort[Iter](Iter first, Iter last)

IF UNAME_SYSNAME == "Windows":
    cdef inline double fmin(double x, double y) nogil:
        return x if x < y else y
    cdef inline double fmax(double x, double y) nogil:
        return x if x > y else y

    @cython.cdivision(True)
    cdef inline double log2(double n) nogil:
        return log(n)/log(2)

#############################################################################
cdef class StratifiedSFCNNPS(NNPS):

    """Finds nearest neighbors using Space-filling curves with stratified grids"""

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int num_levels = 1):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.radius_scale2 = radius_scale*radius_scale
        self.interval_size = 0

        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int num_levels = 1):

        cdef int narrays = len(particles)

        if fixed_h:
            self.num_levels = 1
        else:
            self.num_levels = num_levels

        self.pids = <uint32_t**> malloc(narrays*sizeof(uint32_t*))
        self.keys = <uint64_t**> malloc(narrays*sizeof(uint64_t*))
        self.key_to_idx = <int***> malloc(narrays*sizeof(int**))
        self.cell_sizes = <double**> malloc(narrays*sizeof(double*))
        self.max_keys = <uint64_t*> malloc(narrays*sizeof(uint64_t))

        cdef int i, j, num_particles
        for i from 0<=i<narrays:
            self.pids[i] = NULL
            self.keys[i] = NULL
            self.key_to_idx[i] = <int**> malloc(self.num_levels*sizeof(int*))
            for j from 0<=j<self.num_levels:
                self.key_to_idx[i][j] = NULL
            self.cell_sizes[i] = <double*> malloc(self.num_levels*sizeof(double))

        self.current_pids = NULL
        self.current_keys = NULL
        self.current_key_to_idx = NULL
        self.current_cells = NULL

    def __dealloc__(self):
        cdef uint32_t* current_pids
        cdef uint64_t* current_keys
        cdef int** current_key_to_idx
        cdef int i, j
        for i from 0<=i<self.narrays:
            current_pids = self.pids[i]
            current_keys = self.keys[i]
            current_key_to_idx = self.key_to_idx[i]
            for j from 0<=j<self.num_levels:
                free(current_key_to_idx[j])
            free(current_pids)
            free(current_keys)
            free(current_key_to_idx)
            free(self.cell_sizes[i])
        free(self.pids)
        free(self.keys)
        free(self.key_to_idx)
        free(self.cell_sizes)

    #### Public protocol ################################################

    cpdef double get_binning_size(self, int interval):
        """Get bin size at a level"""
        return self.radius_scale*self.current_cells[interval]

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
        self.current_pids = self.pids[src_index]
        self.current_keys = self.keys[src_index]
        self.current_cells = self.cell_sizes[src_index]
        self.current_key_to_idx = self.key_to_idx[src_index]
        self.current_max_key = self.max_keys[src_index]

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        indices.reset()
        cdef int num_particles = (<NNPSParticleArrayWrapper> \
                self.pa_wrappers[pa_index]).get_number_of_particles()
        cdef uint32_t* current_pids = self.pids[pa_index]

        cdef int j
        for j from 0<=j<num_particles:
            indices.c_append(current_pids[j])

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

        cdef int num_particles = self.src.x.length

        cdef unsigned int* s_gid = self.src.gid.data
        cdef int orig_length = nbrs.length

        cdef int c_x, c_y, c_z
        cdef double* xmin = self.xmin.data
        cdef unsigned int i, j, k, n, pid
        cdef int idx

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2*h*h
        cdef double hj2 = 0

        cdef double h_max
        cdef int H, mask_len, num_boxes

        cdef int* x_boxes
        cdef int* y_boxes
        cdef int* z_boxes

        cdef uint64_t key, key_padded, level_padded
        cdef int* key_idx_level

        for i from 0<=i<self.num_levels:
            key_idx_level = self.current_key_to_idx[i]
            level_padded = i << self.max_num_bits

            h_max = fmax(self.radius_scale*h,
                    self.radius_scale*self.current_cells[i])
            H = <int> ceil(h_max/(self.radius_scale*self.current_cells[i]))

            mask_len = (2*H+1) ** 3

            x_boxes = <int*> malloc(mask_len*sizeof(int))
            y_boxes = <int*> malloc(mask_len*sizeof(int))
            z_boxes = <int*> malloc(mask_len*sizeof(int))

            find_cell_id_raw(
                    x - xmin[0],
                    y - xmin[1],
                    z - xmin[2],
                    self.radius_scale*self.current_cells[i],
                    &c_x, &c_y, &c_z
                    )

            num_boxes = self._neighbor_boxes(c_x, c_y, c_z,
                    x_boxes, y_boxes, z_boxes, H)

            for j from 0<=j<num_boxes:
                key = get_key(x_boxes[j], y_boxes[j], z_boxes[j])
                key_padded = level_padded + key
                idx = -1 if key >= self.current_max_key else key_idx_level[key]

                if idx == -1:
                    continue

                while idx < num_particles and self.current_keys[idx] == key_padded:
                    pid = self.current_pids[idx]

                    hj2 = self.radius_scale2*src_h_ptr[pid]*src_h_ptr[pid]

                    xij2 = norm2(
                        src_x_ptr[pid] - x,
                        src_y_ptr[pid] - y,
                        src_z_ptr[pid] - z
                        )

                    if (xij2 < hi2) or (xij2 < hj2):
                        nbrs.c_append(pid)

                    idx += 1

            free(x_boxes)
            free(y_boxes)
            free(z_boxes)

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )

    cpdef int get_number_of_particles(self, int pa_index, int level):
        cdef int length = 1
        cdef int i = 0
        cdef int num_particles = (<NNPSParticleArrayWrapper> \
                self.pa_wrappers[pa_index]).get_number_of_particles()
        cdef uint32_t* current_pids = self.pids[pa_index]
        cdef uint64_t* current_keys = self.keys[pa_index]

        while (current_keys[i] >> self.max_num_bits) != level:
            i += 1

        while (current_keys[i] >> self.max_num_bits == \
                current_keys[i+1] >> self.max_num_bits):
                    length += 1
                    i += 1
                    if i == num_particles - 1:
                        break
        return length

    #### Private protocol ################################################
    cpdef np.ndarray get_keys(self, pa_index):
        cdef uint64_t* current_keys = self.keys[pa_index]
        cdef int num_particles = (<NNPSParticleArrayWrapper> \
                self.pa_wrappers[pa_index]).get_number_of_particles()
        keys = np.empty(num_particles)
        for i from 0<=i<num_particles:
            keys[i] = current_keys[i]
        return keys

    @cython.cdivision(True)
    cdef inline int _get_level(self, double h) nogil:
        return <int> floor((self.radius_scale*h - self.hmin)/self.interval_size)

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z, int H) nogil:
        cdef int length = 0
        cdef int p, q, r
        for p from -H<=p<=H:
            for q from -H<=q<=H:
                for r from -H<=r<=H:
                    if i+r>=0 and j+q>=0 and k+p>=0:
                        x[length] = i+r
                        y[length] = j+q
                        z[length] = k+p
                        length += 1
        return length

    @cython.cdivision(True)
    cpdef _refresh(self):
        self.interval_size = ((self.cell_size - self.hmin)/self.num_levels) + EPS

        cdef double* current_cells

        cdef int i, j, num_particles
        for i from 0<=i<self.narrays:
            num_particles = (<NNPSParticleArrayWrapper> \
                    self.pa_wrappers[i]).get_number_of_particles()
            if self.pids[i] != NULL:
                free(self.pids[i])
            if self.keys[i] != NULL:
                free(self.keys[i])
            self.pids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))
            self.keys[i] = <uint64_t*> malloc(num_particles*sizeof(uint64_t))
            current_cells = self.cell_sizes[i]
            for j from 0<=j<self.num_levels:
                current_cells[j] = 0
                if self.key_to_idx[i][j] != NULL:
                    free(self.key_to_idx[i][j])
        self.current_pids = self.pids[self.src_index]
        self.current_cells = self.cell_sizes[self.src_index]

        cdef double max_length = fmax(fmax((self.xmax.data[0] - \
                self.xmin.data[0]),
            (self.xmax.data[1] - self.xmin.data[1])),
            (self.xmax.data[2] - self.xmin.data[2]))

        cdef int max_num_cells = (<int> ceil(max_length/self.hmin))

        self.max_num_bits = 1 + 3*(<int> ceil(log2(max_num_cells)))

    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper,
            int pa_index, UIntArray indices, uint32_t* current_pids,
            uint64_t* current_keys, double* current_cells):
        cdef double* x_ptr = pa_wrapper.x.data
        cdef double* y_ptr = pa_wrapper.y.data
        cdef double* z_ptr = pa_wrapper.z.data
        cdef double* h_ptr = pa_wrapper.h.data

        cdef double* xmin = self.xmin.data

        cdef int c_x, c_y, c_z

        cdef int i, j, n, level
        cdef uint64_t level_padded, key

        # Finds cell sizes at each level
        for i from 0<=i<indices.length:
            n = indices.data[i]
            level = self._get_level(h_ptr[n])
            current_cells[level] = fmax(h_ptr[n], current_cells[level])

        cdef double cell_size

        cdef uint64_t max_key = 0

        for i from 0<=i<indices.length:
            n = indices.data[i]
            current_pids[i] = n
            level = self._get_level(h_ptr[n])
            level_padded = level << self.max_num_bits
            cell_size = self.radius_scale*current_cells[level]
            find_cell_id_raw(
                    x_ptr[i] - xmin[0],
                    y_ptr[i] - xmin[1],
                    z_ptr[i] - xmin[2],
                    cell_size,
                    &c_x, &c_y, &c_z
                    )
            key = get_key(c_x, c_y, c_z)
            current_keys[i] = level_padded + key

            max_key = max(max_key, key)

        max_key += 1

        self.max_keys[pa_index] = max_key

        cdef CompareSortWrapper sort_wrapper = CompareSortWrapper(
                current_pids, current_keys, indices.length)

        sort_wrapper.compare_sort()

        cdef int** current_key_to_idx = self.key_to_idx[pa_index]
        cdef int* key_idx_level

        for i from 0<=i<self.num_levels:
            current_key_to_idx[i] = <int*> malloc(max_key * sizeof(int))
            key_idx_level = current_key_to_idx[i]

            for j from 0<=j<max_key:
                key_idx_level[j] = -1

        ################################################################

        cdef uint64_t key_stripped, strip_mask

        strip_mask = (1 << self.max_num_bits) - 1

        key = current_keys[0]
        level = key >> self.max_num_bits
        key_stripped = key & strip_mask
        current_key_to_idx[level][key_stripped] = 0;

        for i from 0<i<indices.length:
            key = current_keys[i]
            if key != current_keys[i-1]:
                level = key >> self.max_num_bits
                key_idx_level = current_key_to_idx[level]
                key_stripped = key & strip_mask
                key_idx_level[key_stripped] = i

        ################################################################

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        cdef double* current_cells = self.cell_sizes[pa_index]
        cdef uint32_t* current_pids = self.pids[pa_index]
        cdef uint64_t* current_keys = self.keys[pa_index]

        self.fill_array(pa_wrapper, pa_index, indices, current_pids,
                current_keys, current_cells)
