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
            bint cache = False, bint sort_gids = False, int num_levels = 1,
            bint asymmetric = True):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.asymmetric = asymmetric

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
        self.hmax = <double***> malloc(narrays*sizeof(double**))
        self.max_keys = <uint64_t*> malloc(narrays*sizeof(uint64_t))
        self.num_cells = <int**> malloc(narrays*sizeof(int*))
        self.nbr_boxes = <vector[int]**> malloc(narrays*sizeof(void*))
        self.key_to_nbr_idx = <int***> malloc(narrays*sizeof(int**))
        self.key_to_nbr_length = <int***> malloc(narrays*sizeof(int**))
        self.total_mask_len = <int*> malloc(narrays*sizeof(int))

        cdef int i, j, num_particles
        for i from 0<=i<narrays:
            self.pids[i] = NULL
            self.keys[i] = NULL
            self.nbr_boxes[i] = NULL
            self.key_to_idx[i] = <int**> malloc(self.num_levels*sizeof(int*))
            self.key_to_nbr_idx[i] = <int**> malloc(self.num_levels*sizeof(int*))
            self.key_to_nbr_length[i] = <int**> malloc(self.num_levels*sizeof(int*))
            for j from 0<=j<self.num_levels:
                self.key_to_idx[i][j] = NULL
                self.key_to_nbr_idx[i][j] = NULL
                self.key_to_nbr_length[i][j] = NULL
            self.cell_sizes[i] = <double*> malloc(self.num_levels*sizeof(double))
            self.hmax[i] = <double**> malloc(self.num_levels*sizeof(double*))
            self.num_cells[i] = <int*> malloc(self.num_levels*sizeof(int))
            self.total_mask_len[i] = 0

        self.current_pids = NULL
        self.current_keys = NULL
        self.current_key_to_idx = NULL
        self.current_cells = NULL
        self.current_hmax = NULL
        self.current_nbr_boxes = NULL
        self.current_num_cells = NULL
        self.current_key_to_nbr_idx = NULL
        self.current_key_to_nbr_length = NULL
        self.current_mask_len = 0

    def __dealloc__(self):
        cdef uint32_t* current_pids
        cdef uint64_t* current_keys
        cdef int** current_key_to_idx
        cdef int** current_key_to_nbr_idx
        cdef int** current_key_to_nbr_length
        cdef double** current_hmax
        cdef int i, j
        for i from 0<=i<self.narrays:
            current_pids = self.pids[i]
            current_keys = self.keys[i]
            current_key_to_idx = self.key_to_idx[i]
            current_key_to_nbr_idx = self.key_to_nbr_idx[i]
            current_key_to_nbr_length = self.key_to_nbr_length[i]
            current_hmax = self.hmax[i]
            del self.nbr_boxes[i]
            for j from 0<=j<self.num_levels:
                free(current_key_to_idx[j])
                free(current_key_to_nbr_idx[j])
                free(current_key_to_nbr_length[j])
                free(current_hmax[j])
            free(current_pids)
            free(current_keys)
            free(current_key_to_idx)
            free(current_key_to_nbr_idx)
            free(current_key_to_nbr_length)
            free(current_hmax)
            free(self.cell_sizes[i])
            free(self.num_cells[i])
        free(self.pids)
        free(self.keys)
        free(self.key_to_idx)
        free(self.cell_sizes)
        free(self.key_to_nbr_idx)
        free(self.key_to_nbr_length)
        free(self.nbr_boxes)
        free(self.hmax)
        free(self.total_mask_len)
        free(self.max_keys)

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
        self.current_key_to_nbr_idx = self.key_to_nbr_idx[src_index]
        self.current_key_to_nbr_length = self.key_to_nbr_length[src_index]
        self.current_max_key = self.max_keys[src_index]
        self.current_nbr_boxes = self.nbr_boxes[src_index]

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
        cdef unsigned int i, pid
        cdef int idx

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2 * h * h
        cdef double hj2 = 0

        cdef uint64_t key

        cdef uint32_t q_level = self._get_level(h)

        find_cell_id_raw(
                x - xmin[0],
                y - xmin[1],
                z - xmin[2],
                self.radius_scale*self.current_cells[q_level],
                &c_x, &c_y, &c_z
                )

        cdef int* current_key_to_nbr_idx_level
        cdef int* current_key_to_nbr_length_level
        cdef int num_boxes, start_idx

        current_key_to_nbr_idx_level = self.current_key_to_nbr_idx[q_level]
        current_key_to_nbr_length_level = self.current_key_to_nbr_length[q_level]

        find_cell_id_raw(
                x - xmin[0],
                y - xmin[1],
                z - xmin[2],
                self.radius_scale*self.current_cells[0],
                &c_x, &c_y, &c_z
                )

        key_bottom = get_key(c_x, c_y, c_z)

        num_boxes = current_key_to_nbr_length_level[key_bottom]
        start_idx = current_key_to_nbr_idx_level[key_bottom]

        if not (start_idx >= 0 and num_boxes > 0):
            return

        for i from start_idx<=i<start_idx + num_boxes:
            idx = deref(self.current_nbr_boxes)[i]
            key = self.current_keys[idx]

            while idx < num_particles and self.current_keys[idx] == key:
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

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )

    cpdef int get_number_of_particles(self, int pa_index, int level):
        cdef int length = 0
        cdef int i = 0
        cdef int num_particles = (<NNPSParticleArrayWrapper> \
                self.pa_wrappers[pa_index]).get_number_of_particles()
        cdef uint64_t* current_keys = self.keys[pa_index]

        while i < num_particles and (current_keys[i] >> self.max_num_bits) != level:
            i += 1

        if i == num_particles:
            return length

        length += 1
        i += 1

        while i < num_particles and (current_keys[i] >> self.max_num_bits == \
                current_keys[i-1] >> self.max_num_bits):
                    length += 1
                    i += 1

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
        return self.num_levels - <int> min(self.num_levels,
                ceil(log2((self.cell_size + EPS)/ self.radius_scale / h)))

    @cython.cdivision(True)
    cdef inline int _get_H(self, double h_q, double h_j):
        return <int> ceil(h_q / h_j)

    cdef inline int get_idx(self, uint64_t key, uint64_t max_key, int* key_to_idx) nogil:
        return -1 if key >= max_key else key_to_idx[key]

    cdef int _neighbor_boxes_func(self, int i, int j, int k, int H,
            int* current_key_to_idx_level, uint64_t max_key,
            double current_cell_size, double* current_hmax_level, 
            vector[int]* nbr_boxes):
        if self.asymmetric:
            return self._neighbor_boxes_asym(i, j, k, H, current_key_to_idx_level,
                    max_key, current_cell_size, current_hmax_level, nbr_boxes)
        else:
            return self._neighbor_boxes_sym(i, j, k, H, current_key_to_idx_level,
                    max_key, current_cell_size, current_hmax_level, nbr_boxes)

    cdef int _neighbor_boxes_asym(self, int i, int j, int k, int H,
            int* current_key_to_idx_level, uint64_t max_key,
            double current_cell_size, double* current_hmax_level,
            vector[int]* nbr_boxes) nogil:
        cdef int length = 0

        cdef uint64_t key
        cdef int found_idx

        cdef int x_temp, y_temp, z_temp

        cdef int s, t, u

        for s from -H<=s<=H:
            for t from -H<=t<=H:
                for u from -H<=u<=H:

                    x_temp = i + u
                    y_temp = j + t
                    z_temp = k + s

                    if x_temp >= 0 and y_temp >= 0 and z_temp >= 0:
                        key = get_key(x_temp, y_temp, z_temp)
                        found_idx = self.get_idx(key, max_key, current_key_to_idx_level)

                        if found_idx == -1:
                            continue

                        nbr_boxes.push_back(found_idx)
                        length += 1

        return length

    @cython.cdivision(True)
    cdef int _neighbor_boxes_sym(self, int i, int j, int k, int H,
            int* current_key_to_idx_level, uint64_t max_key,
            double current_cell_size, double* current_hmax_level,
            vector[int]* nbr_boxes) nogil:
        cdef int length = 0

        cdef uint64_t key

        cdef int x_temp, y_temp, z_temp

        cdef double h_local
        cdef int H_eff

        cdef int s, t, u

        cdef uint64_t qkey = get_key(i, j, k)

        for s from -H<=s<=H:
            for t from -H<=t<=H:
                for u from -H<=u<=H:

                    x_temp = i + u
                    y_temp = j + t
                    z_temp = k + s

                    if x_temp >= 0 and y_temp >= 0 and z_temp >= 0:
                        key = get_key(x_temp, y_temp, z_temp)
                        found_idx = self.get_idx(key, max_key, current_key_to_idx_level)

                        if found_idx == -1:
                            continue

                        h_local = fmax(current_hmax_level[key], current_hmax_level[qkey])
                        H_eff = <int> ceil(h_local / current_cell_size)

                        if abs(u) <= H_eff and abs(t) <= H_eff and abs(s) <= H_eff:
                            nbr_boxes.push_back(found_idx)
                            length += 1

        return length

    cdef void _fill_nbr_boxes(self):
        cdef int i, j, k
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int num_particles

        cdef vector[int]* current_nbr_boxes
        cdef uint32_t* current_pids
        cdef uint64_t* current_keys
        cdef int** current_key_to_idx
        cdef double** current_hmax

        cdef double* x_ptr
        cdef double* y_ptr
        cdef double* z_ptr
        cdef double* h_ptr

        cdef int c_x, c_y, c_z
        cdef int num_boxes

        cdef int n = 0
        cdef uint32_t pid

        cdef uint64_t key, key_bottom
        cdef uint64_t key_stripped, strip_mask, current_max_key
        cdef int mask_length = 0

        strip_mask = (1 << self.max_num_bits) - 1

        for i from 0<=i<self.narrays:
            n = 0
            pa_wrapper = self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()
            x_ptr = pa_wrapper.x.data
            y_ptr = pa_wrapper.y.data
            z_ptr = pa_wrapper.z.data
            h_ptr = pa_wrapper.h.data
            xmin = self.xmin.data

            current_keys = self.keys[i]
            current_key_to_idx = self.key_to_idx[i]
            current_pids = self.pids[i]
            current_nbr_boxes = self.nbr_boxes[i]
            current_cells = self.cell_sizes[i]
            current_hmax = self.hmax[i]
            current_key_to_nbr_idx = self.key_to_nbr_idx[i]
            current_key_to_nbr_length = self.key_to_nbr_length[i]
            current_max_key = self.max_keys[i]

            # init self.nbr_boxes to -1
            for j from 0<=j<self.total_mask_len[i]:
                deref(current_nbr_boxes)[j] = -1

            for j from 0<=j<num_particles:
                key = current_keys[j]
                pid = current_pids[j]
                level = key >> self.max_num_bits
                key_stripped = key & strip_mask
                current_hmax_level = current_hmax[level]
                if j == 0 or key != current_keys[j-1]:
                    current_hmax_level[key_stripped] = h_ptr[pid]
                else:
                    current_hmax_level[key_stripped] = fmax(current_hmax_level[key_stripped], h_ptr[pid])
            
            pid = current_pids[0]
            key = current_keys[0]
            
            find_cell_id_raw(
                x_ptr[pid] - xmin[0],
                y_ptr[pid] - xmin[1],
                z_ptr[pid] - xmin[2],
                self.radius_scale*current_cells[0],
                &c_x, &c_y, &c_z
                )

            key_bottom = get_key(c_x, c_y, c_z)

            level = key >> self.max_num_bits
            key_stripped = key & strip_mask
            current_hmax_level = current_hmax[level]
            hmax_cell = current_hmax_level[key_stripped]
            mask_length = 0
            
            current_key_to_nbr_idx_level = current_key_to_nbr_idx[level]
            current_key_to_nbr_length_level = current_key_to_nbr_length[level]

            current_key_to_nbr_idx_level[key_bottom] = n

            for k from 0<=k<self.num_levels:
                if current_cells[k] == 0:
                    continue

                find_cell_id_raw(
                    x_ptr[pid] - xmin[0],
                    y_ptr[pid] - xmin[1],
                    z_ptr[pid] - xmin[2],
                    self.radius_scale*current_cells[k],
                    &c_x, &c_y, &c_z
                    )

                H = self._get_H(hmax_cell, current_cells[k])
                
                num_boxes = self._neighbor_boxes_func(
                        c_x, c_y, c_z,
                        H, current_key_to_idx[k], current_max_key,
                        current_cells[k], current_hmax[k],
                        current_nbr_boxes
                        )

                n += num_boxes
                mask_length += num_boxes
            
            current_key_to_nbr_length_level[key_bottom] = mask_length

            # nbrs of all cids in particle array i
            for j from 0<j<num_particles:
                key = current_keys[j]
                pid = current_pids[j]

                find_cell_id_raw(
                    x_ptr[pid] - xmin[0],
                    y_ptr[pid] - xmin[1],
                    z_ptr[pid] - xmin[2],
                    self.radius_scale*current_cells[0],
                    &c_x, &c_y, &c_z
                    )

                key_bottom = get_key(c_x, c_y, c_z)

                level = key >> self.max_num_bits
                key_stripped = key & strip_mask
                current_hmax_level = current_hmax[level]
                hmax_cell = current_hmax_level[key_stripped]
                mask_length = 0
                current_key_to_nbr_idx_level = current_key_to_nbr_idx[level]
                current_key_to_nbr_length_level = current_key_to_nbr_length[level]
                    
                if current_key_to_nbr_idx_level[key_bottom] == -1:
                    current_key_to_nbr_idx_level[key_bottom] = n
 
                    for k from 0<=k<self.num_levels:
                        if current_cells[k] == 0:
                            continue

                        find_cell_id_raw(
                            x_ptr[pid] - xmin[0],
                            y_ptr[pid] - xmin[1],
                            z_ptr[pid] - xmin[2],
                            self.radius_scale*current_cells[k],
                            &c_x, &c_y, &c_z
                            )

                        H = self._get_H(hmax_cell, current_cells[k])
                        
                        num_boxes = self._neighbor_boxes_func(
                                c_x, c_y, c_z,
                                H, current_key_to_idx[k], current_max_key,
                                current_cells[k], current_hmax[k],
                                current_nbr_boxes
                                )

                        n += num_boxes
                        mask_length += num_boxes
                    
                    current_key_to_nbr_length_level[key_bottom] = mask_length

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
            if self.nbr_boxes[i] != NULL:
                del self.nbr_boxes[i]
            self.pids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))
            self.keys[i] = <uint64_t*> malloc(num_particles*sizeof(uint64_t))
            current_cells = self.cell_sizes[i]
            for j from 0<=j<self.num_levels:
                current_cells[j] = 0
                if self.key_to_idx[i][j] != NULL:
                    free(self.key_to_idx[i][j])
                if self.key_to_nbr_idx[i][j] != NULL:
                    free(self.key_to_nbr_idx[i][j])
                if self.key_to_nbr_length[i][j] != NULL:
                    free(self.key_to_nbr_length[i][j])
        self.current_pids = self.pids[self.src_index]
        self.current_cells = self.cell_sizes[self.src_index]

        cdef double max_length = fmax(fmax((self.xmax.data[0] - \
                self.xmin.data[0]),
            (self.xmax.data[1] - self.xmin.data[1])),
            (self.xmax.data[2] - self.xmin.data[2]))

        cdef int max_num_cells = (<int> ceil(max_length/self.hmin))

        self.max_num_bits = 1 + 3*(<int> ceil(log2(max_num_cells)))

        self.max_possible_key = 0

        cdef NNPSParticleArrayWrapper pa_wrapper

        cdef uint32_t* current_pids
        cdef uint64_t* current_keys
        cdef int** current_key_to_nbr_idx 
        cdef int** current_key_to_nbr_length
        cdef int* current_key_to_nbr_idx_level
        cdef int* current_key_to_nbr_length_level
        cdef uint64_t key_iter

        cdef int H_jk, num_cells_level

        for i from 0<=i<self.narrays:
            self.total_mask_len[i] = 0
            pa_wrapper = self.pa_wrappers[i]
            current_cells = self.cell_sizes[i]
            current_pids = self.pids[i]
            current_keys = self.keys[i]
            current_num_cells = self.num_cells[i]

            self.fill_array(pa_wrapper, i, current_pids,
                    current_keys, current_cells, current_num_cells)

            current_key_to_nbr_idx = self.key_to_nbr_idx[i]
            current_key_to_nbr_length = self.key_to_nbr_idx[i]
            
            for j from 0<=j<self.num_levels:
                num_cells_level = current_num_cells[j]
                for k from 0<=k<self.num_levels:
                    if current_cells[j] == 0 or current_cells[k] == 0:
                        continue
                    H_jk = self._get_H(current_cells[j], current_cells[k])
                    self.total_mask_len[i] += num_cells_level * (2 * H_jk + 1) ** 3

            self.nbr_boxes[i] = new vector[int]()
            self.nbr_boxes[i].reserve(self.total_mask_len[i])

        self.max_possible_key += 1

        for i from 0<=i<self.narrays:
            current_key_to_nbr_idx = self.key_to_nbr_idx[i]
            current_key_to_nbr_length = self.key_to_nbr_length[i]
            for j from 0<=j<self.num_levels:
                current_key_to_nbr_idx[j] = <int*> malloc(self.max_possible_key * sizeof(int))
                current_key_to_nbr_length[j] = <int*> malloc(self.max_possible_key * sizeof(int))
                current_key_to_nbr_idx_level = current_key_to_nbr_idx[j]
                current_key_to_nbr_length_level = current_key_to_nbr_length[j]

                for key_iter from 0<=key_iter<self.max_possible_key:
                    current_key_to_nbr_idx_level[key_iter] = -1
                    current_key_to_nbr_length_level[key_iter] = -1

        self._fill_nbr_boxes()

    @cython.cdivision(True)
    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper,
            int pa_index, uint32_t* current_pids,
            uint64_t* current_keys, double* current_cells, int* current_num_cells):
        cdef double* x_ptr = pa_wrapper.x.data
        cdef double* y_ptr = pa_wrapper.y.data
        cdef double* z_ptr = pa_wrapper.z.data
        cdef double* h_ptr = pa_wrapper.h.data

        cdef int curr_num_particles = pa_wrapper.get_number_of_particles()

        cdef double* xmax = self.xmax.data
        cdef double* xmin = self.xmin.data

        cdef int c_x, c_y, c_z

        cdef int i, j, level
        cdef uint64_t level_padded, key

        # Finds cell sizes at each level
        for i from 0<=i<self.num_levels:
            current_cells[i] = (self.cell_size / self.radius_scale) / (2 ** (self.num_levels - i - 1))

        cdef double cell_size

        cdef uint64_t max_key = 0

        for i from 0<=i<curr_num_particles:
            current_pids[i] = i
            level = self._get_level(h_ptr[i])
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

            find_cell_id_raw(
                    x_ptr[i] - xmin[0],
                    y_ptr[i] - xmin[1],
                    z_ptr[i] - xmin[2],
                    self.radius_scale*current_cells[0],
                    &c_x, &c_y, &c_z
                    )
            key = get_key(c_x, c_y, c_z)
            self.max_possible_key = max(key, self.max_possible_key)

        max_key += 1

        self.max_keys[pa_index] = max_key

        cdef CompareSortWrapper sort_wrapper = CompareSortWrapper(
                current_pids, current_keys, curr_num_particles)

        sort_wrapper.compare_sort()

        cdef int** current_key_to_idx = self.key_to_idx[pa_index]
        cdef double** current_hmax = self.hmax[pa_index]
        cdef int* key_idx_level
        cdef double* hmax_level

        for i from 0<=i<self.num_levels:
            current_key_to_idx[i] = <int*> malloc(max_key * sizeof(int))
            current_hmax[i] = <double*> malloc(max_key * sizeof(double))
            key_idx_level = current_key_to_idx[i]
            hmax_level = current_hmax[i]
            current_num_cells[i] = 0

            for j from 0<=j<max_key:
                key_idx_level[j] = -1
                hmax_level[j] = -1

        ################################################################

        cdef uint64_t key_stripped, strip_mask

        strip_mask = (1 << self.max_num_bits) - 1

        key = current_keys[0]
        level = key >> self.max_num_bits
        key_stripped = key & strip_mask
        current_key_to_idx[level][key_stripped] = 0;
        current_num_cells[level] += 1

        for i from 0<i<curr_num_particles:
            key = current_keys[i]
            if key != current_keys[i-1]:
                level = key >> self.max_num_bits
                key_idx_level = current_key_to_idx[level]
                key_stripped = key & strip_mask
                key_idx_level[key_stripped] = i
                current_num_cells[level] += 1

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        pass
