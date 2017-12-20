#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stdlib cimport bsearch
from libc.stdio cimport printf

from cython.operator cimport dereference as deref, preincrement as inc

# Cython for compiler directives
cimport cython

import numpy as np
cimport numpy as np

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)
    void sort[Iter](Iter first, Iter last)

#############################################################################

cdef inline int cmp_func(const void* a, const void* b) nogil:
    return (<uint64_t*>a)[0] - (<uint64_t*>b)[0]

cdef class ZOrderNNPS(NNPS):

    """Find nearest neighbors using Z-Order space filling curve"""

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int H=1,
            bint asymmetric=False):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.radius_scale2 = radius_scale*radius_scale
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles

        self.H = H
        self.mask_len = (2 * H + 1) ** 3

        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int H=1,
            bint asymmetric=False):
        cdef int narrays = len(particles)

        self.pids = <uint32_t**> malloc(narrays*sizeof(uint32_t*))
        self.keys = <uint64_t**> malloc(narrays*sizeof(uint64_t*))
        self.cids = <uint32_t**> malloc(narrays*sizeof(uint32_t*))
        self.nbr_boxes = <int**> malloc(narrays*sizeof(int*))
        self.lengths = <int**> malloc(narrays*sizeof(int*))

        cdef int i
        for i from 0<=i<narrays:
            self.pids[i] = NULL
            self.keys[i] = NULL
            self.cids[i] = NULL
            self.nbr_boxes[i] = NULL
            self.lengths[i] = NULL

        self.current_pids = NULL
        self.current_cids_src = NULL
        self.current_cids_dst = NULL
        self.current_keys = NULL
        self.current_nbr_boxes = NULL
        self.current_lengths = NULL

    def __dealloc__(self):
        cdef int i
        for i from 0<=i<self.narrays:
            if self.pids[i] != NULL:
                free(self.pids[i])
            if self.keys[i] != NULL:
                free(self.keys[i])
            if self.cids[i] != NULL:
                free(self.cids[i])
            if self.nbr_boxes[i] != NULL:
                free(self.nbr_boxes[i])
            if self.lengths[i] != NULL:
                free(self.lengths[i])
        free(self.pids)
        free(self.keys)
        free(self.cids)
        free(self.nbr_boxes)
        free(self.lengths)

    cdef inline uint64_t* find(self, uint64_t query, uint64_t* array,
            int num_particles) nogil:
        cdef uint64_t* found_ptr = <uint64_t*> bsearch(&query,
                array, num_particles, sizeof(uint64_t),
                <int(*)(const void *, const void *) nogil> cmp_func)

        if found_ptr != NULL:
            while found_ptr != array and found_ptr[0] == (found_ptr - 1)[0]:
                found_ptr -= 1

        return found_ptr

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
        self.current_cids_dst = self.cids[dst_index]
        self.current_cids_src = self.cids[src_index]
        self.current_nbr_boxes = self.nbr_boxes[src_index]
        self.current_lengths = self.lengths[src_index]

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

        cdef int num_particles = self.src.x.length

        cdef uint32_t cid = self.current_cids_dst[d_idx]
        cdef uint32_t cid_nbr

        cdef unsigned int* s_gid = self.src.gid.data
        cdef int orig_length = nbrs.length

        cdef int c_x, c_y, c_z
        cdef double* xmin = self.xmin.data
        cdef int i, j

        find_cell_id_raw(
                x - xmin[0],
                y - xmin[1],
                z - xmin[2],
                self.h_sub,
                &c_x, &c_y, &c_z
                )

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2*h*h
        cdef double hj2 = 0

        cdef int start_idx, curr_idx
        cdef uint32_t n, idx
        cdef uint64_t key
        cdef int length

        for i from 0<=i<self.mask_len:
            start_idx = self.current_nbr_boxes[self.mask_len * cid + i]

            if start_idx < 0:
                break

            key = self.current_keys[start_idx]
            idx = self.current_pids[start_idx]
            cid_nbr = self.current_cids_src[idx]
            length = self.current_lengths[cid_nbr]

            for j from 0<=j<length:
                idx = self.current_pids[start_idx + j]

                hj2 = self.radius_scale2*src_h_ptr[idx]*src_h_ptr[idx]

                xij2 = norm2(
                    src_x_ptr[idx] - x,
                    src_y_ptr[idx] - y,
                    src_z_ptr[idx] - z
                    )

                if (xij2 < hi2) or (xij2 < hj2):
                    nbrs.c_append(idx)

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

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        indices.reset()
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        cdef int num_particles = pa_wrapper.get_number_of_particles()

        cdef uint32_t* current_pids = self.pids[pa_index]

        cdef int j
        for j from 0<=j<num_particles:
            indices.c_append(<long>current_pids[j])

    cdef int fill_array(self, NNPSParticleArrayWrapper pa_wrapper, int pa_index,
            uint32_t* current_pids, uint64_t* current_keys,
            uint32_t* current_cids, uint32_t curr_cid):

        cdef double* x_ptr = pa_wrapper.x.data
        cdef double* y_ptr = pa_wrapper.y.data
        cdef double* z_ptr = pa_wrapper.z.data

        cdef int curr_num_particles = pa_wrapper.get_number_of_particles()

        cdef double* xmin = self.xmin.data

        cdef int c_x, c_y, c_z

        cdef int i, n
        for i from 0<=i<curr_num_particles:
            find_cell_id_raw(
                    x_ptr[i] - xmin[0],
                    y_ptr[i] - xmin[1],
                    z_ptr[i] - xmin[2],
                    self.h_sub,
                    &c_x, &c_y, &c_z
                    )
            current_pids[i] = i
            current_keys[i] = get_key(c_x, c_y, c_z)

        cdef CompareSortWrapper sort_wrapper = \
                CompareSortWrapper(current_pids, current_keys, curr_num_particles)

        sort_wrapper.compare_sort()

        cdef uint32_t found_cid = UINT_MAX
        cdef int num_particles
        cdef uint64_t* found_ptr = NULL

        cdef uint32_t pid

        cdef uint32_t* iter_cids
        cdef uint64_t* iter_keys

        cdef uint32_t* sorted_cids = <uint32_t*> malloc(
                curr_num_particles*sizeof(uint32_t))

        for j from 0<=j<pa_index:
            iter_cids = self.cids[j]
            iter_keys = self.keys[j]
            iter_pids = self.pids[j]
            num_particles = (<NNPSParticleArrayWrapper> \
                    self.pa_wrappers[j]).get_number_of_particles()
            found_ptr = self.find(current_keys[0], iter_keys, num_particles)

            if found_ptr != NULL:
                pid = iter_pids[found_ptr - iter_keys]
                found_cid = iter_cids[pid]
                break

        if found_cid == UINT_MAX:
            found_cid = curr_cid
            curr_cid += 1

        sorted_cids[0] = found_cid

        for i from 0<i<curr_num_particles:
            if(current_keys[i] != current_keys[i-1]):
                found_ptr = NULL
                found_cid = UINT_MAX

                for j from 0<=j<pa_index:
                    iter_cids = self.cids[j]
                    iter_keys = self.keys[j]
                    iter_pids = self.pids[j]
                    num_particles = (<NNPSParticleArrayWrapper> \
                            self.pa_wrappers[j]).get_number_of_particles()
                    found_ptr = self.find(current_keys[i], iter_keys, num_particles)

                    if found_ptr != NULL:
                        pid = iter_pids[found_ptr - iter_keys]
                        found_cid = iter_cids[pid]
                        break

                if found_cid == UINT_MAX:
                    found_cid = curr_cid
                    curr_cid += 1

            sorted_cids[i] = found_cid

        for i from 0<=i<curr_num_particles:
            pid = current_pids[i]
            current_cids[pid] = sorted_cids[i]
        free(sorted_cids)

        return curr_cid

    cpdef np.ndarray get_nbr_boxes(self, pa_index, cid):
        cdef IntArray nbr_boxes_arr = IntArray()
        cdef int* current_nbr_boxes = self.nbr_boxes[pa_index]
        nbr_boxes_arr.c_set_view(&current_nbr_boxes[self.mask_len*cid],
                self.mask_len)
        return nbr_boxes_arr.get_npy_array()

    cpdef np.ndarray get_keys(self, pa_index):
        cdef uint64_t* current_keys = self.keys[pa_index]
        cdef int num_particles = (<NNPSParticleArrayWrapper> \
                self.pa_wrappers[pa_index]).get_number_of_particles()
        keys = np.empty(num_particles)
        for i from 0<=i<num_particles:
            keys[i] = current_keys[i]
        return keys

    cpdef np.ndarray get_cids(self, pa_index):
        cdef UIntArray cids = UIntArray()
        cdef int num_particles = (<NNPSParticleArrayWrapper> \
                self.pa_wrappers[pa_index]).get_number_of_particles()
        cids.c_set_view(self.cids[pa_index], num_particles)
        return cids.get_npy_array()

    cpdef np.ndarray get_pids(self, pa_index):
        cdef UIntArray pids = UIntArray()
        cdef int num_particles = (<NNPSParticleArrayWrapper> \
                self.pa_wrappers[pa_index]).get_number_of_particles()
        pids.c_set_view(self.pids[pa_index], num_particles)
        return pids.get_npy_array()

    cdef void _fill_nbr_boxes(self):
        cdef int i, j, k
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int num_particles

        cdef int* current_nbr_boxes
        cdef uint32_t* current_cids
        cdef uint32_t* current_pids
        cdef uint64_t* current_keys
        cdef int found_idx

        cdef double* x_ptr
        cdef double* y_ptr
        cdef double* z_ptr

        cdef int c_x, c_y, c_z
        cdef int num_boxes

        cdef int n = 0
        cdef uint32_t cid, pid

        cdef uint64_t* found_ptrs[27]
        cdef uint64_t key

        for i from 0<=i<self.narrays:
            n = 0
            pa_wrapper = self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()
            x_ptr = pa_wrapper.x.data
            y_ptr = pa_wrapper.y.data
            z_ptr = pa_wrapper.z.data
            xmin = self.xmin.data

            current_keys = self.keys[i]
            current_cids = self.cids[i]
            current_pids = self.pids[i]
            current_nbr_boxes = self.nbr_boxes[i]
            current_lengths = self.lengths[i]

            # init self.nbr_boxes to -1
            for j from 0<=j<self.mask_len * self.max_cid:
                current_nbr_boxes[j] = -1

            for j from 0<=j<self.max_cid:
                current_lengths[j] = 1

            pid = current_pids[0]
            cid = current_cids[pid]

            find_cell_id_raw(
                x_ptr[pid] - xmin[0],
                y_ptr[pid] - xmin[1],
                z_ptr[pid] - xmin[2],
                self.h_sub,
                &c_x, &c_y, &c_z
                )

            num_boxes = self._neighbor_boxes(c_x, c_y, c_z, current_keys,
                    num_particles, found_ptrs)

            for k from 0<=k<num_boxes:
                found_ptr = found_ptrs[k]
                found_idx = found_ptr - current_keys
                current_nbr_boxes[self.mask_len*cid + n] = found_idx
                n += 1

            # nbrs of all cids in particle array i
            for j from 0<j<num_particles:
                key = current_keys[j]
                pid = current_pids[j]
                cid = current_cids[pid]
                n = 0

                if key != current_keys[j-1]:
                    find_cell_id_raw(
                        x_ptr[pid] - xmin[0],
                        y_ptr[pid] - xmin[1],
                        z_ptr[pid] - xmin[2],
                        self.h_sub,
                        &c_x, &c_y, &c_z
                        )

                    num_boxes = self._neighbor_boxes(c_x, c_y, c_z, current_keys,
                            num_particles, found_ptrs)

                    for k from 0<=k<num_boxes:
                        found_ptr = found_ptrs[k]
                        found_idx = found_ptr - current_keys
                        current_nbr_boxes[self.mask_len*cid + n] = found_idx
                        n += 1
                else:
                    current_lengths[cid] += 1

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            uint64_t* current_keys, int num_particles,
            uint64_t** found_ptrs) nogil:
        cdef int length = 0
        cdef int p, q, r
        cdef uint64_t key
        cdef uint64_t* found_ptr
        for p from -1<=p<2:
            for q from -1<=q<2:
                for r from -1<=r<2:
                    if i+r>=0 and j+q>=0 and k+p>=0:
                        key = get_key(i+r, j+q, k+p)
                        found_ptr = self.find(key, current_keys,
                                num_particles)
                        if found_ptr != NULL:
                            found_ptrs[length] = found_ptr
                            length += 1
        return length

    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper

        cdef int i, num_particles

        cdef double* xmax = self.xmax.data
        cdef double* xmin = self.xmin.data

        cdef uint32_t max_cid = 0

        cdef uint32_t* current_pids
        cdef uint64_t* current_keys
        cdef uint32_t* current_cids

        self.h_sub = self.cell_size / self.H

        for i from 0<=i<self.narrays:
            if self.pids[i] != NULL:
                free(self.pids[i])
            if self.keys[i] != NULL:
                free(self.keys[i])
            if self.cids[i] != NULL:
                free(self.cids[i])

            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))
            self.keys[i] = <uint64_t*> malloc(num_particles*sizeof(uint64_t))
            self.cids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))

            current_pids = self.pids[i]
            current_cids = self.cids[i]
            current_keys = self.keys[i]

            max_cid = self.fill_array(pa_wrapper, i, current_pids,
                    current_keys, current_cids, max_cid)

        for i from 0<=i<self.narrays:
            if self.nbr_boxes[i] != NULL:
                free(self.nbr_boxes[i])
            if self.lengths[i] != NULL:
                free(self.lengths[i])
            self.nbr_boxes[i] = <int*> malloc(self.mask_len * max_cid * sizeof(int))
            self.lengths[i] = <int*> malloc(max_cid * sizeof(int))

        self.max_cid = max_cid # this is max_cid + 1

        self._fill_nbr_boxes()

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        pass

cdef class ExtendedZOrderNNPS(ZOrderNNPS):

    """Find nearest neighbors using Z-Order space filling curve"""

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int H=3,
            bint asymmetric=False):
        ZOrderNNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids, H=H
        )

        self.asymmetric = asymmetric

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int H=3,
            bint asymmetric=False):

        narrays = len(particles)

        self.hmax = <double**> malloc(narrays*sizeof(double*))
        self.current_hmax = NULL

        cdef int i
        for i from 0<=i<narrays:
            self.hmax[i] = NULL

    def __dealloc__(self):
        cdef int i
        for i from 0<=i<self.narrays:
            if self.hmax[i] != NULL:
                free(self.hmax[i])
        free(self.hmax)

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

    cdef int _neighbor_boxes_func(self, int i, int j, int k,
            uint64_t* current_keys, uint32_t* current_cids,
            double* current_hmax, int num_particles,
            uint64_t** found_ptrs, double h) nogil:
        if self.asymmetric:
            return self._neighbor_boxes_asym(i, j, k, current_keys, current_cids,
                    current_hmax, num_particles, found_ptrs, h)
        else:
            return self._neighbor_boxes_sym(i, j, k, current_keys, current_cids,
                    current_hmax, num_particles, found_ptrs, h)


    cdef int _neighbor_boxes_asym(self, int i, int j, int k,
            uint64_t* current_keys, uint32_t* current_cids,
            double* current_hmax, int num_particles,
            uint64_t** found_ptrs, double h) nogil:
        cdef int length = 0
        cdef int p

        cdef uint64_t key
        cdef uint64_t* found_ptr

        cdef int x_temp, y_temp, z_temp

        cdef int* x_mask = <int*> malloc(self.mask_len*sizeof(int))
        cdef int* y_mask = <int*> malloc(self.mask_len*sizeof(int))
        cdef int* z_mask = <int*> malloc(self.mask_len*sizeof(int))

        cdef int mask_len = self._h_mask_exact(x_mask, y_mask, z_mask)

        for p from 0<=p<mask_len:
            x_temp = i + x_mask[p]
            y_temp = j + y_mask[p]
            z_temp = k + z_mask[p]

            if x_temp >= 0 and y_temp >= 0 and z_temp >= 0:
                key = get_key(x_temp, y_temp, z_temp)
                found_ptr = self.find(key, current_keys, num_particles)

                if found_ptr == NULL:
                    continue

                found_ptrs[length] = found_ptr
                length += 1

        free(x_mask)
        free(y_mask)
        free(z_mask)

        return length


    @cython.cdivision(True)
    cdef int _neighbor_boxes_sym(self, int i, int j, int k,
            uint64_t* current_keys, uint32_t* current_cids,
            double* current_hmax, int num_particles,
            uint64_t** found_ptrs, double h) nogil:
        cdef int length = 0
        cdef int p

        cdef uint64_t key
        cdef uint64_t* found_ptr

        cdef int x_temp, y_temp, z_temp

        cdef double h_local
        cdef int H

        cdef int* x_mask = <int*> malloc(self.mask_len*sizeof(int))
        cdef int* y_mask = <int*> malloc(self.mask_len*sizeof(int))
        cdef int* z_mask = <int*> malloc(self.mask_len*sizeof(int))

        cdef int mask_len = self._h_mask_exact(x_mask, y_mask, z_mask)

        for p from 0<=p<mask_len:
            x_temp = i + x_mask[p]
            y_temp = j + y_mask[p]
            z_temp = k + z_mask[p]

            if x_temp >= 0 and y_temp >= 0 and z_temp >= 0:
                key = get_key(x_temp, y_temp, z_temp)
                found_ptr = self.find(key, current_keys, num_particles)

                if found_ptr == NULL:
                    continue

                cid = current_cids[found_ptr - current_keys]

                h_local = self.radius_scale*fmax(current_hmax[cid], h)
                H = <int> ceil(h_local/self.h_sub)

                if abs(x_mask[p]) <= H and abs(y_mask[p]) <= H and abs(z_mask[p]) <= H:
                    found_ptrs[length] = found_ptr
                    length += 1

        free(x_mask)
        free(y_mask)
        free(z_mask)

        return length


    cdef void _fill_nbr_boxes(self):
        cdef int i, j, k
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int num_particles

        cdef int* current_nbr_boxes
        cdef uint32_t* current_cids
        cdef uint32_t* current_pids
        cdef uint64_t* current_keys
        cdef double* current_hmax
        cdef int found_idx

        cdef double* x_ptr
        cdef double* y_ptr
        cdef double* z_ptr
        cdef double* h_ptr

        cdef int c_x, c_y, c_z
        cdef int num_boxes

        cdef int n = 0
        cdef uint32_t cid, pid

        cdef uint64_t** found_ptrs = <uint64_t**> malloc(self.mask_len * \
                sizeof(uint64_t*))
        cdef uint64_t key

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
            current_cids = self.cids[i]
            current_pids = self.pids[i]
            current_nbr_boxes = self.nbr_boxes[i]
            current_lengths = self.lengths[i]
            current_hmax = self.hmax[i]

            # init self.nbr_boxes to -1
            for j from 0<=j<self.mask_len * self.max_cid:
                current_nbr_boxes[j] = -1

            for j from 0<=j<self.max_cid:
                current_lengths[j] = 1
                current_hmax[j] = 0

            pid = current_pids[0]
            cid = current_cids[pid]

            current_hmax[cid] = h_ptr[pid]

            for j from 0<j<num_particles:
                key = current_keys[j]
                pid = current_pids[j]
                cid = current_cids[pid]

                if key != current_keys[j-1]:
                    current_hmax[cid] = h_ptr[pid]
                else:
                    current_hmax[cid] = fmax(current_hmax[cid], h_ptr[pid])

            pid = current_pids[0]
            cid = current_cids[pid]

            find_cell_id_raw(
                x_ptr[pid] - xmin[0],
                y_ptr[pid] - xmin[1],
                z_ptr[pid] - xmin[2],
                self.h_sub,
                &c_x, &c_y, &c_z
                )

            num_boxes = self._neighbor_boxes_func(c_x, c_y, c_z, current_keys,
                    current_cids, current_hmax, num_particles, found_ptrs, h_ptr[pid])

            for k from 0<=k<num_boxes:
                found_ptr = found_ptrs[k]
                found_idx = found_ptr - current_keys
                current_nbr_boxes[self.mask_len*cid + n] = found_idx
                n += 1

            # nbrs of all cids in particle array i
            for j from 0<j<num_particles:
                key = current_keys[j]
                pid = current_pids[j]
                cid = current_cids[pid]
                n = 0

                if key != current_keys[j-1]:
                    find_cell_id_raw(
                        x_ptr[pid] - xmin[0],
                        y_ptr[pid] - xmin[1],
                        z_ptr[pid] - xmin[2],
                        self.h_sub,
                        &c_x, &c_y, &c_z
                        )

                    num_boxes = self._neighbor_boxes_func(c_x, c_y, c_z, current_keys,
                            current_cids, current_hmax, num_particles, found_ptrs, h_ptr[pid])

                    for k from 0<=k<num_boxes:
                        found_ptr = found_ptrs[k]
                        found_idx = found_ptr - current_keys
                        current_nbr_boxes[self.mask_len*cid + n] = found_idx
                        n += 1
                else:
                    current_lengths[cid] += 1

        free(found_ptrs)

    @cython.cdivision(True)
    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper

        cdef int i, num_particles

        cdef double* xmax = self.xmax.data
        cdef double* xmin = self.xmin.data

        cdef uint32_t max_cid = 0

        cdef uint32_t* current_pids
        cdef uint64_t* current_keys
        cdef uint32_t* current_cids

        self.h_sub = self.cell_size / self.H

        for i from 0<=i<self.narrays:
            if self.pids[i] != NULL:
                free(self.pids[i])
            if self.keys[i] != NULL:
                free(self.keys[i])
            if self.cids[i] != NULL:
                free(self.cids[i])

            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))
            self.keys[i] = <uint64_t*> malloc(num_particles*sizeof(uint64_t))
            self.cids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))

            current_pids = self.pids[i]
            current_cids = self.cids[i]
            current_keys = self.keys[i]

            max_cid = self.fill_array(pa_wrapper, i, current_pids,
                    current_keys, current_cids, max_cid)

        for i from 0<=i<self.narrays:
            if self.nbr_boxes[i] != NULL:
                free(self.nbr_boxes[i])
            if self.lengths[i] != NULL:
                free(self.lengths[i])
            if self.hmax[i] != NULL:
                free(self.hmax[i])
            self.nbr_boxes[i] = <int*> malloc(self.mask_len * max_cid * sizeof(int))
            self.lengths[i] = <int*> malloc(max_cid*sizeof(int))
            self.hmax[i] = <double*> malloc(max_cid*sizeof(double))

        self.max_cid = max_cid # this is max_cid + 1

        self._fill_nbr_boxes()

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
        self.current_cids_dst = self.cids[dst_index]
        self.current_cids_src = self.cids[src_index]
        self.current_nbr_boxes = self.nbr_boxes[src_index]
        self.current_lengths = self.lengths[src_index]
        self.current_hmax = self.hmax[src_index]

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]


