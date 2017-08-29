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

# BSEARCH NEEDS TO FIND FIRST OCCURENCE

cdef inline int cmp_func(const void* a, const void* b) nogil:
    #return (<uint64_t*>a)[0] - (<uint64_t*>b)[0]
    if (<uint64_t*>a)[0] < (<uint64_t*>b)[0]: return -1
    if (<uint64_t*>a)[0] == (<uint64_t*>b)[0]: return 0
    if (<uint64_t*>a)[0] > (<uint64_t*>b)[0]: return 1

cdef class ZOrderNNPS(NNPS):

    """Find nearest neighbors using Z-Order space filling curve"""

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.radius_scale2 = radius_scale*radius_scale
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles

        for i from 0<=i<self.narrays:
            self.pids[i] = NULL
            self.keys[i] = NULL
            self.cids[i] = NULL
            self.nbr_boxes[i] = NULL

        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False):
        cdef int narrays = len(particles)

        self.pids = <uint32_t**> malloc(narrays*sizeof(uint32_t*))
        self.keys = <uint64_t**> malloc(narrays*sizeof(uint64_t*))
        self.cids = <uint32_t**> malloc(narrays*sizeof(uint32_t*))
        self.nbr_boxes = <int**> malloc(narrays*sizeof(int*))

        self.current_pids = NULL
        self.current_cids = NULL
        self.current_keys = NULL
        self.current_nbr_boxes = NULL

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
        free(self.pids)
        free(self.keys)
        free(self.cids)
        free(self.nbr_boxes)

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
        self.current_cids = self.cids[dst_index]
        self.current_nbr_boxes = self.nbr_boxes[src_index]

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

        cdef uint32_t cid = self.current_cids[d_idx]

        cdef unsigned int* s_gid = self.src.gid.data
        cdef int orig_length = nbrs.length

        cdef int c_x, c_y, c_z
        cdef double* xmin = self.xmin.data
        cdef int i, j

        find_cell_id_raw(
                x - xmin[0],
                y - xmin[1],
                z - xmin[2],
                self.cell_size,
                &c_x, &c_y, &c_z
                )

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2*h*h
        cdef double hj2 = 0

        cdef int start_idx, curr_idx
        cdef uint32_t n, idx
        for i from 0<=i<27:
            if self.current_nbr_boxes[27*cid + i] == -1:
                continue

            start_idx = self.current_nbr_boxes[27*cid + i]
            curr_idx = 0

            while start_idx + curr_idx < num_particles and \
                    self.current_keys[start_idx + curr_idx] \
                    == self.current_keys[start_idx]:
                idx = self.current_pids[start_idx + curr_idx]

                hj2 = self.radius_scale2*src_h_ptr[idx]*src_h_ptr[idx]

                xij2 = norm2(
                    src_x_ptr[idx] - x,
                    src_y_ptr[idx] - y,
                    src_z_ptr[idx] - z
                    )

                if (xij2 < hi2) or (xij2 < hj2):
                    nbrs.c_append(idx)

                curr_idx += 1

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
                    self.cell_size,
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

        cdef uint32_t* iter_cids
        cdef uint64_t* iter_keys

        for j from 0<=j<pa_index:
            iter_cids = self.cids[j]
            iter_keys = self.keys[j]
            num_particles = (<NNPSParticleArrayWrapper> \
                    self.pa_wrappers[j]).get_number_of_particles()
            found_ptr = self.find(current_keys[0], iter_keys, num_particles)

            if found_ptr != NULL:
                found_cid = iter_cids[found_ptr - iter_keys]
                break

        if found_cid == UINT_MAX:
            found_cid = curr_cid
            curr_cid += 1

        current_cids[0] = found_cid

        for i from 0<i<curr_num_particles:
            if(current_keys[i] != current_keys[i-1]):
                found_ptr = NULL
                found_cid = UINT_MAX

                for j from 0<=j<pa_index:
                    iter_cids = self.cids[j]
                    iter_keys = self.keys[j]
                    num_particles = (<NNPSParticleArrayWrapper> \
                            self.pa_wrappers[j]).get_number_of_particles()
                    found_ptr = self.find(current_keys[j], iter_keys, num_particles)

                    if found_ptr != NULL:
                        found_cid = iter_cids[found_ptr - iter_keys]
                        break

                if found_cid == UINT_MAX:
                    found_cid = curr_cid
                    curr_cid += 1

            current_cids[i] = found_cid

        return curr_cid

    cpdef IntArray get_nbr_boxes(self, pa_index, cid):
        cdef IntArray nbr_boxes_arr = IntArray()
        cdef int* current_nbr_boxes = self.nbr_boxes[pa_index]
        nbr_boxes_arr.c_set_view(&current_nbr_boxes[27*cid],
                27)
        return nbr_boxes_arr

    cpdef get_key(self, pa_index, idx):
        cdef uint64_t* current_keys = self.keys[pa_index]
        return current_keys[idx]

    cpdef UIntArray get_cids(self, pa_index):
        cdef UIntArray cids = UIntArray()
        cdef int num_particles = (<NNPSParticleArrayWrapper> \
                self.pa_wrappers[pa_index]).get_number_of_particles()
        cids.c_set_view(self.cids[pa_index], num_particles)
        return cids

    cpdef UIntArray get_pids(self, pa_index):
        cdef UIntArray pids = UIntArray()
        cdef int num_particles = (<NNPSParticleArrayWrapper> \
                self.pa_wrappers[pa_index]).get_number_of_particles()
        pids.c_set_view(self.pids[pa_index], num_particles)
        return pids

    cdef void _fill_nbr_boxes(self):
        cdef int i, j, k
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int num_particles

        cdef int* current_nbr_boxes
        cdef int found_idx

        cdef double* x_ptr
        cdef double* y_ptr
        cdef double* z_ptr

        cdef int c_x, c_y, c_z
        cdef uint64_t nbr_keys[27]
        cdef int num_boxes

        cdef int n = 0
        cdef uint32_t cid

        cdef uint64_t* found_ptr

        cdef IntArray dummy = IntArray()

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
            current_nbr_boxes = self.nbr_boxes[i]

            # init self.nbr_boxes to -1
            for j from 0<=j<27*self.max_cid:
                current_nbr_boxes[j] = -1

            cid = current_cids[0]

            find_cell_id_raw(
                x_ptr[0] - xmin[0],
                y_ptr[0] - xmin[1],
                z_ptr[0] - xmin[2],
                self.cell_size,
                &c_x, &c_y, &c_z
                )

            num_boxes = self._neighbor_boxes(c_x, c_y, c_z, nbr_keys)

            for k from 0<=k<num_boxes:
                found_ptr = self.find(nbr_keys[k], current_keys, num_particles)

                if found_ptr == NULL:
                    continue

                found_idx = found_ptr - current_keys
                current_nbr_boxes[27*cid + n] = found_idx
                n += 1

            # nbrs of all cids in particle array i
            for j from 0<j<num_particles:
                cid = current_cids[j]
                n = 0

                if cid != current_cids[j-1]:
                    find_cell_id_raw(
                        x_ptr[j] - xmin[0],
                        y_ptr[j] - xmin[1],
                        z_ptr[j] - xmin[2],
                        self.cell_size,
                        &c_x, &c_y, &c_z
                        )

                    num_boxes = self._neighbor_boxes(c_x, c_y, c_z, nbr_keys)

                    for k from 0<=k<num_boxes:
                        found_ptr = self.find(nbr_keys[k], current_keys, num_particles)

                        if found_ptr == NULL:
                            continue

                        found_idx = found_ptr - current_keys
                        current_nbr_boxes[27*cid + n] = found_idx
                        if cid == 4:
                            print found_idx
                        n += 1


    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            uint64_t* nbr_keys) nogil:
        cdef int length = 0
        cdef int p, q, r
        for p from -1<=p<2:
            for q from -1<=q<2:
                for r from -1<=r<2:
                    if i+r>=0 and j+q>=0 and k+p>=0:
                        nbr_keys[length] = get_key(<uint32_t>(i+r),
                                <uint32_t>(j+q), <uint32_t>(k+p))
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
            self.nbr_boxes[i] = <int*> malloc(27*max_cid*sizeof(int))

        self.max_cid = max_cid # this is max_cid + 1

        self._fill_nbr_boxes()

        self.current_pids = self.pids[self.src_index]

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        pass

