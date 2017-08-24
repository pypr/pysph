#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stdlib cimport bsearch

from cython.operator cimport dereference as deref, preincrement as inc

# Cython for compiler directives
cimport cython

import numpy as np
cimport numpy as np

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)
    void sort[Iter](Iter first, Iter last)

#############################################################################

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
            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))
            self.keys[i] = <uint64_t*> malloc(num_particles*sizeof(uint64_t))
            self.cids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))

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

        self.current_pids = NULL
        self.current_cids = NULL
        self.current_keys = NULL

    def __dealloc__(self):
        cdef int i
        for i from 0<=i<self.narrays:
            free(self.pids[i])
            free(self.keys[i])
            del self.key_maps[i]
        free(self.pids)
        free(self.keys)
        free(self.key_maps)

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
        self.current_map = self.key_maps[src_index]

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

        cdef uint32_t cid = self.current_cids[d_idx]
        cdef int* nbr_indices = &self.current_nbr_boxes[27*cid]

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
            if nbr_indices[i] == -1:
                continue
            start_idx = nbr_indices[i]
            curr_idx = 0

            while self.current_cids[start_idx + curr_idx] \
                    == self.current_cids[start_idx]:
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
            UIntArray indices, uint32_t* current_pids, uint64_t* current_keys,
            uint32_t curr_cid):

        cdef double* x_ptr = pa_wrapper.x.data
        cdef double* y_ptr = pa_wrapper.y.data
        cdef double* z_ptr = pa_wrapper.z.data

        cdef double* xmin = self.xmin.data

        cdef int c_x, c_y, c_z

        cdef int i, n
        for i from 0<=i<indices.length:
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
                CompareSortWrapper(current_pids, current_keys, indices.length)

        sort_wrapper.compare_sort()

        cdef uint32_t prev_max_cid = 0
        if pa_index != 0:
            prev_max_cid = self.max_cid[pa_index - 1]

        cdef uint32_t found_cid = UINT_MAX
        cdef int num_particles
        cdef uint64_t* found_ptr = NULL

        cdef uint32_t* iter_cids
        cdef uint64_t* iter_keys

        for j from 0<=j<self.narrays:
            iter_cids = self.cids[j]
            num_particles = (NNPSParticleArrayWrapper> \
                    self.pa_wrappers[j]).get_number_of_particles()
            found_ptr = <uint64_t*> bsearch(&current_keys[0],
                    self.keys[j], num_particles, sizeof(uint64_t))

            if found_ptr != NULL:
                found_cid = iter_cids[idx]
                break

        if found_cid == UINT_MAX:
            found_cid = curr_cid
            curr_cid += 1

        current_cids[0] = found_cid

        for i from 0<i<indices.length:
            if(current_keys[i] != current_keys[i-1]):

                for j from 0<=j<pa_index:
                    iter_cids = self.cids[j]
                    iter_keys = self.keys[j]
                    num_particles = (NNPSParticleArrayWrapper> \
                            self.pa_wrappers[j]).get_number_of_particles()
                    found_ptr = <uint64_t*> bsearch(&current_keys[0],
                            iter_keys, num_particles, sizeof(uint64_t))

                    if found_ptr != NULL:
                        found_cid = iter_cids[found_ptr - iter_keys]
                        break

                if found_cid == UINT_MAX:
                    found_cid = curr_cid
                    curr_cid += 1

            current_cids[i] = found_cid

        return curr_cid


    cdef void _fill_nbr_boxes(self):
        cdef int i, j, k
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int num_particles

        cdef int* current_nbr_boxes
        cdef uint32_t max_cid = self.max_cids[pa_index]

        cdef double* x_ptr
        cdef double* y_ptr
        cdef double* z_ptr

        cdef int c_x, c_y, c_z
        cdef uint64_t nbr_keys[27]
        cdef int num_boxes

        cdef int n = 0
        cdef uint32_t cid

        cdef int* found_ptr
        
        for i from 0<=i<self.narrays:
            pa_wrapper = self.pa_wrappers[pa_index]
            num_particles = pa_wrapper.get_number_of_particles()
            x_ptr = pa_wrapper.x.data
            y_ptr = pa_wrapper.y.data
            z_ptr = pa_wrapper.z.data

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
                found_ptr = bsearch(&nbr_keys[k], current_keys,
                        num_particles, sizeof(uint64_t))
                if found_ptr == NULL:
                    continue
                    
                found_idx = found_ptr - current_keys
                current_nbr_boxes[27*cid + n] = found_idx
                n += 1

            # nbrs of all cids in particle array i
            for j from 0<=j<num_particles:
                cid = current_cids[j]

                if cid == current_cids[j-1]:
                    find_cell_id_raw(
                        x_ptr[j] - xmin[0],
                        y_ptr[j] - xmin[1],
                        z_ptr[j] - xmin[2],
                        self.cell_size,
                        &c_x, &c_y, &c_z
                        )

                    num_boxes = self._neighbor_boxes(c_x, c_y, c_z, nbr_keys)

                    for k from 0<=k<num_boxes:
                        found_ptr = bsearch(&nbr_keys[k], current_keys,
                                num_particles, sizeof(uint64_t))
                        if found_ptr == NULL:
                            continue
                            
                        found_idx = found_ptr - current_keys
                        current_nbr_boxes[27*cid + n] = found_idx
                        n += 1

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            uint64_t* nbr_keys) nogil:
        cdef int length = 0
        cdef int p, q, r
        for p from -1<=p<2:
            for q from -1<=q<2:
                for r from -1<=r<2:
                    if i+r>=0 and j+q>=0 and k+p>=0:
                        nbr_keys[length] = get_key(i+r, j+q, k+p)
                        length += 1
        return length

    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper

        cdef int i, num_particles

        cdef double* xmax = self.xmax.data
        cdef double* xmin = self.xmin.data

        cdef uint32_t max_cid = 0

        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int num_particles

        cdef uint32_t* current_pids
        cdef uint64_t* current_keys

        for i from 0<=i<self.narrays:
            free(self.pids[i])
            free(self.keys[i])
            free(self.cids[i])

            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))
            self.keys[i] = <uint64_t*> malloc(num_particles*sizeof(uint64_t))
            self.cids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))

            pa_wrapper = self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            current_pids = self.pids[i]
            current_keys = self.keys[i]

            max_cid = self.fill_array(pa_wrapper, i, indices, current_pids,
                    current_keys, max_cid)

        self.nbr_boxes = <int**> malloc(self.narrays*sizeof(int*))
        
        for i from 0<=i<self.narrays:
            self.nbr_boxes[i] = <int*> malloc(max_cid*sizeof(int))

            self._fill_nbr_boxes()

        self.max_cid = max_cid # this is max_cid + 1
        self.current_pids = self.pids[self.src_index]

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        pass

