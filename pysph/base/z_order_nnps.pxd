# cython: embedsignature=True
from libcpp.map cimport map
from libcpp.pair cimport pair

from nnps_base cimport *

ctypedef unsigned int u_int
ctypedef map[u_int, pair[u_int, u_int]] key_to_idx_t

cdef extern from "math.h":
    double log2(double) nogil

cdef extern from "z_order.h":
    ctypedef unsigned long long uint64_t
    inline uint64_t get_key(uint64_t i, uint64_t j, uint64_t k) nogil

    cdef cppclass CompareSortWrapper:
        CompareSortWrapper() nogil except +
        CompareSortWrapper(double* x_ptr, double* y_ptr, double* z_ptr,
                double* xmin, double cell_size, u_int* current_pids,
                int length) nogil except +
        inline void compare_sort() nogil

cdef class ZOrderNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef u_int** pids
    cdef u_int* current_pids

    cdef key_to_idx_t** pid_indices
    cdef key_to_idx_t* current_indices

    cdef double radius_scale2
    cdef NNPSParticleArrayWrapper dst, src

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z) nogil

    cpdef set_context(self, int src_index, int dst_index)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc)

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices)

    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper, int pa_index,
            UIntArray indices, u_int* current_keys, key_to_idx_t* current_indices)

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)


