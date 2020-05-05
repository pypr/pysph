# cython: language_level=3, embedsignature=True
# distutils: language=c++
from libcpp.map cimport map
from libcpp.pair cimport pair

from .nnps_base cimport *

cdef extern from "math.h":
    double log2(double) nogil

cdef extern from "z_order.h":
    ctypedef unsigned long long uint64_t
    ctypedef unsigned int uint32_t
    inline uint64_t get_key(uint64_t i, uint64_t j, uint64_t k) nogil

    cdef cppclass CompareSortWrapper:
        CompareSortWrapper() nogil except +
        CompareSortWrapper(uint32_t* current_pids, uint64_t* current_keys,
                int length) nogil except +
        inline void compare_sort() nogil

ctypedef map[uint64_t, pair[uint32_t, uint32_t]] key_to_idx_t

cdef class ZOrderNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef uint32_t** pids
    cdef uint32_t* current_pids

    cdef uint64_t** keys
    cdef uint64_t* current_keys

    cdef int** key_to_idx
    cdef int* current_key_to_idx

    cdef uint32_t** cids
    cdef uint32_t* current_cids_dst
    cdef uint32_t* current_cids_src

    cdef int** nbr_boxes
    cdef int* current_nbr_boxes

    cdef int** lengths
    cdef int* current_lengths

    cdef public uint32_t max_cid

    cdef double radius_scale2
    cdef NNPSParticleArrayWrapper dst, src

    cdef int H
    cdef double h_sub

    cdef int mask_len
    cdef uint64_t max_key

    cdef bint asymmetric

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* current_key_to_idx, int num_particles,
            int* found_indices) nogil

    cpdef set_context(self, int src_index, int dst_index)

    cdef inline int get_idx(self, uint64_t key, int* key_to_idx) nogil

    cpdef np.ndarray get_nbr_boxes(self, pa_index, cid)

    cpdef np.ndarray get_pids(self, pa_index)

    cpdef np.ndarray get_cids(self, pa_index)

    cpdef np.ndarray get_keys(self, pa_index)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc)

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices)

    cdef int fill_array(self, NNPSParticleArrayWrapper pa_wrapper, int pa_index,
            uint32_t* current_pids, uint64_t* current_keys,
            uint32_t* current_cids, uint32_t curr_cid)

    cdef void _fill_nbr_boxes(self)

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)

cdef class ExtendedZOrderNNPS(ZOrderNNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef double** hmax
    cdef double* current_hmax

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef inline int _h_mask_exact(self, int* x, int* y, int* z) nogil

    cdef int _neighbor_boxes_func(self, int i, int j, int k,
            int* current_key_to_idx, uint32_t* current_cids,
            double* current_hmax, int num_particles,
            int* found_indices, double h)

    cdef int _neighbor_boxes_asym(self, int i, int j, int k,
            int* current_key_to_idx, uint32_t* current_cids,
            double* current_hmax, int num_particles,
            int* found_indices, double h) nogil

    cdef int _neighbor_boxes_sym(self, int i, int j, int k,
            int* current_key_to_idx, uint32_t* current_cids,
            double* current_hmax, int num_particles,
            int* found_indices, double h) nogil

    cdef void _fill_nbr_boxes(self)

    cpdef _refresh(self)

    cpdef set_context(self, int src_index, int dst_index)
