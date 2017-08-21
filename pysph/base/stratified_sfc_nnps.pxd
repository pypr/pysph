from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair

from nnps_base cimport *

cdef extern from 'math.h':
    int abs(int) nogil
    double ceil(double) nogil
    double floor(double) nogil
    double fabs(double) nogil
    double fmax(double, double) nogil
    double fmin(double, double) nogil

cdef extern from 'math.h':
    double log(double) nogil
    double log2(double) nogil

cdef extern from "z_order.h":
    ctypedef unsigned int uint32_t
    ctypedef unsigned long long uint64_t
    inline uint64_t get_key(uint64_t i, uint64_t j, uint64_t k) nogil

    cdef cppclass CompareSortWrapper:
        CompareSortWrapper() nogil except +
        CompareSortWrapper(uint32_t* current_pids, uint64_t* current_keys,
                int length) nogil except +
        inline void compare_sort() nogil

ctypedef map[uint64_t, pair[uint32_t, uint32_t]] key_to_idx_t

cdef class StratifiedSFCNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef double radius_scale2

    cdef public int num_levels
    cdef int max_num_bits

    cdef double interval_size

    cdef uint32_t** pids
    cdef uint32_t* current_pids

    cdef uint64_t** keys
    cdef uint64_t* current_keys

    cdef key_to_idx_t** pid_indices
    cdef key_to_idx_t* current_indices

    cdef double** cell_sizes
    cdef double* current_cells

    cdef NNPSParticleArrayWrapper dst, src

    ##########################################################################
    # Member functions
    ##########################################################################

    cpdef set_context(self, int src_index, int dst_index)

    cpdef double get_binning_size(self, int interval)

    cpdef int get_number_of_particles(self, int pa_index, int level)

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z, int H) nogil

    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper,
            int pa_index, UIntArray indices, uint32_t* current_pids,
            uint64_t* current_keys, key_to_idx_t* current_indices,
            double* current_cells)

    cdef inline int _get_level(self, double h) nogil

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)


