# cython: embedsignature=True
from libcpp cimport map

ctypedef long long int LL_INT
ctypedef map[LL_INT, int] key_to_idx_t

cdef class CellIndexing(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef LL_INT** keys
    cdef LL_INT* current_keys

    cdef key_to_idx_t** key_indices
    cdef key_to_idx_t* current_indices

    cdef LL_INT* I
    cdef LL_INT* J
    cdef LL_INT* K

    cdef double radius_scale2
    cdef NNPSParticleArrayWrapper dst, src

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef inline LL_INT get_key(self, LL_INT n, LL_INT i, LL_INT j,
            LL_INT k, int pa_index) nogil

    cdef inline int _get_id(self, LL_INT key, int pa_index) nogil

    cdef inline int _get_x(self, LL_INT key, int pa_index) nogil

    cdef inline int _get_y(self, LL_INT key, int pa_index) nogil

    cdef inline int _get_z(self, LL_INT key, int pa_index) nogil

    cpdef set_context(self, int src_index, int dst_index)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc)

    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper, int pa_index,
            UIntArray indices, LL_INT* current_keys, key_to_idx_t* current_indices) nogil

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)



