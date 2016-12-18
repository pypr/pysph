# cython: embedsignature=True
from libcpp.map cimport map
from libcpp.pair cimport pair

from nnps_base cimport *

ctypedef unsigned int u_int
ctypedef map[u_int, pair[u_int, u_int]] key_to_idx_t

cdef extern from 'math.h':
    double log(double) nogil
    double log2(double) nogil

cdef class CellIndexingNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef u_int** keys
    cdef u_int* current_keys

    cdef key_to_idx_t** key_indices
    cdef key_to_idx_t* current_indices

    cdef u_int* I
    cdef u_int J
    cdef u_int K

    cdef double radius_scale2
    cdef NNPSParticleArrayWrapper dst, src

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef inline u_int _get_key(self, u_int n, u_int i, u_int j,
            u_int k, int pa_index) nogil

    cdef inline int _get_id(self, u_int key, int pa_index) nogil

    cdef inline int _get_x(self, u_int key, int pa_index) nogil

    cdef inline int _get_y(self, u_int key, int pa_index) nogil

    cdef inline int _get_z(self, u_int key, int pa_index) nogil

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z) nogil

    cpdef set_context(self, int src_index, int dst_index)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc)

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices)

    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper, int pa_index,
            UIntArray indices, u_int* current_keys, key_to_idx_t* current_indices) nogil

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)



