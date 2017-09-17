from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair

from pysph.base.gpu_nnps_base cimport *

ctypedef unsigned int u_int
ctypedef map[u_int, pair[u_int, u_int]] key_to_idx_t
ctypedef vector[u_int] u_int_vector_t

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

cdef class StratifiedSFCGPUNNPS(GPUNNPS):
    cdef NNPSParticleArrayWrapper src, dst # Current source and destination.

    cdef public list pids
    cdef public list pid_keys
    cdef public list start_idx_levels
    cdef public list num_particles_levels
    cdef public int max_num_bits
    cdef int num_levels
    cdef double interval_size
    cdef double eps

    cdef object helper

    cdef bint _sorted

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices)

    cpdef _bin(self, int pa_index)

    cpdef _refresh(self)

    cdef void find_neighbor_lengths(self, nbr_lengths)

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices)
