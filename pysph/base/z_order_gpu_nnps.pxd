# cython: embedsignature=True
from libcpp.map cimport map
from libcpp.pair cimport pair

from pysph.base.gpu_nnps_base cimport *

ctypedef unsigned int u_int
ctypedef map[u_int, pair[u_int, u_int]] key_to_idx_t

cdef extern from "math.h":
    double log2(double) nogil

cdef class ZOrderGPUNNPS(GPUNNPS):
    cdef NNPSParticleArrayWrapper src, dst # Current source and destination.

    cdef list pids
    cdef list pid_keys
    cdef list cids
    cdef list cid_to_idx
    cdef list max_cid
    cdef object dst_to_src
    cdef object overflow_cid_to_idx
    cdef object curr_cid
    cdef object max_cid_src

    cdef object helper
    cdef object radix_sort
    cdef object make_vec

    cdef bint _sorted
    cdef bint dst_src

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices)

    cpdef _bin(self, int pa_index)

    cpdef _refresh(self)

    cdef void find_neighbor_lengths(self, nbr_lengths)

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices)
