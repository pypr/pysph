# cython: language_level=3, embedsignature=True
# distutils: language=c++
from pysph.base.gpu_nnps_base cimport *


cdef class OctreeGPUNNPS(GPUNNPS):
    cdef NNPSParticleArrayWrapper src, dst # Current source and destination.

    cdef public list pids
    cdef public list pid_keys
    cdef public list cids
    cdef public list cid_to_idx
    cdef public list max_cid
    cdef public object dst_to_src
    cdef object overflow_cid_to_idx
    cdef object curr_cid
    cdef object max_cid_src

    cdef object helper
    cdef object radix_sort
    cdef object make_vec

    cdef public bint allow_sort
    cdef bint dst_src

    cdef public object neighbor_cid_counts
    cdef public object neighbor_cids
    cdef public list octrees
    cdef public bint use_elementwise
    cdef public bint use_partitions
    cdef public object leaf_size

    cpdef _bin(self, int pa_index)

    cpdef _refresh(self)

    cdef void find_neighbor_lengths(self, nbr_lengths)
    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices)

    cpdef get_kernel_args(self, c_type)
