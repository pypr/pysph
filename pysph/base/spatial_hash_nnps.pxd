# cython: language_level=3, embedsignature=True
# distutils: language=c++
from libcpp.vector cimport vector

from .nnps_base cimport *

#Imports for SpatialHashNNPS
cdef extern from "spatial_hash.h":
    cdef cppclass HashEntry:
        double h_max

        vector[unsigned int] *get_indices() nogil

    cdef cppclass HashTable:
        HashTable(long long int) nogil except +
        void add(int, int, int, int, double) nogil
        HashEntry* get(int, int, int) nogil

# NNPS using Spatial Hashing algorithm
cdef class SpatialHashNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef long long int table_size               # Size of hashtable
    cdef double radius_scale2

    cdef HashTable** hashtable
    cdef HashTable* current_hash

    cdef NNPSParticleArrayWrapper dst, src

    ##########################################################################
    # Member functions
    ##########################################################################

    cpdef set_context(self, int src_index, int dst_index)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cdef inline void _add_to_hashtable(self, int hash_id, unsigned int pid, double h,
            int i, int j, int k) nogil

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z) nogil

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)

# NNPS using Extended Spatial Hashing algorithm
cdef class ExtendedSpatialHashNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef long long int table_size               # Size of hashtable
    cdef double radius_scale2

    cdef HashTable** hashtable
    cdef HashTable* current_hash

    cdef NNPSParticleArrayWrapper dst, src

    cdef int H
    cdef double h_sub
    cdef bint approximate

    ##########################################################################
    # Member functions
    ##########################################################################

    cpdef set_context(self, int src_index, int dst_index)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cdef inline int _h_mask_approx(self, int* x, int* y, int* z) nogil

    cdef inline int _h_mask_exact(self, int* x, int* y, int* z) nogil

    cdef int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z, double h) nogil

    cdef inline void _add_to_hashtable(self, int hash_id, unsigned int pid, double h,
            int i, int j, int k) nogil

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)
