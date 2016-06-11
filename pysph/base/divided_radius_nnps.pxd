from libcpp.vector cimport vector

from nnps_base cimport *

ctypedef unsigned int u_int

#Imports for SpatialHashNNPS
cdef extern from "spatial_hash.h":
    cdef cppclass HashTable:
        long long int table_size

        HashTable(long long int) nogil except +
        void add(int, int, int, int) nogil
        vector[u_int]* get(int, int, int) nogil

# NNPS using Spatial Hashing algorithm
cdef class DividedRadiusNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef long long int table_size               # Size of hashtable
    cdef double radius_scale2

    cdef public int max_levels
    cdef double interval_size

    cdef HashTable*** hashtable
    cdef HashTable** current_hash

    cdef NNPSParticleArrayWrapper dst, src

    ##########################################################################
    # Member functions
    ##########################################################################

    cpdef set_context(self, int src_index, int dst_index)

    cpdef int count_particles(self, int interval)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc)

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z) nogil

    cdef inline int _get_hash_id(self, double h) nogil

    cdef inline double _get_cell_size(self, int hash_id) nogil

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)


