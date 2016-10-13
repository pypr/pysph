from libcpp.vector cimport vector

from nnps_base cimport *

ctypedef unsigned int u_int

cdef extern from 'math.h':
    int abs(int) nogil
    double ceil(double) nogil
    double floor(double) nogil
    double fabs(double) nogil
    double fmax(double, double) nogil
    double fmin(double, double) nogil

#Imports for SpatialHashNNPS
cdef extern from "spatial_hash.h":
    cdef cppclass HashEntry:
        double h_max

        vector[unsigned int] *get_indices() nogil

    cdef cppclass HashTable:
        long long int table_size

        HashTable(long long int) nogil except +
        void add(int, int, int, int, double) nogil
        HashEntry* get(int, int, int) nogil
        int number_of_particles() nogil

cdef class StratifiedHashNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef long long int table_size               # Size of hashtable
    cdef double radius_scale2

    cdef public int num_levels
    cdef public int H

    cdef double interval_size

    cdef HashTable*** hashtable
    cdef HashTable** current_hash

    cdef double** cell_sizes
    cdef double* current_cells

    cdef NNPSParticleArrayWrapper dst, src

    ##########################################################################
    # Member functions
    ##########################################################################

    cpdef set_context(self, int src_index, int dst_index)

    cpdef int count_particles(self, int interval)

    cpdef double get_binning_size(self, int interval)

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cdef inline int _h_mask_exact(self, int* x, int* y, int* z, int H) nogil

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z, int H) nogil

    cdef inline int _get_hash_id(self, double h) nogil

    cdef inline void _set_h_max(self, double* current_cells, double* src_h_ptr,
            int num_particles) nogil

    cdef inline double _get_h_max(self, double* current_cells, int hash_id) nogil

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)


