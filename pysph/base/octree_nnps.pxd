#cython: embedsignature=True

from nnps_base cimport *
from libcpp.vector cimport vector
cimport cython

ctypedef unsigned int u_int

cdef extern from 'math.h':
    int abs(int) nogil
    double ceil(double) nogil
    double floor(double) nogil
    double fabs(double) nogil
    double fmax(double, double) nogil
    double fmin(double, double) nogil

cdef struct OctreeNode:
    bint is_leaf
    double length
    double xmin[3]
    double hmax
    int num_particles

    void* indices
    OctreeNode* children[2][2][2]

cdef inline OctreeNode* new_node(double* xmin, double length,
        double hmax = *, int num_particles = *, bint is_leaf = *) nogil

cdef inline void delete_tree(OctreeNode* root)

cdef class OctreeNNPS(NNPS):
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef OctreeNode** root
    cdef OctreeNode* current_tree

    cdef double radius_scale2
    cdef NNPSParticleArrayWrapper dst, src
    cdef int leaf_max_particles

    cdef double _eps0

    ##########################################################################
    # Member functions
    ##########################################################################
    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc)

    cpdef set_context(self, int src_index, int dst_index)

    cdef void _build_tree(self, NNPSParticleArrayWrapper pa, UIntArray indices,
            double* xmin, double length, OctreeNode* node, double eps)

    cdef void _get_neighbors(self, double q_x, double q_y, double q_z, double q_h,
            double* src_x_ptr, double* src_y_ptr, double* src_z_ptr, double* src_h_ptr,
            UIntArray nbrs, OctreeNode* node) nogil

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)


