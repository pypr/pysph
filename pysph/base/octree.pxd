
from nnps_base cimport *
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libc.stdint cimport uint64_t
cimport cython

ctypedef unsigned int u_int

cdef extern from 'math.h':
    int abs(int) nogil
    double ceil(double) nogil
    double floor(double) nogil
    double fabs(double) nogil
    double fmax(double, double) nogil
    double fmin(double, double) nogil

cdef extern from 'float.h':
    cdef double DBL_MAX

cdef struct cOctreeNode:
    bint is_leaf
    double length
    double xmin[3]
    double hmax
    int num_particles

    void* indices
    cOctreeNode* children[8]

cdef class OctreeNode:
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef cOctreeNode* _node

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef void wrap_node(self, cOctreeNode* node)

cdef class Octree:
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef cOctreeNode* tree
    cdef cOctreeNode** linear_tree

    cdef int leaf_max_particles
    cdef double radius_scale

    cdef double xmin[3]
    cdef double xmax[3]
    cdef double hmax
    cdef int depth

    cdef double _eps0

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef inline void _calculate_domain(self, NNPSParticleArrayWrapper pa)

    cdef inline cOctreeNode* _new_node(self, double* xmin, double length,
            double hmax = *, int num_particles = *, bint is_leaf = *) nogil

    cdef inline void _delete_tree(self, cOctreeNode* node)

    cdef int _c_build_tree(self, NNPSParticleArrayWrapper pa, UIntArray indices,
            double* xmin, double length, cOctreeNode* node, double eps)

    cdef int c_build_tree(self, NNPSParticleArrayWrapper pa_wrapper)

    cdef void _c_build_linear_index(self, cOctreeNode* node, uint64_t parent_key) nogil

    cdef void c_build_linear_index(self)

    cpdef int build_tree(self, ParticleArray pa)

    cpdef build_linear_index(self)

    cpdef OctreeNode get_node(self, uint64_t key)

