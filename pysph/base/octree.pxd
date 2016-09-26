
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

cdef extern from 'float.h':
    cdef double DBL_MAX

cdef struct OctreeNode:
    bint is_leaf
    double length
    double xmin[3]
    double hmax
    int num_particles

    void* indices
    OctreeNode* children[8]

cdef class Octree:
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef OctreeNode* tree

    cdef int leaf_max_particles
    cdef double radius_scale

    cdef double xmin[3]
    cdef double xmax[3]
    cdef double hmax

    cdef double _eps0

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef inline void _calculate_domain(self, NNPSParticleArrayWrapper pa)

    cdef inline OctreeNode* _new_node(self, double* xmin, double length,
            double hmax = *, int num_particles = *, bint is_leaf = *) nogil

    cdef inline void _delete_tree(self, OctreeNode* node)

    cdef void _c_build_tree(self, NNPSParticleArrayWrapper pa, UIntArray indices,
            double* xmin, double length, OctreeNode* node, double eps)

    cdef void c_build_tree(self, NNPSParticleArrayWrapper pa_wrapper)

    cpdef build_tree(self, ParticleArray pa)


