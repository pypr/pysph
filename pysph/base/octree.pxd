
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

    vector[u_int]* indices
    cOctreeNode* children[8]
    cOctreeNode* parent

cdef class OctreeNode:
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef cOctreeNode* _node

    cdef public bint is_leaf
    cdef public double length
    cdef public DoubleArray xmin
    cdef public double hmax
    cdef public int num_particles

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef void wrap_node(self, cOctreeNode* node)

    cpdef UIntArray get_indices(self)

    cpdef OctreeNode get_parent(self)

    cpdef list get_children(self)

    cpdef plot(self, ax)

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
    cdef double length
    cdef int depth

    cdef double _eps0

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef inline void _calculate_domain(self, NNPSParticleArrayWrapper pa)

    cdef inline cOctreeNode* _new_node(self, double* xmin, double length,
            double hmax = *, cOctreeNode* parent = *, int num_particles = *,
            bint is_leaf = *) nogil

    cdef inline void _delete_tree(self, cOctreeNode* node)

    cdef int _c_build_tree(self, NNPSParticleArrayWrapper pa,
            vector[u_int]* indices_ptr, double* xmin, double length,
            cOctreeNode* node, double eps)

    cdef int c_build_tree(self, NNPSParticleArrayWrapper pa_wrapper)

    cpdef int build_tree(self, ParticleArray pa)

    cpdef OctreeNode get_root(self)

    cdef void _plot_tree(self, OctreeNode node, ax)

    cpdef plot(self, ax)

