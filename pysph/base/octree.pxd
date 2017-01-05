#cython: embedsignature=True

from nnps_base cimport *
from libcpp.vector cimport vector
cimport cython

import numpy as np
cimport numpy as np

ctypedef unsigned int u_int

cdef extern from 'math.h':
    int abs(int) nogil
    double ceil(double) nogil
    double floor(double) nogil
    double fabs(double) nogil
    double fmax(double, double) nogil
    double fmin(double, double) nogil

cdef extern from "<algorithm>" namespace "std" nogil:
    OutputIter copy[InputIter,OutputIter](InputIter,InputIter,OutputIter)

cdef struct cOctreeNode:
    bint is_leaf
    double length
    double xmin[3]
    double hmax
    int num_particles
    int level

    int start_index
    cOctreeNode* children[8]
    cOctreeNode* parent

cdef class OctreeNode:
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef cOctreeNode* _node

    cdef public bint is_leaf
    cdef public double length
    cdef public np.ndarray xmin
    cdef public double hmax
    cdef public int num_particles
    cdef public int level

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef void wrap_node(self, cOctreeNode* node)

    cpdef OctreeNode get_parent(self)

    cpdef UIntArray get_indices(self, Octree tree)

    cpdef list get_children(self)

    cpdef plot(self, ax, color = *)

cdef class Octree:
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef cOctreeNode* root
    cdef vector[cOctreeNode*]* leaf_cells
    cdef u_int* pids

    cdef int _next_pid
    cdef public int num_particles

    cdef public int leaf_max_particles
    cdef public double hmax
    cdef public double length
    cdef public int depth

    cdef double machine_eps

    cdef double xmin[3]
    cdef double xmax[3]

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef inline double _get_eps(self, double length, double* xmin) nogil

    cdef inline void _calculate_domain(self, NNPSParticleArrayWrapper pa)

    cdef inline cOctreeNode* _new_node(self, double* xmin, double length,
            double hmax = *, int level = *, cOctreeNode* parent = *,
            int num_particles = *, bint is_leaf = *) nogil

    cdef inline void _delete_tree(self, cOctreeNode* node)

    cdef int _c_build_tree(self, NNPSParticleArrayWrapper pa,
            vector[u_int]* indices, double* xmin, double length,
            cOctreeNode* node, int level) nogil

    cdef void _plot_tree(self, OctreeNode node, ax)

    cdef int c_build_tree(self, NNPSParticleArrayWrapper pa_wrapper)

    cdef void _c_get_leaf_cells(self, cOctreeNode* node)

    cdef void c_get_leaf_cells(self)

    cdef cOctreeNode* c_find_point(self, double x, double y, double z)

    cpdef int build_tree(self, ParticleArray pa)

    cpdef delete_tree(self)

    cpdef OctreeNode get_root(self)

    cpdef list get_leaf_cells(self)

    cpdef OctreeNode find_point(self, double x, double y, double z)

    cpdef plot(self, ax)

cdef class CompressedOctree(Octree):
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef double dbl_max

    ##########################################################################
    # Member functions
    ##########################################################################

    cdef int _c_build_tree(self, NNPSParticleArrayWrapper pa,
            vector[u_int]* indices, double* xmin, double length,
            cOctreeNode* node, int level) nogil


