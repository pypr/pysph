#cython: embedsignature=True

from nnps_base cimport *
from octree cimport Octree, CompressedOctree, cOctreeNode

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

cdef class OctreeNNPS(NNPS):
    ##########################################################################
    # Data Attributes
    ##########################################################################
    cdef list tree
    cdef cOctreeNode* current_tree
    cdef u_int* current_pids

    cdef double radius_scale2
    cdef NNPSParticleArrayWrapper dst, src
    cdef int leaf_max_particles

    ##########################################################################
    # Member functions
    ##########################################################################
    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil

    cpdef set_context(self, int src_index, int dst_index)

    cdef void _get_neighbors(self, double q_x, double q_y, double q_z, double q_h,
            double* src_x_ptr, double* src_y_ptr, double* src_z_ptr, double* src_h_ptr,
            UIntArray nbrs, cOctreeNode* node) nogil

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices)

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)

cdef class CompressedOctreeNNPS(OctreeNNPS):
    ##########################################################################
    # Member functions
    ##########################################################################
    cpdef set_context(self, int src_index, int dst_index)

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)



