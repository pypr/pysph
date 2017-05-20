#cython: embedsignature=True

from nnps_base cimport *
from octree cimport Octree, CompressedOctree, cOctreeNode

from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

cimport cython
from cython.operator cimport dereference as deref, preincrement as inc

IF UNAME_SYSNAME == "Windows":
    cdef inline double fmin(double x, double y) nogil:
        return x if x < y else y
    cdef inline double fmax(double x, double y) nogil:
        return x if x > y else y


#############################################################################
cdef class OctreeNNPS(NNPS):
    """Nearest neighbor search using Octree.
    """
    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int leaf_max_particles = 10):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        cdef int i
        self.tree = [Octree(leaf_max_particles) for i in range(self.narrays)]

        self.radius_scale2 = radius_scale*radius_scale

        self.src_index = 0
        self.dst_index = 0
        self.leaf_max_particles = leaf_max_particles
        self.current_tree = NULL
        self.current_pids = NULL

        self.sort_gids = sort_gids
        self.domain.update()
        self.update()


    #### Public protocol ################################################

    cpdef set_context(self, int src_index, int dst_index):
        """Set context for nearest neighbor searches.

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        """

        NNPS.set_context(self, src_index, dst_index)
        self.current_tree = (<Octree>self.tree[src_index]).root
        self.current_pids = (<Octree>self.tree[src_index]).pids

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil:
        """Low level, high-performance non-gil method to find neighbors.
        This requires that `set_context()` be called beforehand.  This method
        does not reset the neighbors array before it appends the
        neighbors to it.

        """
        cdef double* dst_x_ptr = self.dst.x.data
        cdef double* dst_y_ptr = self.dst.y.data
        cdef double* dst_z_ptr = self.dst.z.data
        cdef double* dst_h_ptr = self.dst.h.data

        cdef double* src_x_ptr = self.src.x.data
        cdef double* src_y_ptr = self.src.y.data
        cdef double* src_z_ptr = self.src.z.data
        cdef double* src_h_ptr = self.src.h.data

        cdef double x = dst_x_ptr[d_idx]
        cdef double y = dst_y_ptr[d_idx]
        cdef double z = dst_z_ptr[d_idx]
        cdef double h = dst_h_ptr[d_idx]

        cdef u_int* s_gid = self.src.gid.data
        cdef int orig_length = nbrs.length

        self._get_neighbors(x, y, z, h, src_x_ptr, src_y_ptr, src_z_ptr,
                src_h_ptr, nbrs, self.current_tree)

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )


    #### Private protocol ################################################

    @cython.cdivision(True)
    cdef void _get_neighbors(self, double q_x, double q_y, double q_z, double q_h,
            double* src_x_ptr, double* src_y_ptr, double* src_z_ptr, double* src_h_ptr,
            UIntArray nbrs, cOctreeNode* node) nogil:
        """Find neighbors recursively"""
        cdef double x_centre = node.xmin[0] + node.length/2
        cdef double y_centre = node.xmin[1] + node.length/2
        cdef double z_centre = node.xmin[2] + node.length/2

        cdef u_int i, j, k
        cdef double hi2 = self.radius_scale2*q_h*q_h
        cdef double hj2 = 0
        cdef double xij2 = 0

        cdef double eff_radius = 0.5*(node.length) + \
                fmax(self.radius_scale*q_h, self.radius_scale*node.hmax)

        if  fabs(x_centre - q_x) >= eff_radius or \
            fabs(y_centre - q_y) >= eff_radius or \
            fabs(z_centre - q_z) >= eff_radius:
            return

        if node.is_leaf:
            for i from 0<=i<node.num_particles:
                k = self.current_pids[node.start_index + i]
                hj2 = self.radius_scale2*src_h_ptr[k]*src_h_ptr[k]
                xij2 = norm2(
                        src_x_ptr[k] - q_x,
                        src_y_ptr[k] - q_y,
                        src_z_ptr[k] - q_z
                        )
                if (xij2 < hi2) or (xij2 < hj2):
                    nbrs.c_append(k)
            return

        for i from 0<=i<8:
            if node.children[i] == NULL:
                continue
            self._get_neighbors(q_x, q_y, q_z, q_h,
                    src_x_ptr, src_y_ptr, src_z_ptr, src_h_ptr,
                    nbrs, node.children[i])

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        indices.reset()
        cdef int num_particles = (<Octree>self.tree[pa_index]).num_particles
        cdef u_int* current_pids = (<Octree>self.tree[pa_index]).pids

        cdef int i
        for i from 0<=i<num_particles:
            indices.c_append(<long>current_pids[i])

    cpdef _refresh(self):
        cdef int i
        for i from 0<=i<self.narrays:
            (<Octree>self.tree[i]).c_build_tree(self.pa_wrappers[i])
        self.current_tree = (<Octree>self.tree[self.src_index]).root
        self.current_pids = (<Octree>self.tree[self.src_index]).pids

    cpdef _bin(self, int pa_index, UIntArray indices):
        pass

#############################################################################
cdef class CompressedOctreeNNPS(OctreeNNPS):
    """Nearest neighbor search using Compressed Octree.
    """
    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int leaf_max_particles = 10):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        cdef int i
        self.tree = [CompressedOctree(leaf_max_particles) for i in range(self.narrays)]

        self.radius_scale2 = radius_scale*radius_scale

        self.src_index = 0
        self.dst_index = 0
        self.leaf_max_particles = leaf_max_particles
        self.current_tree = NULL
        self.current_pids = NULL

        self.sort_gids = sort_gids
        self.domain.update()
        self.update()


    #### Public protocol ################################################

    cpdef set_context(self, int src_index, int dst_index):
        """Set context for nearest neighbor searches.

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        """

        NNPS.set_context(self, src_index, dst_index)
        self.current_tree = (<CompressedOctree>self.tree[src_index]).root
        self.current_pids = (<CompressedOctree>self.tree[src_index]).pids

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]


    #### Private protocol ################################################

    cpdef _refresh(self):
        cdef int i
        for i from 0<=i<self.narrays:
            (<CompressedOctree>self.tree[i]).c_build_tree(self.pa_wrappers[i])
        self.current_tree = (<CompressedOctree>self.tree[self.src_index]).root
        self.current_pids = (<CompressedOctree>self.tree[self.src_index]).pids

    cpdef _bin(self, int pa_index, UIntArray indices):
        pass


