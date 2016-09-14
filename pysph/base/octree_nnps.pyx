#cython: embedsignature=True

from nnps_base cimport *

from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

cimport cython
from cython.operator cimport dereference as deref, preincrement as inc

DEF EPS = 1e-6

cdef inline OctreeNode* new_node(double* xmin, double length,
        double hmax = 0, int num_particles = 0, bint is_leaf = False) nogil:
    cdef OctreeNode* node = <OctreeNode*> malloc(sizeof(OctreeNode))

    node.xmin[0] = xmin[0]
    node.xmin[1] = xmin[1]
    node.xmin[2] = xmin[2]

    node.length = length
    node.hmax = hmax
    node.num_particles = num_particles
    node.is_leaf = is_leaf
    node.indices = new vector[u_int]()

    cdef int i, j, k

    for i from 0<=i<2:
        for j from 0<=j<2:
            for k from 0<=k<2:
                node.children[i][j][k] = NULL

    return node

cdef inline void delete_tree(OctreeNode* root) nogil:
    """Delete octree"""
    cdef int i, j, k
    cdef OctreeNode* temp[8]

    for i from 0<=i<2:
        for j from 0<=j<2:
            for k from 0<=k<2:
                temp[k+2*j+4*i] = root.children[i][j][k]

    del root.indices
    free(root)

    for i from 0<=i<8:
        if temp[i] == NULL:
            return
        else:
            delete_tree(temp[i])


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
        self.radius_scale2 = radius_scale*radius_scale

        self.src_index = 0
        self.dst_index = 0
        self.leaf_max_particles = leaf_max_particles

        self.sort_gids = sort_gids
        self.domain.update()

        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int leaf_max_particles = 10):
        self.root = <OctreeNode**> malloc(self.narrays*sizeof(OctreeNode*))
        self.current_tree = NULL

    def __dealloc__(self):
        cdef int i
        for i from 0<=i<self.narrays:
            delete_tree(self.root[i])
        free(self.root)


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
        self.current_tree = self.root[src_index]

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
                src_h_ptr, nbrs, self.root[self.src_index])

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Find nearest neighbors for particle id 'd_idx' without cache

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        d_idx: size_t
            Index of the query point in the destination particle array

        nbrs: UIntArray
            Array to be populated by nearest neighbors of 'd_idx'

        """
        self.set_context(src_index, dst_index)

        if prealloc:
            nbrs.length = 0
        else:
            nbrs.c_reset()

        self.find_nearest_neighbors(d_idx, nbrs)


    #### Private protocol ################################################

    @cython.cdivision(True)
    cdef void _build_tree(self, NNPSParticleArrayWrapper pa, UIntArray indices,
            double* xmin, double length, OctreeNode* root):
        """Build octree"""

        cdef double* src_x_ptr = pa.x.data
        cdef double* src_y_ptr = pa.y.data
        cdef double* src_z_ptr = pa.z.data
        cdef double* src_h_ptr = pa.h.data

        cdef double xmin_new[3]
        cdef double hmax_children[8]

        cdef int i, j, k
        cdef u_int p, q

        for i from 0<=i<8:
            hmax_children[i] = 0

        cdef OctreeNode* temp = NULL

        if indices.length < self.leaf_max_particles:
            root.indices.assign(indices.data, indices.data + indices.length)
            root.num_particles = indices.length
            root.is_leaf = True
            return

        cdef list new_indices = [UIntArray() for i in range(8)]

        for p from 0<=p<indices.length:
            q = indices.data[p]

            find_cell_id_raw(
                    src_x_ptr[q] - xmin[0],
                    src_y_ptr[q] - xmin[1],
                    src_z_ptr[q] - xmin[2],
                    length/2,
                    &i, &j, &k
                    )

            (<UIntArray>new_indices[k+2*j+4*i]).c_append(q)
            hmax_children[k+2*j+4*i] = fmax(hmax_children[k+2*j+4*i],
                    self.radius_scale*src_h_ptr[q])

        cdef double length_padded = (length/2)*(1 + 2*EPS)

        for i from 0<=i<2:
            for j from 0<=j<2:
                for k from 0<=k<2:

                    xmin_new[0] = xmin[0] + (i - EPS)*length/2
                    xmin_new[1] = xmin[1] + (j - EPS)*length/2
                    xmin_new[2] = xmin[2] + (k - EPS)*length/2

                    root.children[i][j][k] = new_node(xmin_new, length_padded,
                            hmax=hmax_children[k+2*j+4*i])

                    self._build_tree(pa, <UIntArray>new_indices[k+2*j+4*i],
                            xmin_new, length_padded, root.children[i][j][k])


    @cython.cdivision(True)
    cdef void _get_neighbors(self, double q_x, double q_y, double q_z, double q_h,
            double* src_x_ptr, double* src_y_ptr, double* src_z_ptr, double* src_h_ptr,
            UIntArray nbrs, OctreeNode* root) nogil:
        """Find neighbors recursively"""
        cdef double x_centre = root.xmin[0] + root.length/2
        cdef double y_centre = root.xmin[1] + root.length/2
        cdef double z_centre = root.xmin[2] + root.length/2

        cdef u_int i, j, k
        cdef double hi2 = self.radius_scale2*q_h*q_h
        cdef double hj2 = 0
        cdef double xij2 = 0

        if root.is_leaf:
            for i from 0<=i<root.indices.size():
                k = deref(root.indices)[i]
                hj2 = self.radius_scale2*src_h_ptr[k]*src_h_ptr[k]
                xij2 = norm2(
                        src_x_ptr[k] - q_x,
                        src_y_ptr[k] - q_y,
                        src_z_ptr[k] - q_z
                        )
                if (xij2 < hi2) or (xij2 < hj2):
                    nbrs.c_append(k)
            return

        cdef double eff_radius = 0.5*(root.length) + fmax(self.radius_scale*q_h, root.hmax)

        if  fabs(x_centre - q_x) >= eff_radius or \
            fabs(y_centre - q_y) >= eff_radius or \
            fabs(z_centre - q_z) >= eff_radius:
            return

        for i from 0<=i<2:
            for j from 0<=j<2:
                for k from 0<=k<2:
                    self._get_neighbors(q_x, q_y, q_z, q_h,
                            src_x_ptr, src_y_ptr, src_z_ptr, src_h_ptr,
                            nbrs, root.children[i][j][k])


    cpdef _refresh(self):
        cdef double* xmin = self.xmin.data
        cdef double* xmax = self.xmax.data

        cdef double x_length = xmax[0] - xmin[0]
        cdef double y_length = xmax[1] - xmin[1]
        cdef double z_length = xmax[2] - xmin[2]

        cdef double length = fmax(x_length, fmax(y_length, z_length))

        cdef double x_centre = xmin[0] + length/2
        cdef double y_centre = xmin[1] + length/2
        cdef double z_centre = xmin[2] + length/2

        length *= (1 + 2*EPS)

        xmin[0] = x_centre - length/2
        xmin[1] = y_centre - length/2
        xmin[2] = z_centre - length/2

        cdef int i
        for i from 0<=i<self.narrays:
            if self.root[i] != NULL:
                delete_tree(self.root[i])
            self.root[i] = new_node(xmin, length, hmax=self.cell_size)
        self.current_tree = self.root[self.src_index]

    cpdef _bin(self, int pa_index, UIntArray indices):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        cdef OctreeNode* tree = self.root[pa_index]

        self._build_tree(pa_wrapper, indices, tree.xmin, tree.length,
                tree)


