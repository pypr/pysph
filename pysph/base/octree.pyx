#cython: embedsignature=True

from nnps_base cimport *

from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

cimport cython
from cython.operator cimport dereference as deref, preincrement as inc

import numpy as np
cimport numpy as np

# EPS_MAX is maximum value of eps in tree building
DEF EPS_MAX = 1e-3

cdef class OctreeNode:
    def __init__(self):
        self.xmin = np.zeros(3, dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __richcmp__(self, OctreeNode other, int op):
        cdef bint equal_xmin, equal_length
        equal_xmin = True
        equal_length = (self.length == other.length)

        cdef int i
        for i from 0<=i<3:
            if self.xmin[i] != other.xmin[i]:
                equal_xmin = False

        cdef bint equal = equal_xmin and equal_length

        if op == 2:
            return equal
        if op == 3:
            return not equal

        return NotImplemented


    #### Public protocol ################################################

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void wrap_node(self, cOctreeNode* node):
        self._node = node
        self.hmax = node.hmax
        self.length = node.length
        self.is_leaf = node.is_leaf
        self.level = node.level

        self.xmin[0] = self._node.xmin[0]
        self.xmin[1] = self._node.xmin[1]
        self.xmin[2] = self._node.xmin[2]

    cpdef UIntArray get_indices(self):
        """ Get the indices in a node.

        Returns
        -------

        indices : UIntArray

        """
        if not self._node.is_leaf:
            return UIntArray()
        cdef UIntArray py_indices = UIntArray()
        py_indices.c_set_view(&self._node.indices.front(), self._node.num_particles)
        return py_indices

    cpdef OctreeNode get_parent(self):
        """ Get parent of the node.

        Returns
        -------

        parent : OctreeNode

        """
        if self._node.parent == NULL:
            return None
        cdef OctreeNode parent = OctreeNode()
        parent.wrap_node(self._node.parent)
        return parent

    cpdef list get_children(self):
        """ Get the children of a node.

        Returns
        -------

        children : list

        """
        if self._node.is_leaf:
            return []
        cdef int i
        cdef list py_children = [OctreeNode() for i in range(8)]
        for i from 0<=i<8:
            (<OctreeNode>py_children[i]).wrap_node(self._node.children[i])
        return py_children

    cpdef plot(self, ax, color="k"):
        """ Plots a node.

        Parameters
        ----------

        ax : mpl_toolkits.mplot3d.Axes3D instance

        color : color hex string/letter, default ("k")

        """
        cdef int i, j, k
        cdef double x, y, z
        cdef list ax_points = [0,0]

        for i from 0<=i<2:
            for j from 0<=j<2:
                x = self.xmin.data[0] + i*self.length
                y = self.xmin.data[1] + j*self.length
                for k from 0<=k<2:
                    ax_points[k] = self.xmin.data[2] + k*self.length

                ax.plot([x,x], [y,y], zs=ax_points[:], color=color)

        for i from 0<=i<2:
            for k from 0<=k<2:
                x = self.xmin.data[0] + i*self.length
                z = self.xmin.data[2] + k*self.length
                for j from 0<=j<2:
                    ax_points[j] = self.xmin.data[1] + j*self.length

                ax.plot([x,x], ax_points[:], zs=[z,z], color=color)

        for j from 0<=j<2:
            for k from 0<=k<2:
                y = self.xmin.data[1] + j*self.length
                z = self.xmin.data[2] + k*self.length
                for i from 0<=i<2:
                    ax_points[i] = self.xmin.data[0] + i*self.length

                ax.plot(ax_points[:], [y,y], zs=[z,z], color=color)

cdef class Octree:
    def __init__(self, int leaf_max_particles):
        self.leaf_max_particles = leaf_max_particles
        self.depth = 0
        self.tree = NULL
        self.leaf_cells = NULL
        self.machine_eps = np.finfo(float).eps

    def __dealloc__(self):
        if self.tree != NULL:
            self._delete_tree(self.tree)
        if self.leaf_cells != NULL:
            del self.leaf_cells


    #### Private protocol ################################################

    @cython.cdivision(True)
    cdef inline double _get_eps(self, double length, double* xmin) nogil:
        return (2*self.machine_eps/length)*fmax(length,
                fmax(fmax(fabs(xmin[0]), fabs(xmin[1])), fabs(xmin[2])))

    @cython.cdivision(True)
    cdef inline void _calculate_domain(self, NNPSParticleArrayWrapper pa_wrapper):
        cdef int num_particles = pa_wrapper.get_number_of_particles()
        pa_wrapper.pa.update_min_max()

        self.xmin[0] = pa_wrapper.x.minimum
        self.xmin[1] = pa_wrapper.y.minimum
        self.xmin[2] = pa_wrapper.z.minimum

        self.xmax[0] = pa_wrapper.x.maximum
        self.xmax[1] = pa_wrapper.y.maximum
        self.xmax[2] = pa_wrapper.z.maximum

        self.hmax = pa_wrapper.h.maximum

        cdef double x_length = self.xmax[0] - self.xmin[0]
        cdef double y_length = self.xmax[1] - self.xmin[1]
        cdef double z_length = self.xmax[2] - self.xmin[2]

        self.length = fmax(x_length, fmax(y_length, z_length))

        cdef double eps = self._get_eps(self.length, self.xmin)

        self.xmin[0] -= self.length*eps
        self.xmin[1] -= self.length*eps
        self.xmin[2] -= self.length*eps

        self.length *= (1 + 2*eps)

        # This is required to fix floating point errors. One such case
        # is mentioned in pysph.base.tests.test_octree
        self._eps0 = 2*self._get_eps(self.length, self.xmin)

    cdef inline cOctreeNode* _new_node(self, double* xmin, double length,
            double hmax = 0, int level = 0, cOctreeNode* parent = NULL,
            int num_particles = 0, bint is_leaf = False) nogil:
        """Create a new cOctreeNode"""
        cdef cOctreeNode* node = <cOctreeNode*> malloc(sizeof(cOctreeNode))

        node.xmin[0] = xmin[0]
        node.xmin[1] = xmin[1]
        node.xmin[2] = xmin[2]

        node.length = length
        node.hmax = hmax
        node.num_particles = num_particles
        node.is_leaf = is_leaf
        node.level = level

        node.parent = parent
        node.indices = NULL

        cdef int i

        for i from 0<=i<8:
            node.children[i] = NULL

        return node

    cdef inline void _delete_tree(self, cOctreeNode* node) nogil:
        """Delete octree"""
        cdef int i, j, k
        cdef cOctreeNode* temp[8]

        for i from 0<=i<8:
            temp[i] = node.children[i]

        if node.indices != NULL:
            del node.indices
        free(node)

        for i from 0<=i<8:
            if temp[i] == NULL:
                return
            else:
                self._delete_tree(temp[i])

    @cython.cdivision(True)
    cdef int _c_build_tree(self, NNPSParticleArrayWrapper pa,
            vector[u_int]* indices, double* xmin, double length,
            cOctreeNode* node, int level, double eps) nogil:
        cdef double* src_x_ptr = pa.x.data
        cdef double* src_y_ptr = pa.y.data
        cdef double* src_z_ptr = pa.z.data
        cdef double* src_h_ptr = pa.h.data

        cdef double xmin_new[3]
        cdef double hmax_children[8]
        cdef int depth_child = 0
        cdef int depth_max = 0

        cdef int i, j, k
        cdef u_int p, q

        for i from 0<=i<8:
            hmax_children[i] = 0

        cdef cOctreeNode* temp = NULL
        cdef int oct_id

        if (indices.size() < self.leaf_max_particles) or (eps > EPS_MAX):
            node.indices = indices
            node.num_particles = indices.size()
            node.is_leaf = True
            return 1

        cdef vector[u_int]* new_indices[8]
        for i from 0<=i<8:
            new_indices[i] = new vector[u_int]()

        for p from 0<=p<indices.size():
            q = deref(indices)[p]

            find_cell_id_raw(
                    src_x_ptr[q] - xmin[0],
                    src_y_ptr[q] - xmin[1],
                    src_z_ptr[q] - xmin[2],
                    length/2,
                    &i, &j, &k
                    )

            oct_id = k+2*j+4*i

            (new_indices[oct_id]).push_back(q)
            hmax_children[oct_id] = fmax(hmax_children[oct_id], src_h_ptr[q])

        cdef double length_padded = (length/2)*(1 + 2*eps)
        cdef double eps_new = 0

        del indices

        for i from 0<=i<2:
            for j from 0<=j<2:
                for k from 0<=k<2:

                    xmin_new[0] = xmin[0] + (i - eps)*length/2
                    xmin_new[1] = xmin[1] + (j - eps)*length/2
                    xmin_new[2] = xmin[2] + (k - eps)*length/2

                    eps_new = 2*self._get_eps(length_padded, xmin_new)

                    oct_id = k+2*j+4*i

                    node.children[oct_id] = self._new_node(xmin_new, length_padded,
                            hmax=hmax_children[oct_id], level=level+1, parent=node)

                    depth_child = self._c_build_tree(pa, new_indices[oct_id],
                            xmin_new, length_padded, node.children[oct_id], level+1, eps_new)

                    depth_max = <int>fmax(depth_max, depth_child)

        return 1 + depth_max

    cdef void _plot_tree(self, OctreeNode node, ax):
        node.plot(ax)

        cdef OctreeNode child
        cdef list children = node.get_children()

        for child in children:
            self._plot_tree(child, ax)

    cdef void _c_get_leaf_cells(self, cOctreeNode* node):
        if node.is_leaf:
            self.leaf_cells.push_back(node)
            return

        cdef int i
        for i from 0<=i<8:
            if node.children[i] != NULL:
                self._c_get_leaf_cells(node.children[i])


    #### Public protocol ################################################

    cdef int c_build_tree(self, NNPSParticleArrayWrapper pa_wrapper):

        self._calculate_domain(pa_wrapper)

        cdef int num_particles = pa_wrapper.get_number_of_particles()
        cdef vector[u_int]* indices_ptr = new vector[u_int]()

        cdef int i
        for i from 0<=i<num_particles:
            indices_ptr.push_back(i)

        if self.tree != NULL:
            self._delete_tree(self.tree)
        if self.leaf_cells != NULL:
            del self.leaf_cells

        self.tree = self._new_node(self.xmin, self.length,
                hmax=self.hmax, level=0)

        self.depth = self._c_build_tree(pa_wrapper, indices_ptr, self.tree.xmin,
                self.tree.length, self.tree, 0, self._eps0)

        return self.depth

    cdef void c_get_leaf_cells(self):
        if self.leaf_cells != NULL:
            return

        self.leaf_cells = new vector[cOctreeNode*]()
        self._c_get_leaf_cells(self.tree)

    @cython.cdivision(True)
    cdef cOctreeNode* c_find_point(self, double x, double y, double z):
        cdef cOctreeNode* node = self.tree
        cdef cOctreeNode* prev = self.tree

        cdef int i, j, k, oct_id
        while node != NULL:
            find_cell_id_raw(
                    x - node.xmin[0],
                    y - node.xmin[1],
                    z - node.xmin[2],
                    node.length/2,
                    &i, &j, &k
                    )

            oct_id = k+2*j+4*i
            prev = node
            node = node.children[oct_id]

        return prev


    ######################################################################

    cpdef int build_tree(self, ParticleArray pa):
        """ Build tree.

        Parameters
        ----------

        pa : ParticleArray

        Returns
        -------

        depth : int
        Maximum depth of the tree

        """
        cdef NNPSParticleArrayWrapper pa_wrapper = NNPSParticleArrayWrapper(pa)
        return self.c_build_tree(pa_wrapper)

    cpdef OctreeNode get_root(self):
        """ Get root of the tree

        Returns
        -------

        root : OctreeNode
        Root of the tree

        """
        cdef OctreeNode py_node = OctreeNode()
        py_node.wrap_node(self.tree)
        return py_node

    cpdef list get_leaf_cells(self):
        """ Get all leaf cells in the tree

        Returns
        -------

        leaf_cells : list
        List of leaf cells in the tree

        """
        self.c_get_leaf_cells()
        cdef int i
        cdef list py_leaf_cells = [OctreeNode() for i in range(self.leaf_cells.size())]
        for i from 0<=i<self.leaf_cells.size():
            (<OctreeNode>py_leaf_cells[i]).wrap_node(deref(self.leaf_cells)[i])
        return py_leaf_cells

    cpdef OctreeNode find_point(self, double x, double y, double z):
        """Get the leaf node to which a point belongs

        Parameters
        ----------

        x, y, z : double
        Co-ordinates of the point

        Returns
        -------

        node : OctreeNode
        Leaf node to which the point belongs

        """
        cdef cOctreeNode* node = self.c_find_point(x, y, z)
        cdef OctreeNode py_node = OctreeNode()
        py_node.wrap_node(node)
        return py_node

    cpdef plot(self, ax):
        """Plots the tree

        Parameters
        ----------

        ax : mpl_toolkits.mplot3d.Axes3D instance

        """
        cdef OctreeNode root = self.get_root()
        self._plot_tree(root, ax)

