#cython: embedsignature=True

from nnps_base cimport *

from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t

cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
from cpython cimport PyObject, Py_XINCREF, Py_XDECREF

DEF EPS_MAX = 1e-3
DEF MACHINE_EPS = 1e-14

cdef class Octree:
    def __init__(self, int leaf_max_particles, double radius_scale):
        self.leaf_max_particles = leaf_max_particles
        self.radius_scale = radius_scale
        self.depth = 0
        self.tree = NULL
        self.linear_tree = NULL

    def __dealloc__(self):
        if self.linear_tree != NULL:
            free(self.linear_tree)
        self._delete_tree(self.tree)


    cdef inline void _calculate_domain(self, NNPSParticleArrayWrapper pa):
        cdef int num_particles = pa.get_number_of_particles()

        cdef double xmin = DBL_MAX
        cdef double ymin = DBL_MAX
        cdef double zmin = DBL_MAX

        cdef double xmax = -DBL_MAX
        cdef double ymax = -DBL_MAX
        cdef double zmax = -DBL_MAX

        cdef double hmax = 0

        for i from 0<=i<num_particles:
            xmax = fmax(xmax, pa.x.data[i])
            ymax = fmax(ymax, pa.y.data[i])
            zmax = fmax(zmax, pa.z.data[i])

            xmin = fmin(xmin, pa.x.data[i])
            ymin = fmin(ymin, pa.y.data[i])
            zmin = fmin(zmin, pa.z.data[i])

            hmax = fmax(hmax, pa.h.data[i])

        self.xmin[0] = xmin
        self.xmin[1] = ymin
        self.xmin[2] = zmin

        self.xmax[0] = xmax
        self.xmax[1] = ymax
        self.xmax[2] = zmax

        self.hmax = hmax

    cdef inline cOctreeNode* _new_node(self, double* xmin, double length,
            double hmax = 0, int num_particles = 0, bint is_leaf = False) nogil:
        """Create a new cOctreeNode"""
        cdef cOctreeNode* node = <cOctreeNode*> malloc(sizeof(cOctreeNode))

        node.xmin[0] = xmin[0]
        node.xmin[1] = xmin[1]
        node.xmin[2] = xmin[2]

        node.length = length
        node.hmax = hmax
        node.num_particles = num_particles
        node.is_leaf = is_leaf
        node.indices = NULL

        cdef int i

        for i from 0<=i<8:
            node.children[i] = NULL

        return node

    cdef inline void _delete_tree(self, cOctreeNode* node):
        """Delete octree"""
        cdef int i, j, k
        cdef cOctreeNode* temp[8]

        for i from 0<=i<8:
            temp[i] = node.children[i]

        Py_XDECREF(<PyObject*>node.indices)
        free(node)

        for i from 0<=i<8:
            if temp[i] == NULL:
                return
            else:
                self._delete_tree(temp[i])

    cdef int _c_build_tree(self, NNPSParticleArrayWrapper pa, UIntArray indices,
            double* xmin, double length, cOctreeNode* node, double eps):

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

        if (indices.length < self.leaf_max_particles) or (eps > EPS_MAX):
            node.indices = <void*>indices
            Py_XINCREF(<PyObject*>indices)
            node.num_particles = indices.length
            node.is_leaf = True
            return 1

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

            oct_id = k+2*j+4*i

            (<UIntArray>new_indices[oct_id]).c_append(q)
            hmax_children[oct_id] = fmax(hmax_children[oct_id],
                    self.radius_scale*src_h_ptr[q])

        cdef double length_padded = (length/2)*(1 + 2*eps)

        for i from 0<=i<2:
            for j from 0<=j<2:
                for k from 0<=k<2:

                    xmin_new[0] = xmin[0] + (i - eps)*length/2
                    xmin_new[1] = xmin[1] + (j - eps)*length/2
                    xmin_new[2] = xmin[2] + (k - eps)*length/2

                    oct_id = k+2*j+4*i

                    node.children[oct_id] = self._new_node(xmin_new, length_padded,
                            hmax=hmax_children[oct_id])

                    depth_child = self._c_build_tree(pa, <UIntArray>new_indices[oct_id],
                            xmin_new, length_padded, node.children[oct_id], 2*eps)

                    depth_max = <int>fmax(depth_max, depth_child)

        return 1 + depth_max

    cdef int c_build_tree(self, NNPSParticleArrayWrapper pa_wrapper):

        self._calculate_domain(pa_wrapper)

        cdef double x_length = self.xmax[0] - self.xmin[0]
        cdef double y_length = self.xmax[1] - self.xmin[1]
        cdef double z_length = self.xmax[2] - self.xmin[2]

        cdef double length = fmax(x_length, fmax(y_length, z_length))

        cdef double eps = (MACHINE_EPS/length)*fmax(length,
                fmax(fmax(fabs(self.xmin[0]), fabs(self.xmin[1])), fabs(self.xmin[2])))

        self.xmin[0] -= length*eps
        self.xmin[1] -= length*eps
        self.xmin[2] -= length*eps

        length *= (1 + 2*eps)

        cdef double xmax_padded = self.xmin[0] + length
        cdef double ymax_padded = self.xmin[1] + length
        cdef double zmax_padded = self.xmin[2] + length

        self._eps0 = (2*MACHINE_EPS/length)*fmax(length,
                fmax(fmax(fabs(xmax_padded), fabs(ymax_padded)), fabs(zmax_padded)))

        cdef int num_particles = pa_wrapper.get_number_of_particles()
        cdef UIntArray indices = UIntArray()
        indices.c_reserve(num_particles)

        cdef int i
        for i from 0<=i<num_particles:
            indices.c_append(i)

        if self.tree != NULL:
            self._delete_tree(self.tree)
        self.tree = self._new_node(self.xmin, length, hmax=self.radius_scale*self.hmax)

        self.depth = self._c_build_tree(pa_wrapper, indices, self.tree.xmin,
                self.tree.length, self.tree, self._eps0)

        return self.depth

    cdef void _c_build_linear_index(self, cOctreeNode* node, uint64_t parent_key) nogil:
        cdef uint64_t child_base = parent_key << 3
        cdef uint64_t child_key
        cdef cOctreeNode* current_node

        cdef int i, j, k, oct_id
        for i from 0<=i<2:
            for j from 0<=j<2:
                for k from 0<=k<2:
                    oct_id = k+2*j+4*i
                    current_node = node.children[oct_id]
                    if current_node == NULL:
                        return
                    child_key = child_base + oct_id
                    self.linear_tree[child_key] = current_node
                    self._c_build_linear_index(current_node, child_key)

    cdef void c_build_linear_index(self):
        if self.depth > 22:
            raise ValueError("Depth of tree too large for a linear index")
        cdef uint64_t length_array = (1 << (3*self.depth-2)) - 1
        if self.linear_tree != NULL:
            free(self.linear_tree)
        self.linear_tree = <cOctreeNode**> malloc(length_array*sizeof(cOctreeNode*))
        self.linear_tree[1] = self.tree
        self._c_build_linear_index(self.tree, 1)

    cpdef int build_tree(self, ParticleArray pa):
        cdef NNPSParticleArrayWrapper pa_wrapper = NNPSParticleArrayWrapper(pa)
        return self.c_build_tree(pa_wrapper)

    cpdef build_linear_index(self):
        self.c_build_linear_index()



