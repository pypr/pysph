#cython: embedsignature=True

from .nnps_base cimport *

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

cimport cython
from cython.operator cimport dereference as deref, preincrement as inc

import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange, threadid

# EPS_MAX is maximum value of eps in tree building
DEF EPS_MAX = 1e-3

IF UNAME_SYSNAME == "Windows":
    cdef inline double fmin(double x, double y) nogil:
        return x if x < y else y
    cdef inline double fmax(double x, double y) nogil:
        return x if x > y else y

ctypedef cOctreeNode* node_ptr
ctypedef double* dbl_ptr

cdef extern from *:
    """
    #define START_OMP_PARALLEL_PRAGMA() _Pragma("omp parallel") {
    #define END_OMP_PRAGMA() }
    #define START_OMP_SINGLE_PRAGMA() _Pragma("omp single") {
    #define START_OMP_BARRIER_PRAGMA() _Pragma("omp barrier") {
    """
    void START_OMP_PARALLEL_PRAGMA() nogil
    void END_OMP_PRAGMA() nogil
    void START_OMP_SINGLE_PRAGMA() nogil
    void START_OMP_BARRIER_PRAGMA() nogil

########################################################################

cdef class OctreeNode:
    def __init__(self):
        self.xmin = np.zeros(3, dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __richcmp__(self, other, int op):
        # Checks if two nodes are the same.
        # Two nodes are equal if xmin and length are equal.
        # op = 2 corresponds to "=="
        # op = 3 corresponds to "!="
        if type(other) != OctreeNode:
            if op == 2:
                return False
            if op == 3:
                return True
            return NotImplemented

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
        self.num_particles = node.num_particles

        self.xmin[0] = self._node.xmin[0]
        self.xmin[1] = self._node.xmin[1]
        self.xmin[2] = self._node.xmin[2]

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

    cpdef UIntArray get_indices(self, Octree tree):
        """ Get indices of a node. Returns empty UIntArray
        if node is not a leaf.

        Returns
        -------

        indices : UIntArray

        """
        if not self._node.is_leaf:
            return UIntArray()
        cdef int idx = self._node.start_index
        cdef UIntArray node_indices = UIntArray()
        cdef u_int* indices = tree.pids
        node_indices.c_set_view(indices + idx,
                self._node.num_particles)
        return node_indices

    cpdef list get_children(self):
        """ Get the children of a node.

        Returns
        -------

        children : list

        """
        if self._node.is_leaf:
            return []
        cdef int i
        cdef list py_children = [None for i in range(8)]
        cdef OctreeNode py_node
        for i from 0<=i<8:
            if self._node.children[i] != NULL:
                py_node = OctreeNode()
                py_node.wrap_node(self._node.children[i])
                py_children[i] = py_node
        return py_children

    @cython.boundscheck(False)
    @cython.wraparound(False)
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
                x = self.xmin[0] + i*self.length
                y = self.xmin[1] + j*self.length
                for k from 0<=k<2:
                    ax_points[k] = self.xmin[2] + k*self.length

                ax.plot([x,x], [y,y], zs=ax_points[:], color=color)

        for i from 0<=i<2:
            for k from 0<=k<2:
                x = self.xmin[0] + i*self.length
                z = self.xmin[2] + k*self.length
                for j from 0<=j<2:
                    ax_points[j] = self.xmin[1] + j*self.length

                ax.plot([x,x], ax_points[:], zs=[z,z], color=color)

        for j from 0<=j<2:
            for k from 0<=k<2:
                y = self.xmin[1] + j*self.length
                z = self.xmin[2] + k*self.length
                for i from 0<=i<2:
                    ax_points[i] = self.xmin[0] + i*self.length

                ax.plot(ax_points[:], [y,y], zs=[z,z], color=color)

cdef class Octree:
    def __init__(self, int leaf_max_particles):
        self.leaf_max_particles = leaf_max_particles
        self.depth = 0
        self.root = NULL
        self.leaf_cells = NULL
        self.machine_eps = 16*np.finfo(float).eps
        self.pids = NULL

    def __dealloc__(self):
        if self.root != NULL:
            self._delete_tree(self.root)
        if self.pids != NULL:
            free(self.pids)
        if self.leaf_cells != NULL:
            del self.leaf_cells


    #### Private protocol ################################################

    @cython.cdivision(True)
    cdef inline double _get_eps(self, double length, double* xmin) nogil:
        return (self.machine_eps/length)*fmax(length,
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
        node.start_index = -1

        node.parent = parent

        cdef int i

        for i from 0<=i<8:
            node.children[i] = NULL

        return node

    cdef inline void _delete_tree(self, cOctreeNode* node):
        """Delete octree"""
        cdef int i
        cdef cOctreeNode* temp[8]

        for i from 0<=i<8:
            temp[i] = node.children[i]

        free(node)

        for i from 0<=i<8:
            if temp[i] == NULL:
                continue
            self._delete_tree(temp[i])

    @cython.cdivision(True)
    cdef int _c_build_tree(self, NNPSParticleArrayWrapper pa,
            vector[u_int]* indices, double* xmin, double length,
            cOctreeNode* node, int level) nogil:
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

        cdef int oct_id

        # This is required to fix floating point errors. One such case
        # is mentioned in pysph.base.tests.test_octree
        cdef double eps = 2*self._get_eps(length, xmin)

        if (indices.size() < self.leaf_max_particles) or (eps > EPS_MAX):
            copy(indices.begin(), indices.end(), self.pids + self._next_pid)
            node.start_index = self._next_pid
            self._next_pid += indices.size()
            node.num_particles = indices.size()
            del indices
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

        del indices

        for i from 0<=i<2:
            for j from 0<=j<2:
                for k from 0<=k<2:

                    oct_id = k+2*j+4*i

                    if new_indices[oct_id].empty():
                        del new_indices[oct_id]
                        continue

                    xmin_new[0] = xmin[0] + (i - eps)*length/2
                    xmin_new[1] = xmin[1] + (j - eps)*length/2
                    xmin_new[2] = xmin[2] + (k - eps)*length/2

                    eps_new = 2*self._get_eps(length_padded, xmin_new)

                    node.children[oct_id] = self._new_node(xmin_new, length_padded,
                            hmax=hmax_children[oct_id], level=level+1, parent=node)

                    depth_child = self._c_build_tree(pa, new_indices[oct_id],
                            xmin_new, length_padded, node.children[oct_id], level+1)

                    depth_max = <int>fmax(depth_max, depth_child)

        return 1 + depth_max
    
    @cython.cdivision(True)
    @cython.boundscheck(False) 
    cdef int _c_build_tree_level1(self, NNPSParticleArrayWrapper pa, double* xmin, double length,
            cOctreeNode* node, int num_threads) nogil:

        cdef double* src_x_ptr = pa.x.data
        cdef double* src_y_ptr = pa.y.data
        cdef double* src_z_ptr = pa.z.data
        cdef double* src_h_ptr = pa.h.data

        cdef double xmin_new[3]
        cdef double hmax_children[8]
        cdef cOctreeNode* new_node
        cdef int i, j, k, c, tid, oct_cid, oct_id, count_thread, n, p
        cdef u_int  q
        for i from 0<=i<8:
            hmax_children[i] = 0

        # This is required to fix floating point errors. One such case
        # is mentioned in pysph.base.tests.test_octree
        cdef double eps = 2*self._get_eps(length, xmin)
        n =  node.num_particles

        if (n < self.leaf_max_particles) or (eps > EPS_MAX):
            for i from 0<=i<n:
                self.pids[i] = i
            node.is_leaf = True
            return 1

        cdef u_int* p_indices = self.pids
        cdef vector[u_int]* child_indices = new vector[u_int](n)

        # Number of children of a thread belonging to octant, cumulative_map[thread_id][octant_id]
        cdef vector[vector[int]] cumulative_map = vector[vector[int]](num_threads)
        cdef vector[vector[double]] threads_hmax = vector[vector[double]](num_threads)
        cdef vector[int] count = vector[int](8)

        for i from 0<=i<num_threads:
            cumulative_map[i] = vector[int](8)
            threads_hmax[i] = vector[double](8)

        with nogil, parallel():
            # Required to ensure cython treats i,j,k as private-variables for each thread
            i = j = k = -1
            for p in prange(n,schedule='static'):
                tid = threadid()
                find_cell_id_raw(
                        src_x_ptr[p] - xmin[0],
                        src_y_ptr[p] - xmin[1],
                        src_z_ptr[p] - xmin[2],
                        length/2,
                        &i, &j, &k
                        )

                oct_id = k+2*j+4*i
                deref(child_indices)[p] = oct_id
                cumulative_map[tid][oct_id] += 1
                threads_hmax[tid][oct_id] = fmax(threads_hmax[tid][oct_id], src_h_ptr[p])

        for oct_id from 0<=oct_id<8:
            for tid from 0<=tid<num_threads:
                count_thread = cumulative_map[tid][oct_id]
                cumulative_map[tid][oct_id] = count[oct_id]
                count[oct_id] += count_thread
                hmax_children[oct_id] = fmax(threads_hmax[tid][oct_id], hmax_children[oct_id])

        cdef double length_padded = (length/2)*(1 + 2*eps)
        cdef vector[cOctreeNode *]* next_level_nodes = new vector[node_ptr]()

        c = 0
        for i from 0<=i<2:
            for j from 0<=j<2:
                for k from 0<=k<2:

                    oct_id = k+2*j+4*i

                    if (count[oct_id] == 0):
                        continue

                    xmin_new[0] = xmin[0] + (i - eps)*length/2
                    xmin_new[1] = xmin[1] + (j - eps)*length/2
                    xmin_new[2] = xmin[2] + (k - eps)*length/2

                    eps_new = 2*self._get_eps(length_padded, xmin_new)

                    new_node = self._new_node(xmin_new, length_padded,
                            hmax=hmax_children[oct_id], level=1, parent=node)
                    node.children[oct_id] = new_node
                    new_node.start_index = c
                    new_node.num_particles = count[oct_id]
                    count[oct_id] = c
                    c = c + new_node.num_particles
                    if (new_node.num_particles < self.leaf_max_particles) or (eps_new > EPS_MAX):
                        new_node.is_leaf = True
                        continue
                    next_level_nodes.push_back(new_node)

        # Assign p_indices directly according to which child the index belongs to and
        # where each child of root starts
        with nogil, parallel():
            for p in prange(n, schedule='static'):
                tid = threadid()
                oct_id = deref(child_indices)[p]
                oct_cid = count[oct_id] + cumulative_map[tid][oct_id]
                p_indices[oct_cid] = p
                cumulative_map[tid][oct_id] += 1

        del child_indices

        if (next_level_nodes.empty()):
            del next_level_nodes
            return 2
        else:
            return self._c_build_tree_bfs(pa, p_indices, next_level_nodes, 1, num_threads)

    @cython.cdivision(True)
    @cython.boundscheck(False) 
    cdef int _c_build_tree_bfs(self, NNPSParticleArrayWrapper pa, u_int* p_indices,
             vector[cOctreeNode *]* level_nodes,
            int level, int num_threads) nogil:
        cdef double* src_x_ptr = pa.x.data
        cdef double* src_y_ptr = pa.y.data
        cdef double* src_z_ptr = pa.z.data
        cdef double* src_h_ptr = pa.h.data

        cdef double eps, eps_new, length_padded, length
        cdef cOctreeNode* node
        cdef int i, j, k, n, num_nodes, l, start, num_p, oct_id, sid, jid, tid, r, d, children
        cdef u_int p, q
        cdef vector[double] hmax_children
        cdef vector[vector[u_int]] new_indices
        cdef double* xmin_new
        cdef double* xmin

        cdef vector[vector[node_ptr]] old_nodes = vector[vector[node_ptr]](num_threads)
        cdef vector[vector[node_ptr]] new_nodes = vector[vector[node_ptr]](num_threads)

        for i from 0<=i<num_threads:
            old_nodes[i] = vector[node_ptr]()
            new_nodes[i] = vector[node_ptr]()

        cdef vector[int] count = vector[int](num_threads+2)
        d = 0
        num_nodes = level_nodes.size()
        for i from 0<=i<num_nodes:
            count[d] = count[d] + 1
            old_nodes[d].push_back(deref(level_nodes)[i])
            d = (d + 1)%num_threads

        # Stores the total nodes that need to be operated on
        count[num_threads] = num_nodes
        # Stores the depth
        count[num_threads + 1] = 2

        del level_nodes

        with nogil, parallel():
            xmin_new = <double *> malloc(3 * sizeof(double))
            xmin = <double *> malloc(3 * sizeof(double))
            hmax_children = vector[double](8)
            new_indices = vector[vector[u_int]](8)

            tid = threadid()

            while(count[num_threads] > 0):
                for i from 0<=i<8:
                    new_indices[i] = vector[u_int]()
                children = 0

                for n from 0<=n<old_nodes[tid].size():
                    for i from 0<=i<8:
                        new_indices[i].clear()
                        hmax_children[i] = 0
                    node = old_nodes[tid][n]
                    length = node.length
                    start = node.start_index
                    num_p = node.num_particles
                    level = node.level
                    xmin[0] = node.xmin[0]
                    xmin[1] = node.xmin[1]
                    xmin[2] = node.xmin[2]

                    for p from start<=p<(start + num_p):
                        q = p_indices[p]
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

                    eps = 2*self._get_eps(length, xmin)
                    length_padded = (length/2)*(1 + 2*eps)


                    for i from 0<=i<2:
                        for j from 0<=j<2:
                            for k from 0<=k<2:
                                oct_id = k+2*j+4*i

                                if new_indices[oct_id].empty():
                                    continue

                                xmin_new[0] = xmin[0] + (i - eps)*length/2
                                xmin_new[1] = xmin[1] + (j - eps)*length/2
                                xmin_new[2] = xmin[2] + (k - eps)*length/2

                                node.children[oct_id] = self._new_node(xmin_new, length_padded,
                                        hmax=hmax_children[oct_id], level=level+1, parent=node)
                                node.children[oct_id].start_index = start
                                num_p = new_indices[oct_id].size()
                                node.children[oct_id].num_particles = num_p

                                # Change the position of the indices in p_indices
                                # according to which child they belong to
                                for l from 0<=l<num_p:
                                    p_indices[start+l] = new_indices[oct_id][l]
                                start = start + new_indices[oct_id].size()

                                eps_new = 2*self._get_eps(length_padded, xmin_new)

                                if (num_p < self.leaf_max_particles) or (eps_new > EPS_MAX):
                                    node.children[oct_id].is_leaf = True
                                    continue
                                new_nodes[tid].push_back(node.children[oct_id])
                                children = children + 1

                count[tid] = children
                old_nodes[tid].clear()

                START_OMP_BARRIER_PRAGMA()
                END_OMP_PRAGMA()

                # Load Balancing is done by single thread

                START_OMP_SINGLE_PRAGMA()
                count[num_threads + 1] = count[num_threads + 1] + 1
                count[num_threads] = 0
                for sid from 0<=sid<num_threads:
                    count[num_threads]  = count[num_threads] + count[sid]
                d =  int(count[num_threads]/num_threads)
                r = count[num_threads]%num_threads
                for sid from 0<=sid<r:
                    count[sid] = d + 1
                for sid from r<=sid<num_threads:
                    count[sid] = d
                END_OMP_PRAGMA()

                old_nodes[tid] = vector[node_ptr](count[tid])

                START_OMP_BARRIER_PRAGMA()
                END_OMP_PRAGMA()

                START_OMP_SINGLE_PRAGMA()
                d = 0
                r = 0
                for sid from 0<=sid<num_threads:
                    for jid from 0<=jid<new_nodes[sid].size():
                        old_nodes[d][r] = new_nodes[sid][jid]
                        r += 1
                        if (r == count[d]):
                            d = d + 1
                            r = 0
                END_OMP_PRAGMA()

                new_nodes[tid].clear()

            free(xmin)
            free(xmin_new)

        return count[num_threads + 1]


    cdef void _plot_tree(self, OctreeNode node, ax):
        node.plot(ax)

        cdef OctreeNode child
        cdef list children = node.get_children()

        for child in children:
            if child == None:
                continue
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

    cdef int c_build_tree(self, NNPSParticleArrayWrapper pa_wrapper, bint test_parallel = False):

        self._calculate_domain(pa_wrapper)

        cdef int num_particles = pa_wrapper.get_number_of_particles()
        self.num_particles = num_particles

        cdef int i

        cdef vector[u_int]* indices_ptr = new vector[u_int]()

        if self.root != NULL:
            self._delete_tree(self.root)
        if self.pids != NULL:
            free(self.pids)
        if self.leaf_cells != NULL:
            del self.leaf_cells

        self.pids = <u_int*> malloc(num_particles*sizeof(u_int))

        self.root = self._new_node(self.xmin, self.length,
                hmax=self.hmax, level=0)

        cdef int num_threads = get_number_of_threads()

        # Use the parallel method
        if ( (num_threads > 1) or test_parallel ):
            self.root.start_index = 0
            self.root.num_particles = num_particles
            self.depth = self._c_build_tree_level1(pa_wrapper, self.root.xmin,
                    self.root.length, self.root, num_threads)

        # Use the serial method
        else:
            self._next_pid = 0
            for i from 0<=i<num_particles:
                indices_ptr.push_back(i)
            self.depth = self._c_build_tree(pa_wrapper, indices_ptr, self.root.xmin,
                self.root.length, self.root, 0)

        return self.depth

    cdef void c_get_leaf_cells(self):
        if self.leaf_cells != NULL:
            return

        self.leaf_cells = new vector[cOctreeNode*]()
        self._c_get_leaf_cells(self.root)

    @cython.cdivision(True)
    cdef cOctreeNode* c_find_point(self, double x, double y, double z):
        cdef cOctreeNode* node = self.root
        cdef cOctreeNode* prev = self.root

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

    cpdef int build_tree(self, ParticleArray pa, bint test_parallel = False):
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
        return self.c_build_tree(pa_wrapper, test_parallel)

    cpdef delete_tree(self):
        """ Delete tree"""
        if self.root != NULL:
            self._delete_tree(self.root)
        if self.leaf_cells != NULL:
            del self.leaf_cells
        self.root = NULL
        self.leaf_cells = NULL

    cpdef OctreeNode get_root(self):
        """ Get root of the tree

        Returns
        -------

        root : OctreeNode
        Root of the tree

        """
        cdef OctreeNode py_node = OctreeNode()
        py_node.wrap_node(self.root)
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
        """ Get the leaf node to which a point belongs

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
        """ Plots the tree

        Parameters
        ----------

        ax : mpl_toolkits.mplot3d.Axes3D instance

        """
        cdef OctreeNode root = self.get_root()
        self._plot_tree(root, ax)


cdef class CompressedOctree(Octree):
    def __init__(self, int leaf_max_particles):
        Octree.__init__(self, leaf_max_particles)
        self.dbl_max = np.finfo(float).max

    @cython.cdivision(True)
    cdef int _c_build_tree(self, NNPSParticleArrayWrapper pa,
            vector[u_int]* indices, double* xmin, double length,
            cOctreeNode* node, int level) nogil:

        cdef double* src_x_ptr = pa.x.data
        cdef double* src_y_ptr = pa.y.data
        cdef double* src_z_ptr = pa.z.data
        cdef double* src_h_ptr = pa.h.data

        cdef double xmin_new[8][3]
        cdef double xmax_new[8][3]
        cdef double length_new = 0
        cdef double hmax_children[8]

        cdef int depth_child = 0
        cdef int depth_max = 0

        cdef int i, j, k
        cdef u_int p, q

        for i from 0<=i<8:
            hmax_children[i] = 0
            for j from 0<=j<3:
                xmin_new[i][j] = self.dbl_max
                xmax_new[i][j] = -self.dbl_max

        cdef int oct_id

        if (indices.size() < self.leaf_max_particles):
            copy(indices.begin(), indices.end(), self.pids + self._next_pid)
            node.start_index = self._next_pid
            self._next_pid += indices.size()
            node.num_particles = indices.size()
            del indices
            node.is_leaf = True
            return 1

        cdef vector[u_int]* new_indices[8]
        for i from 0<=i<8:
            new_indices[i] = new vector[u_int]()

        cdef double* xmin_current
        cdef double* xmax_current

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

            xmin_current = xmin_new[oct_id]
            xmax_current = xmax_new[oct_id]

            hmax_children[oct_id] = fmax(hmax_children[oct_id], src_h_ptr[q])
            xmin_current[0] = fmin(xmin_current[0], src_x_ptr[q])
            xmin_current[1] = fmin(xmin_current[1], src_y_ptr[q])
            xmin_current[2] = fmin(xmin_current[2], src_z_ptr[q])

            xmax_current[0] = fmax(xmax_current[0], src_x_ptr[q])
            xmax_current[1] = fmax(xmax_current[1], src_y_ptr[q])
            xmax_current[2] = fmax(xmax_current[2], src_z_ptr[q])

        cdef double x_length, y_length, z_length
        cdef double length_padded
        cdef double eps

        del indices

        for i from 0<=i<8:
            if new_indices[i].empty():
                del new_indices[i]
                continue

            xmin_current = xmin_new[i]
            xmax_current = xmax_new[i]

            x_length = xmax_current[0] - xmin_current[0]
            y_length = xmax_current[1] - xmin_current[1]
            z_length = xmax_current[2] - xmin_current[2]

            length_new = fmax(x_length, fmax(y_length, z_length))

            eps = self._get_eps(length_new, xmin_current)

            length_padded = length_new*(1 + 2*eps)

            xmin_current[0] -= length_new*eps
            xmin_current[1] -= length_new*eps
            xmin_current[2] -= length_new*eps

            node.children[i] = self._new_node(xmin_current, length_padded,
                    hmax=hmax_children[i], level=level+1, parent=node)

            depth_child = self._c_build_tree(pa, new_indices[i],
                    xmin_current, length_padded, node.children[i], level+1)

            depth_max = <int>fmax(depth_max, depth_child)

        return 1 + depth_max

    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef int _c_build_tree_level1(self, NNPSParticleArrayWrapper pa, double* xmin, double length,
            cOctreeNode* node, int num_threads) nogil:

        cdef double* src_x_ptr = pa.x.data
        cdef double* src_y_ptr = pa.y.data
        cdef double* src_z_ptr = pa.z.data
        cdef double* src_h_ptr = pa.h.data

        cdef double xmin_new[8][3]
        cdef double xmax_new[8][3]
        cdef double hmax_children[8]
        cdef cOctreeNode* new_node
        cdef int i, j, k, c, tid, oct_cid, oct_id, count_thread, n, p
        cdef u_int  q

        for i from 0<=i<8:
            hmax_children[i] = 0
            for j from 0<=j<3:
                xmin_new[i][j] = self.dbl_max
                xmax_new[i][j] = -self.dbl_max


        cdef double eps
        n =  node.num_particles

        if (n < self.leaf_max_particles):
            for i from 0<=i<n:
                self.pids[i] = i
            node.is_leaf = True
            return 1

        cdef u_int* p_indices = self.pids
        cdef vector[u_int]* child_indices = new vector[u_int](n)

        # Number of children of a thread belonging to octant, cumulative_map[thread_id][octant_id]
        cdef vector[vector[int]] cumulative_map = vector[vector[int]](num_threads)
        cdef vector[vector[double]] threads_hmax = vector[vector[double]](num_threads)

        cdef vector[vector[double]] threads_xmin = vector[vector[double]](num_threads)
        cdef vector[vector[double]] threads_ymin = vector[vector[double]](num_threads)
        cdef vector[vector[double]] threads_zmin = vector[vector[double]](num_threads)

        cdef vector[vector[double]] threads_xmax = vector[vector[double]](num_threads)
        cdef vector[vector[double]] threads_ymax = vector[vector[double]](num_threads)
        cdef vector[vector[double]] threads_zmax = vector[vector[double]](num_threads)

        cdef vector[int] count = vector[int](8)

        for i from 0<=i<num_threads:
            cumulative_map[i] = vector[int](8)
            threads_hmax[i] = vector[double](8)

            threads_xmin[i] = vector[double](8)
            threads_ymin[i] = vector[double](8)
            threads_zmin[i] = vector[double](8)

            threads_xmax[i] = vector[double](8)
            threads_ymax[i] = vector[double](8)
            threads_zmax[i] = vector[double](8)

            for j from 0<=j<8:
                threads_xmin[i][j] = self.dbl_max
                threads_ymin[i][j] = self.dbl_max
                threads_zmin[i][j] = self.dbl_max

                threads_xmax[i][j] = -self.dbl_max
                threads_ymax[i][j] = -self.dbl_max
                threads_zmax[i][j] = -self.dbl_max

        with nogil, parallel():
            # Required to ensure cython treats i,j,k as private-variables for each thread
            i = j = k = -1
            for p in prange(n,schedule='static'):
                tid = threadid()
                find_cell_id_raw(
                        src_x_ptr[p] - xmin[0],
                        src_y_ptr[p] - xmin[1],
                        src_z_ptr[p] - xmin[2],
                        length/2,
                        &i, &j, &k
                        )

                oct_id = k+2*j+4*i
                deref(child_indices)[p] = oct_id
                cumulative_map[tid][oct_id] += 1
                threads_hmax[tid][oct_id] = fmax(threads_hmax[tid][oct_id], src_h_ptr[p])

                threads_xmin[tid][oct_id] = fmin(threads_xmin[tid][oct_id], src_x_ptr[p])
                threads_ymin[tid][oct_id] = fmin(threads_ymin[tid][oct_id], src_y_ptr[p])
                threads_zmin[tid][oct_id] = fmin(threads_zmin[tid][oct_id], src_z_ptr[p])

                threads_xmax[tid][oct_id] = fmax(threads_xmax[tid][oct_id], src_x_ptr[p])
                threads_ymax[tid][oct_id] = fmax(threads_ymax[tid][oct_id], src_y_ptr[p])
                threads_zmax[tid][oct_id] = fmax(threads_zmax[tid][oct_id], src_z_ptr[p])

        for oct_id from 0<=oct_id<8:
            for tid from 0<=tid<num_threads:
                count_thread = cumulative_map[tid][oct_id]
                cumulative_map[tid][oct_id] = count[oct_id]
                count[oct_id] += count_thread
                hmax_children[oct_id] = fmax(threads_hmax[tid][oct_id], hmax_children[oct_id])

                xmin_new[oct_id][0] = fmin(threads_xmin[tid][oct_id], xmin_new[oct_id][0])
                xmin_new[oct_id][1] = fmin(threads_ymin[tid][oct_id], xmin_new[oct_id][1])
                xmin_new[oct_id][2] = fmin(threads_zmin[tid][oct_id], xmin_new[oct_id][2])

                xmax_new[oct_id][0] = fmax(threads_xmax[tid][oct_id], xmax_new[oct_id][0])
                xmax_new[oct_id][1] = fmax(threads_ymax[tid][oct_id], xmax_new[oct_id][1])
                xmax_new[oct_id][2] = fmax(threads_zmax[tid][oct_id], xmax_new[oct_id][2])

        cdef double x_length, y_length, z_length, length_new, length_padded
        cdef vector[cOctreeNode *]* next_level_nodes = new vector[node_ptr]()

        cdef double* xmin_current
        cdef double* xmax_current
        c = 0
        for oct_id from 0<=oct_id<8:
            if (count[oct_id] == 0):
                continue

            xmin_current = xmin_new[oct_id]
            xmax_current = xmax_new[oct_id]

            x_length = xmax_current[0] - xmin_current[0]
            y_length = xmax_current[1] - xmin_current[1]
            z_length = xmax_current[2] - xmin_current[2]

            length_new = fmax(x_length, fmax(y_length, z_length))

            eps = self._get_eps(length_new, xmin_current)

            length_padded = length_new*(1 + 2*eps)

            xmin_current[0] -= length_new*eps
            xmin_current[1] -= length_new*eps
            xmin_current[2] -= length_new*eps

            new_node = self._new_node(xmin_current, length_padded,
                    hmax=hmax_children[oct_id], level=1, parent=node)
            node.children[oct_id] = new_node
            new_node.start_index = c
            new_node.num_particles = count[oct_id]
            count[oct_id] = c
            c = c + new_node.num_particles
            if (new_node.num_particles < self.leaf_max_particles):
                new_node.is_leaf = True
                continue
            next_level_nodes.push_back(new_node)

        # Assign p_indices directly according to which child the index belongs to and
        # where each child of root starts
        with nogil, parallel():
            for p in prange(n, schedule='static'):
                tid = threadid()
                oct_id = deref(child_indices)[p]
                oct_cid = count[oct_id] + cumulative_map[tid][oct_id]
                p_indices[oct_cid] = p
                cumulative_map[tid][oct_id] += 1

        del child_indices

        if (next_level_nodes.empty()):
            del next_level_nodes
            return 2
        else:
            return self._c_build_tree_bfs(pa, p_indices, next_level_nodes, 1, num_threads)


    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef int _c_build_tree_bfs(self, NNPSParticleArrayWrapper pa, u_int* p_indices,
             vector[cOctreeNode *]* level_nodes, int level, int num_threads) nogil:

        cdef double* src_x_ptr = pa.x.data
        cdef double* src_y_ptr = pa.y.data
        cdef double* src_z_ptr = pa.z.data
        cdef double* src_h_ptr = pa.h.data

        cdef double eps, length_padded, length, x_length, y_length, z_length, length_new
        cdef cOctreeNode* node
        cdef int i, j, k, n, num_nodes, l, start, num_p, oct_id, sid, jid, tid, r, d, children
        cdef u_int p, q
        cdef vector[double] hmax_children
        cdef vector[vector[u_int]] new_indices
        cdef double *xmin
        cdef vector[dbl_ptr] xmax_new
        cdef vector[dbl_ptr] xmin_new
        cdef double *xmin_current
        cdef double *xmax_current

        cdef vector[vector[node_ptr]] old_nodes = vector[vector[node_ptr]](num_threads)
        cdef vector[vector[node_ptr]] new_nodes = vector[vector[node_ptr]](num_threads)

        for i from 0<=i<num_threads:
            old_nodes[i] = vector[node_ptr]()
            new_nodes[i] = vector[node_ptr]()

        cdef vector[int] count = vector[int](num_threads+2)
        d = 0
        num_nodes = level_nodes.size()
        for i from 0<=i<num_nodes:
            count[d] = count[d] + 1
            old_nodes[d].push_back(deref(level_nodes)[i])
            d = (d + 1)%num_threads
        count[num_threads] = num_nodes
        count[num_threads + 1] = 2

        del level_nodes

        with nogil, parallel():
            xmin = <double *> malloc(3 * sizeof(double))
            hmax_children = vector[double](8)
            new_indices = vector[vector[u_int]](8)
            xmin_new = vector[dbl_ptr](8)
            xmax_new = vector[dbl_ptr](8)

            for i from 0<=i<8:
                xmin_new[i] = <double *> malloc(3 * sizeof(double))
                xmax_new[i] = <double *> malloc(3 * sizeof(double))

            tid = threadid()

            while(count[num_threads]>0):
                for i from 0<=i<8:
                    new_indices[i] = vector[u_int]()
                children = 0

                for n from 0<=n<old_nodes[tid].size():
                    for i from 0<=i<8:
                        new_indices[i].clear()
                        hmax_children[i] = 0
                        for j from 0<=j<3:
                            xmin_new[i][j] = self.dbl_max
                            xmax_new[i][j] = -self.dbl_max

                    node = old_nodes[tid][n]
                    length = node.length
                    start = node.start_index
                    num_p = node.num_particles
                    level = node.level
                    xmin[0] = node.xmin[0]
                    xmin[1] = node.xmin[1]
                    xmin[2] = node.xmin[2]

                    for p from start<=p<(start + num_p):
                        q = p_indices[p]
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

                        xmin_current = xmin_new[oct_id]
                        xmax_current = xmax_new[oct_id]

                        xmin_current[0] = fmin(xmin_current[0], src_x_ptr[q])
                        xmin_current[1] = fmin(xmin_current[1], src_y_ptr[q])
                        xmin_current[2] = fmin(xmin_current[2], src_z_ptr[q])

                        xmax_current[0] = fmax(xmax_current[0], src_x_ptr[q])
                        xmax_current[1] = fmax(xmax_current[1], src_y_ptr[q])
                        xmax_current[2] = fmax(xmax_current[2], src_z_ptr[q])

                    for oct_id from 0<=oct_id<8:
                        if new_indices[oct_id].empty():
                            continue

                        xmin_current = xmin_new[oct_id]
                        xmax_current = xmax_new[oct_id]

                        x_length = xmax_current[0] - xmin_current[0]
                        y_length = xmax_current[1] - xmin_current[1]
                        z_length = xmax_current[2] - xmin_current[2]

                        length_new = fmax(x_length, fmax(y_length, z_length))
                        eps = 2*self._get_eps(length_new, xmin_current)
                        length_padded = length_new*(1 + 2*eps)

                        xmin_current[0] -= length_new*eps
                        xmin_current[1] -= length_new*eps
                        xmin_current[2] -= length_new*eps

                        node.children[oct_id] = self._new_node(xmin_current, length_padded,
                                hmax=hmax_children[oct_id], level=level+1, parent=node)

                        node.children[oct_id].start_index = start
                        num_p = new_indices[oct_id].size()
                        node.children[oct_id].num_particles = num_p

                        # Change the position of the indices in p_indices
                        # according to which child they belong to
                        for l from 0<=l<num_p:
                            p_indices[start+l] = new_indices[oct_id][l]
                        start = start + new_indices[oct_id].size()

                        if (num_p < self.leaf_max_particles):
                            node.children[oct_id].is_leaf = True
                            continue
                        new_nodes[tid].push_back(node.children[oct_id])
                        children = children + 1

                count[tid] = children
                old_nodes[tid].clear()

                START_OMP_BARRIER_PRAGMA()
                END_OMP_PRAGMA()

                # Load Balancing is done by single thread

                START_OMP_SINGLE_PRAGMA()
                count[num_threads + 1] = count[num_threads + 1] + 1
                count[num_threads] = 0
                for sid from 0<=sid<num_threads:
                    count[num_threads]  = count[num_threads] + count[sid]
                d =  int(count[num_threads]/num_threads)
                r = count[num_threads]%num_threads
                for sid from 0<=sid<r:
                    count[sid] = d + 1
                for sid from r<=sid<num_threads:
                    count[sid] = d
                END_OMP_PRAGMA()

                old_nodes[tid] = vector[node_ptr](count[tid])

                START_OMP_BARRIER_PRAGMA()
                END_OMP_PRAGMA()

                START_OMP_SINGLE_PRAGMA()
                d = 0
                r = 0
                for sid from 0<=sid<num_threads:
                    for jid from 0<=jid<new_nodes[sid].size():
                        old_nodes[d][r] = new_nodes[sid][jid]
                        r += 1
                        if (r == count[d]):
                            d = d + 1
                            r = 0
                END_OMP_PRAGMA()

                new_nodes[tid].clear()

            free(xmin)
            for i from 0<=i<8:
                free(xmin_new[i])
                free(xmax_new[i])

        return count[num_threads + 1]

    #### Public protocol ################################################

    cdef int c_build_tree(self, NNPSParticleArrayWrapper pa_wrapper, bint test_parallel = False):

        self._calculate_domain(pa_wrapper)

        cdef int num_particles = pa_wrapper.get_number_of_particles()
        self.num_particles = num_particles

        cdef int i

        cdef vector[u_int]* indices_ptr = new vector[u_int]()

        if self.root != NULL:
            self._delete_tree(self.root)
        if self.pids != NULL:
            free(self.pids)
        if self.leaf_cells != NULL:
            del self.leaf_cells

        self.pids = <u_int*> malloc(num_particles*sizeof(u_int))

        self.root = self._new_node(self.xmin, self.length,
                hmax=self.hmax, level=0)

        cdef int num_threads = get_number_of_threads()

        # Use the parallel method
        if ( (num_threads > 2) or test_parallel ):
            self.root.start_index = 0
            self.root.num_particles = num_particles
            self.depth = self._c_build_tree_level1(pa_wrapper, self.root.xmin,
                    self.root.length, self.root, num_threads)

        # Use the serial method
        else:
            self._next_pid = 0
            for i from 0<=i<num_particles:
                indices_ptr.push_back(i)
            self.depth = self._c_build_tree(pa_wrapper, indices_ptr, self.root.xmin,
                self.root.length, self.root, 0)

        return self.depth
