import os
import numpy as np
from pytools import memoize

import pyopencl as cl
import pyopencl.cltypes
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.scan import GenericScanKernel

from compyle.opencl import get_context, get_queue, named_profile
from compyle.array import Array
from pysph.base.tree.helpers import get_vector_dtype, get_helper

NODE_KERNEL_TEMPLATE = r"""
uint node_idx = i;
%(setup)s;
int child_offset = offsets[node_idx];
if (child_offset == -1) {
    uint2 pbound = pbounds[node_idx];
    if (pbound.s0 < pbound.s1) {
        %(leaf_operation)s;
    }
} else {
    %(node_operation)s;
}
%(output_expr)s;
"""

LEAF_DFS_TEMPLATE = r"""
    /*
     * Owner of properties -
     * dst tree: cids, unique_cids
     * src tree: offsets
     */
   int cid_dst = unique_cids[i];

    /*
     * Assuming max depth of 21
     * stack_idx is also equal to current layer of tree
     * child_idx = number of children iterated through
     * idx_stack = current node
     */
    char child_stack[%(max_depth)s];
    int cid_stack[%(max_depth)s];

    char idx = 0;
    child_stack[0] = 0;
    cid_stack[0] = 1;
    char flag;
    int cid_src;
    int child_offset;

    %(setup)s;
    while (idx >= 0) {

        // Recurse to find either leaf node or invalid node
        cid_src = cid_stack[idx];

        child_offset = offsets[cid_src];
        %(common_operation)s;

        while (child_offset != -1) {
            %(node_operation)s;

            idx++;
            cid_src = child_offset;
            cid_stack[idx] = cid_src;
            child_stack[idx] = 0;
            child_offset = offsets[cid_src];
        }

        if (child_offset == -1) {
            %(leaf_operation)s;
        }

        // Recurse back to find node with a valid neighbor
        while (child_stack[idx] >= %(k)s-1 && idx >= 0)
            idx--;

        // Iterate to next neighbor
        if (idx >= 0) {
            cid_stack[idx]++;
            child_stack[idx]++;
        }
    }

    %(output_expr)s;
"""

POINT_DFS_TEMPLATE = r"""
    /*
     * Owner of properties -
     * dst tree: cids, unique_cids
     * src tree: offsets
     */
   int cid_dst = cids[i];

    /*
     * Assuming max depth of 21
     * stack_idx is also equal to current layer of tree
     * child_idx = number of children iterated through
     * idx_stack = current node
     */
    char child_stack[%(max_depth)s];
    int cid_stack[%(max_depth)s];

    char idx = 0;
    child_stack[0] = 0;
    cid_stack[0] = 1;
    char flag;
    int cid_src;
    int child_offset;

    %(setup)s;
    while (idx >= 0) {

        // Recurse to find either leaf node or invalid node
        cid_src = cid_stack[idx];

        child_offset = offsets[cid_src];
        %(common_operation)s;

        while (child_offset != -1) {
            %(node_operation)s;

            idx++;
            cid_src = child_offset;
            cid_stack[idx] = cid_src;
            child_stack[idx] = 0;
            child_offset = offsets[cid_src];
        }

        if (child_offset == -1) {
            %(leaf_operation)s;
        }

        // Recurse back to find node with a valid neighbor
        while (child_stack[idx] >= %(k)s-1 && idx >= 0)
            idx--;

        // Iterate to next neighbor
        if (idx >= 0) {
            cid_stack[idx]++;
            child_stack[idx]++;
        }
    }

    %(output_expr)s;
"""


def get_M_array_initialization(k):
    result = "uint%(k)s constant M[%(k)s]  = {" % dict(k=k)
    for i in range(k):
        result += "(uint%(k)s)(%(init_value)s)," % dict(
            k=k, init_value=','.join((np.arange(k) >= i).astype(
                int).astype(str))
        )
    result += "};"
    return result


# This segmented scan is the core of the tree construction algorithm.
# Please look at a few segmented scan examples here if you haven't already
# https://en.wikipedia.org/wiki/Segmented_scan
#
# This scan generates an array `v` of k-dimensional vectors where v[i][j]
# corresponds to how many particles are in the same node as particle `i`,
# are ordered before it and are going to lie in childs 0..j.
#
# For example, consider the first layer. In this case v[23][4] gives the
# particles numbered from 0 to 23 which are going to lie in children 0 to 4
# of the root node. The segment flag bits just let us extend this to further
# layers.
#
# The array of vectors M are just a math trick. M[j] gives a vector with j
# zeros and k - j ones (With k = 4, M[1] = {0, 1, 1, 1}). Adding this up in
# the prefix sum directly gives us the required result.
@named_profile('particle_reordering', backend='opencl')
@memoize
def _get_particle_kernel(ctx, k, args, index_code):
    return GenericScanKernel(
        ctx, get_vector_dtype('uint', k), neutral="0",
        arguments=r"""__global char *seg_flag,
                    __global uint%(k)s *prefix_sum_vector,
                    """ % dict(k=k) + args,
        input_expr="M[%(index_code)s]" % dict(index_code=index_code),
        scan_expr="(across_seg_boundary ? b : a + b)",
        is_segment_start_expr="seg_flag[i]",
        output_statement=r"""prefix_sum_vector[i]=item;""",
        preamble=get_M_array_initialization(k)
    )


# The offset of a node's child is given by:
# offset of first child in next layer + k * (number of non-leaf nodes before
# given node).
# If the node is a leaf, we set this value to be -1.
@named_profile('set_offset', backend='opencl')
@memoize
def _get_set_offset_kernel(ctx, k, leaf_size):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""__global uint2 *pbounds, __global uint *offsets,
                      __global int *leaf_count, int csum_nodes_next""",
        input_expr="(pbounds[i].s1 - pbounds[i].s0 > %(leaf_size)s)" % {
            'leaf_size': leaf_size},
        scan_expr="a + b",
        output_statement=r"""{
            offsets[i] = ((pbounds[i].s1 - pbounds[i].s0 > %(leaf_size)s) ?
                           csum_nodes_next + (%(k)s * (item - 1)) : -1);
            if (i == N - 1) { *leaf_count = (N - item); }
        }""" % {'leaf_size': leaf_size, 'k': k}
    )


# Each particle belongs to a given leaf / last layer node and this cell is
# indexed by the cid array.
# The unique_cids algorithm
# 1) unique_cids_map: unique_cids_map[i] gives the number of unique_cids
#                     before index i.
# 2) unique_cids: List of unique cids.
# 3) unique_cids_count: Number of unique cids
#
# Note that unique_cids also gives us the list of leaves / last layer nodes
# which are not empty.
@named_profile('unique_cids', backend='opencl')
@memoize
def _get_unique_cids_kernel(ctx):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""int *cids, int *unique_cids_map,
                int *unique_cids, int *unique_cids_count""",
        input_expr="(i == 0 || cids[i] != cids[i-1])",
        scan_expr="a + b",
        output_statement=r"""
            if (item != prev_item) {
                unique_cids[item - 1] = cids[i];
            }
            unique_cids_map[i] = item - 1;
            if (i == N - 1) *unique_cids_count = item;
        """
    )


# A lot of leaves are going to be empty. Not really sure if this guy is of
# any use.
@named_profile('leaves', backend='opencl')
@memoize
def _get_leaves_kernel(ctx, leaf_size):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments="int *offsets, uint2 pbounds, int *leaf_cids, "
                  "int *num_leaves",
        input_expr="(pbounds[i].s1 - pbounds[i].s0 <= %(leaf_size)s)" % dict(
            leaf_size=leaf_size
        ),
        scan_expr="a+b",
        output_statement=r"""
            if (item != prev_item) {
                leaf_cids[item - 1] = i;
            }
            if (i == N - 1) *num_leaves = item;
        """
    )


@named_profile("group_cids", backend='opencl')
@memoize
def _get_cid_groups_kernel(ctx):
    return GenericScanKernel(
        ctx, np.uint32, neutral="0",
        arguments="""int *unique_cids, uint2 *pbounds,
            int *group_cids, int *group_count, int gmin, int gmax""",
        input_expr="pass(pbounds[unique_cids[i]], gmin, gmax)",
        scan_expr="(a + b)",
        output_statement=r"""
        if (item != prev_item) {
            group_cids[item - 1] = unique_cids[i];
        }
        if (i == N - 1) *group_count = item;
        """,
        preamble="""
        char pass(uint2 pbound, int gmin, int gmax) {
            int leaf_size = pbound.s1 - pbound.s0;
            return (leaf_size > gmin && leaf_size <= gmax);
        }
        """
    )


@memoize
def tree_bottom_up(ctx, args, setup, leaf_operation, node_operation,
                   output_expr, preamble=""):
    operation = NODE_KERNEL_TEMPLATE % dict(
        setup=setup,
        leaf_operation=leaf_operation,
        node_operation=node_operation,
        output_expr=output_expr
    )

    args = ', '.join(["int *offsets, uint2 *pbounds", args])

    kernel = ElementwiseKernel(ctx, args, operation=operation,
                               preamble=preamble)

    def callable(tree, *args):
        csum_nodes = tree.total_nodes
        out = None
        for i in range(tree.depth, -1, -1):
            csum_nodes_next = csum_nodes
            csum_nodes -= tree.num_nodes[i]
            out = kernel(tree.offsets.dev, tree.pbounds.dev,
                         *args,
                         slice=slice(csum_nodes, csum_nodes_next))
        return out

    return callable


@memoize
def leaf_tree_traverse(ctx, k, args, setup, node_operation, leaf_operation,
                       output_expr, common_operation, preamble=""):
    # FIXME: variable max_depth
    operation = LEAF_DFS_TEMPLATE % dict(
        setup=setup,
        leaf_operation=leaf_operation,
        node_operation=node_operation,
        common_operation=common_operation,
        output_expr=output_expr,
        max_depth=21,
        k=k
    )

    args = ', '.join(["int *unique_cids, int *cids, int *offsets", args])

    kernel = ElementwiseKernel(
        ctx, args, operation=operation, preamble=preamble)

    def callable(tree_src, tree_dst, *args):
        return kernel(
            tree_dst.unique_cids.dev[:tree_dst.unique_cid_count],
            tree_dst.cids.dev, tree_src.offsets.dev, *args
        )

    return callable


@memoize
def point_tree_traverse(ctx, k, args, setup, node_operation, leaf_operation,
                        output_expr, common_operation, preamble=""):
    # FIXME: variable max_depth
    operation = POINT_DFS_TEMPLATE % dict(
        setup=setup,
        leaf_operation=leaf_operation,
        node_operation=node_operation,
        common_operation=common_operation,
        output_expr=output_expr,
        max_depth=21,
        k=k
    )

    args = ', '.join(["int *cids, int *offsets", args])

    kernel = ElementwiseKernel(
        ctx, args, operation=operation, preamble=preamble)

    def callable(tree_src, tree_dst, *args):
        return kernel(
            tree_dst.cids.dev, tree_src.offsets.dev, *args
        )

    return callable


class Tree(object):
    """k-ary Tree
    """

    def __init__(self, n, k=8, leaf_size=32):
        self.ctx = get_context()
        self.queue = get_queue()
        self.sorted = False
        self.main_helper = get_helper(os.path.join('tree', 'tree.mako'))

        self.initialized = False
        self.preamble = ""
        self.leaf_size = leaf_size
        self.k = k
        self.n = n
        self.sorted = False
        self.depth = 0

        self.index_function_args = []
        self.index_function_arg_ctypes = []
        self.index_function_arg_dtypes = []
        self.index_function_consts = []
        self.index_function_const_ctypes = []
        self.index_code = ""

        self.set_index_function_info()

    def set_index_function_info(self):
        raise NotImplementedError

    def get_data_args(self):
        return [getattr(self, v) for v in self.index_function_args]

    def get_index_constants(self, depth):
        raise NotImplementedError

    def _initialize_data(self):
        self.sorted = False
        num_particles = self.n
        self.pids = Array(np.uint32, n=num_particles, backend='opencl')
        self.cids = Array(np.uint32, n=num_particles, backend='opencl')
        self.cids.fill(0)

        for var, dtype in zip(self.index_function_args,
                              self.index_function_arg_dtypes):
            setattr(self, var, Array(dtype, n=num_particles, backend='opencl'))

        # Filled after tree built
        self.pbounds = None
        self.offsets = None
        self.initialized = True

    def _reinitialize_data(self):
        self.sorted = False
        num_particles = self.n
        self.pids.resize(num_particles)
        self.cids.resize(num_particles)
        self.cids.fill(0)

        for var in self.index_function_args:
            getattr(self, var).resize(num_particles)

        # Filled after tree built
        self.pbounds = None
        self.offsets = None

    def _setup_build(self):
        if not self.initialized:
            self._initialize_data()
        else:
            self._reinitialize_data()

    def _build(self, fixed_depth=None):
        self._build_tree(fixed_depth)

    ###########################################################################
    # Core construction algorithm and helper functions
    ###########################################################################

    # A little bit of manual book-keeping for temporary variables.
    # More specifically, these temporary variables would otherwise be thrown
    # away after building each layer of the tree.
    # We could instead just allocate new arrays after building each layer and
    # and let the GC take care of stuff but I'm guessing this is a
    # a better approach to save on memory
    def _create_temp_vars(self, temp_vars):
        n = self.n
        temp_vars['pids'] = Array(np.uint32, n=n, backend='opencl')
        for var, dtype in zip(self.index_function_args,
                              self.index_function_arg_dtypes):
            temp_vars[var] = Array(dtype, n=n, backend='opencl')
        temp_vars['cids'] = Array(np.uint32, n=n, backend='opencl')

    def _exchange_temp_vars(self, temp_vars):
        for k in temp_vars.keys():
            t = temp_vars[k]
            temp_vars[k] = getattr(self, k)
            setattr(self, k, t)

    def _clean_temp_vars(self, temp_vars):
        for k in list(temp_vars.keys()):
            del temp_vars[k]

    def _get_temp_data_args(self, temp_vars):
        result = [temp_vars[v] for v in self.index_function_args]
        return result

    def _reorder_particles(self, depth, child_count_prefix_sum, offsets_parent,
                           pbounds_parent,
                           seg_flag, csum_nodes_prev, temp_vars):
        # Scan

        args = [('__global ' + ctype + ' *' + v) for v, ctype in
                zip(self.index_function_args, self.index_function_arg_ctypes)]
        args += [(ctype + ' ' + v) for v, ctype in
                 zip(self.index_function_consts,
                     self.index_function_const_ctypes)]
        args = ', '.join(args)

        particle_kernel = _get_particle_kernel(self.ctx, self.k,
                                               args, self.index_code)
        args = [seg_flag.dev, child_count_prefix_sum.dev]
        args += [x.dev for x in self.get_data_args()]
        args += self.get_index_constants(depth)
        particle_kernel(*args)

        # Reorder particles
        reorder_particles = self.main_helper.get_kernel(
            'reorder_particles', k=self.k,
            data_vars=tuple(self.index_function_args),
            data_var_ctypes=tuple(self.index_function_arg_ctypes),
            const_vars=tuple(self.index_function_consts),
            const_var_ctypes=tuple(self.index_function_const_ctypes),
            index_code=self.index_code
        )

        args = [self.pids.dev, self.cids.dev,
                seg_flag.dev,
                pbounds_parent.dev, offsets_parent.dev,
                child_count_prefix_sum.dev,
                temp_vars['pids'].dev, temp_vars['cids'].dev]
        args += [x.dev for x in self.get_data_args()]
        args += [x.dev for x in self._get_temp_data_args(temp_vars)]
        args += self.get_index_constants(depth)
        args += [np.uint32(csum_nodes_prev)]

        reorder_particles(*args)
        self._exchange_temp_vars(temp_vars)

    def _merge_layers(self, offsets_temp, pbounds_temp):
        curr_offset = 0
        total_nodes = 0

        for i in range(self.depth + 1):
            total_nodes += self.num_nodes[i]

        self.offsets = Array(np.int32, n=total_nodes, backend='opencl')
        self.pbounds = Array(cl.cltypes.uint2, n=total_nodes, backend='opencl')

        append_layer = self.main_helper.get_kernel('append_layer')

        self.total_nodes = total_nodes
        for i in range(self.depth + 1):
            append_layer(
                offsets_temp[i].dev, pbounds_temp[i].dev,
                self.offsets.dev, self.pbounds.dev,
                np.int32(curr_offset), np.uint8(i == self.depth)
            )
            curr_offset += self.num_nodes[i]

    def _update_node_data(self, offsets_prev, pbounds_prev, offsets, pbounds,
                          seg_flag, child_count_prefix_sum, csum_nodes,
                          csum_nodes_next, n):
        """Update node data and return number of children which are leaves."""

        # Update particle-related data of children
        set_node_data = self.main_helper.get_kernel("set_node_data", k=self.k)
        set_node_data(offsets_prev.dev, pbounds_prev.dev,
                      offsets.dev, pbounds.dev,
                      seg_flag.dev, child_count_prefix_sum.dev,
                      np.uint32(csum_nodes),
                      np.uint32(n))

        # Set children offsets
        leaf_count = Array(np.uint32, n=1, backend='opencl')
        set_offsets = _get_set_offset_kernel(self.ctx, self.k, self.leaf_size)
        set_offsets(pbounds.dev, offsets.dev, leaf_count.dev,
                    np.uint32(csum_nodes_next))
        return leaf_count.dev[0].get()

    def _build_tree(self, fixed_depth=None):
        # We build the tree one layer at a time. We stop building new
        # layers after either all the
        # nodes are leaves or after reaching the target depth (fixed_depth).
        # At this point, the information for each layer is segmented / not
        # contiguous in memory, and so we run a merge_layers procedure to
        # move the data for all layers into a single array.
        #
        # The procedure for building each layer can be split up as follows
        # 1) Determine which child each particle is going to belong to in the
        #    next layer
        # 2) Perform a kind of segmented scan over this. This gives us the
        #    new order of the particles so that consecutive particles lie in
        #    the same child
        # 3) Reorder the particles based on this order
        # 4) Create a new layer and set the node data for the new layer. We
        #    get to know which particles belong to each node directly from the
        #    results of step 2
        # 5) Set the predicted offsets of the children of the nodes in the
        #    new layer. If a node has fewer than leaf_size particles, it's a
        #    leaf. A kind of prefix sum over this directly let's us know the
        #    predicted offsets.
        # Rinse and repeat for building more layers.
        #
        # Note that after building the last layer, the predicted offsets for
        # the children might not be correctly since we're not going to build
        # more layers. The _merge_layers procedure sets the offsets in the
        # last layer to -1 to correct this.

        num_leaves_here = 0
        n = self.n
        temp_vars = {}

        self.depth = 0
        self.num_nodes = [1]

        # Cumulative sum of nodes in the previous layers
        csum_nodes_prev = 0
        csum_nodes = 1

        # Initialize temporary data (but persistent across layers)
        self._create_temp_vars(temp_vars)

        child_count_prefix_sum = Array(get_vector_dtype('uint', self.k),
                                       n=n, backend='opencl')

        seg_flag = Array(cl.cltypes.char, n=n, backend='opencl')
        seg_flag.fill(0)
        seg_flag.dev[0] = 1

        offsets_temp = [Array(np.int32, n=1, backend='opencl')]
        offsets_temp[-1].fill(1)

        pbounds_temp = [Array(cl.cltypes.uint2, n=1, backend='opencl')]
        pbounds_temp[-1].dev[0].set(cl.cltypes.make_uint2(0, n))

        # FIXME: Depths above 20 possible and feasible for binary / quad trees
        loop_lim = min(fixed_depth, 20)

        for depth in range(1, loop_lim):
            num_nodes = self.k * (self.num_nodes[-1] - num_leaves_here)
            if num_nodes == 0:
                break
            else:
                self.depth += 1
            self.num_nodes.append(num_nodes)

            # Allocate new layer
            offsets_temp.append(Array(np.int32, n=self.num_nodes[-1],
                                      backend='opencl'))
            pbounds_temp.append(Array(cl.cltypes.uint2,
                                      n=self.num_nodes[-1], backend='opencl'))

            # Generate particle index and reorder the particles
            self._reorder_particles(depth, child_count_prefix_sum,
                                    offsets_temp[-2],
                                    pbounds_temp[-2], seg_flag,
                                    csum_nodes_prev,
                                    temp_vars)

            num_leaves_here = self._update_node_data(
                offsets_temp[-2], pbounds_temp[-2],
                offsets_temp[-1], pbounds_temp[-1],
                seg_flag, child_count_prefix_sum,
                csum_nodes, csum_nodes + self.num_nodes[-1], n
            )

            csum_nodes_prev = csum_nodes
            csum_nodes += self.num_nodes[-1]

        self._merge_layers(offsets_temp, pbounds_temp)
        self._clean_temp_vars(temp_vars)

    ###########################################################################
    # Misc
    ###########################################################################

    def _get_unique_cids_and_count(self):
        n = self.n
        self.unique_cids = Array(np.uint32, n=n, backend='opencl')
        self.unique_cids_map = Array(np.uint32, n=n, backend='opencl')
        uniq_count = Array(np.uint32, n=1, backend='opencl')
        unique_cids_kernel = _get_unique_cids_kernel(self.ctx)
        unique_cids_kernel(self.cids.dev, self.unique_cids_map.dev,
                           self.unique_cids.dev, uniq_count.dev)
        self.unique_cid_count = uniq_count.dev[0].get()

    def get_leaves(self):
        leaves = Array(np.uint32, n=self.offsets.dev.shape[0], backend='opencl')
        num_leaves = Array(np.uint32, n=1, backend='opencl')
        leaves_kernel = _get_leaves_kernel(self.ctx, self.leaf_size)
        leaves_kernel(self.offsets.dev, self.pbounds.dev,
                      leaves.dev, num_leaves.dev)

        num_leaves = num_leaves.dev[0].get()
        return leaves.dev[:num_leaves], num_leaves

    def _sort(self):
        """Set tree as being sorted

        The particle array needs to be aligned by the caller!
        """
        if not self.sorted:
            self.sorted = 1

    ###########################################################################
    # Tree API
    ###########################################################################
    def allocate_node_prop(self, dtype):
        return Array(dtype, n=self.total_nodes, backend='opencl')

    def allocate_leaf_prop(self, dtype):
        return Array(dtype, n=int(self.unique_cid_count), backend='opencl')

    def get_preamble(self):
        if self.sorted:
            return "#define PID(idx) (idx)"
        else:
            return "#define PID(idx) (pids[idx])"

    def get_leaf_size_partitions(self, group_min, group_max):
        """Partition leaves based on leaf size

        Parameters
        ----------
        group_min
            Minimum leaf size
        group_max
            Maximum leaf size
        Returns
        -------
        groups : Array
            An array which contains the cell ids of leaves
            with leaf size > group_min and leaf size <= group_max
        group_count : int
            The number of leaves which satisfy the given condition
            on the leaf size
        """
        groups = Array(np.uint32, n=int(self.unique_cid_count),
                       backend='opencl')
        group_count = Array(np.uint32, n=1, backend='opencl')

        get_cid_groups = _get_cid_groups_kernel(self.ctx)
        get_cid_groups(self.unique_cids.dev[:self.unique_cid_count],
                       self.pbounds.dev, groups.dev, group_count.dev,
                       np.int32(group_min), np.int32(group_max))
        result = groups, int(group_count.dev[0].get())
        return result

    def tree_bottom_up(self, args, setup, leaf_operation, node_operation,
                       output_expr, preamble=""):
        return tree_bottom_up(self.ctx, args, setup, leaf_operation,
                              node_operation, output_expr, preamble)

    def leaf_tree_traverse(self, args, setup, node_operation, leaf_operation,
                           output_expr, common_operation="", preamble=""):
        """
        Traverse this (source) tree. One thread for each leaf of
        destination tree.
        """

        return leaf_tree_traverse(self.ctx, self.k, args, setup,
                                  node_operation, leaf_operation,
                                  output_expr, common_operation, preamble)

    def point_tree_traverse(self, args, setup, node_operation, leaf_operation,
                            output_expr, common_operation="", preamble=""):
        """
        Traverse this (source) tree. One thread for each particle of
        destination tree.
        """

        return point_tree_traverse(self.ctx, self.k, args, setup,
                                   node_operation, leaf_operation,
                                   output_expr, common_operation, preamble)
