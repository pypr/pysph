from pysph.base.tree.tree import Tree
from pysph.base.tree.helpers import ParticleArrayWrapper, get_helper, \
    make_vec_dict, ctype_to_dtype, get_vector_dtype
from compyle.opencl import profile_kernel, DeviceWGSException, get_queue, \
    named_profile, get_context
from compyle.array import Array
from pytools import memoize

import sys
import numpy as np

import pyopencl as cl
from pyopencl.scan import GenericScanKernel
import pyopencl.tools

from mako.template import Template

# For Mako
disable_unicode = False if sys.version_info.major > 2 else True


class IncompatibleTreesException(Exception):
    pass


@named_profile('neighbor_count_prefix_sum', backend='opencl')
@memoize
def _get_neighbor_count_prefix_sum_kernel(ctx):
    return GenericScanKernel(ctx, np.int32,
                             arguments="__global int *ary",
                             input_expr="ary[i]",
                             scan_expr="a+b", neutral="0",
                             output_statement="ary[i] = prev_item")


@memoize
def _get_macros_preamble(c_type, sorted, dim):
    result = Template("""
    #define IN_BOUNDS(X, MIN, MAX) ((X >= MIN) && (X < MAX))
    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
    #define NORM2_2D(X, Y) ((X)*(X) + (Y)*(Y))
    #define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
    #define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
    #define AVG(X, Y) (((X) + (Y)) / 2)
    #define ABS(X) ((X) > 0 ? (X) : -(X))
    #define SQR(X) ((X) * (X))

    typedef struct {
        union {
            float s0;
            float x;
        };
    } float1;

    typedef struct {
        union {
            double s0;
            double x;
        };
    } double1;

    % if sorted:
    #define PID(idx) (idx)
    % else:
    #define PID(idx) (pids[idx])
    % endif

    char contains(${data_t}${dim} node_xmin1, ${data_t}${dim} node_xmax1,
                  ${data_t}${dim} node_xmin2, ${data_t}${dim} node_xmax2)
    {
        // Check if node n1 contains node n2
        char res = 1;
        % for i in range(dim):
            res = res && (node_xmin1.s${i} <= node_xmin2.s${i}) &&
                  (node_xmax1.s${i} >= node_xmax2.s${i});
        % endfor

        return res;
    }

    char contains_search(${data_t}${dim} node_xmin1,
                  ${data_t}${dim} node_xmax1,
                  ${data_t} node_hmax1,
                  ${data_t}${dim} node_xmin2, ${data_t}${dim} node_xmax2)
    {
        // Check if node n1 contains node n2 with n1 having
        // its search radius extension
        ${data_t} h = node_hmax1;
        char res = 1;
        %for i in range(dim):
            res = res & (node_xmin1.s${i} - h <= node_xmin2.s${i}) &
                  (node_xmax1.s${i} + h >= node_xmax2.s${i});
        %endfor

        return res;
    }

    char intersects(${data_t}${dim} node_xmin1, ${data_t}${dim} node_xmax1,
                    ${data_t} node_hmax1,
                    ${data_t}${dim} node_xmin2, ${data_t}${dim} node_xmax2,
                    ${data_t} node_hmax2) {
        // Check if node n1 'intersects' node n2
        ${data_t} cdist;
        ${data_t} w1, w2, wavg = 0;
        char res = 1;
        ${data_t} h = MAX(node_hmax1, node_hmax2);

        % for i in range(dim):
            cdist = fabs((node_xmin1.s${i} + node_xmax1.s${i}) / 2 -
                         (node_xmin2.s${i} + node_xmax2.s${i}) / 2);
            w1 = fabs(node_xmin1.s${i} - node_xmax1.s${i});
            w2 = fabs(node_xmin2.s${i} - node_xmax2.s${i});
            wavg = AVG(w1, w2);
            res &= (cdist - wavg <= h);
        % endfor

        return res;
    }
    """, disable_unicode=disable_unicode).render(data_t=c_type, sorted=sorted,
                                                 dim=dim)
    return result


@memoize
def _get_node_bound_kernel_parameters(dim, data_t, xvars):
    result = {}
    result['setup'] = Template(
        r"""
        ${data_t} xmin[${dim}] = {${', '.join(['INFINITY'] * dim)}};
        ${data_t} xmax[${dim}] = {${', '.join(['-INFINITY'] * dim)}};
        ${data_t} hmax = 0;
        """, disable_unicode=disable_unicode).render(dim=dim, data_t=data_t)

    result['args'] = Template(
        r"""int *pids,
        % for v in xvars:
        ${data_t} *${v},
        % endfor
        ${data_t} *h,
        ${data_t} radius_scale,
        ${data_t}${dim} *node_xmin,
        ${data_t}${dim} *node_xmax,
        ${data_t} *node_hmax
        """, disable_unicode=disable_unicode).render(dim=dim,
                                                     data_t=data_t,
                                                     xvars=xvars)

    result['leaf_operation'] = Template(
        r"""
        for (int j=pbound.s0; j < pbound.s1; j++) {
        int pid = PID(j);
        % for d in range(dim):
            xmin[${d}] = fmin(xmin[${d}], ${xvars[d]}[pid]);
            xmax[${d}] = fmax(xmax[${d}], ${xvars[d]}[pid]);
        % endfor
        hmax = fmax(h[pid] * radius_scale, hmax);
        }
        """, disable_unicode=disable_unicode).render(dim=dim, xvars=xvars)

    result['node_operation'] = Template(
        r"""
        % for i in range(2 ** dim):
            % for d in range(dim):
                xmin[${d}] = fmin(
                    xmin[${d}], node_xmin[child_offset + ${i}].s${d}
                );
                xmax[${d}] = fmax(
                    xmax[${d}], node_xmax[child_offset + ${i}].s${d}
                );
            % endfor
            hmax = fmax(hmax, node_hmax[child_offset + ${i}]);
        % endfor
        """, disable_unicode=disable_unicode).render(dim=dim)

    result['output_expr'] = Template(
        """
        % for d in range(dim):
            node_xmin[node_idx].s${d} = xmin[${d}];
            node_xmax[node_idx].s${d} = xmax[${d}];
        % endfor
        node_hmax[node_idx] = hmax;
        """, disable_unicode=disable_unicode).render(dim=dim)

    return result


@memoize
def _get_leaf_neighbor_kernel_parameters(data_t, dim, args, setup, operation,
                                         output_expr):
    result = {
        'setup': Template(r"""
            ${data_t}${dim} node_xmin1;
            ${data_t}${dim} node_xmax1;
            ${data_t} node_hmax1;

            ${data_t}${dim} node_xmin2;
            ${data_t}${dim} node_xmax2;
            ${data_t} node_hmax2;

            node_xmin1 = node_xmin_dst[cid_dst];
            node_xmax1 = node_xmax_dst[cid_dst];
            node_hmax1 = node_hmax_dst[cid_dst];

            %(setup)s;
            """ % dict(setup=setup),
                          disable_unicode=disable_unicode).render(
            data_t=data_t, dim=dim),
        'node_operation': Template("""
            node_xmin2 = node_xmin_src[cid_src];
            node_xmax2 = node_xmax_src[cid_src];
            node_hmax2 = node_hmax_src[cid_src];

            if (!intersects(node_xmin1, node_xmax1, node_hmax1,
                            node_xmin2, node_xmax2, node_hmax2) &&
                !contains(node_xmin2, node_xmax2, node_xmin1, node_xmax1)) {
                flag = 0;
                break;
            }
            """, disable_unicode=disable_unicode).render(data_t=data_t),
        'leaf_operation': Template("""
            node_xmin2 = node_xmin_src[cid_src];
            node_xmax2 = node_xmax_src[cid_src];
            node_hmax2 = node_hmax_src[cid_src];

            if (intersects(node_xmin1, node_xmax1, node_hmax1,
                            node_xmin2, node_xmax2, node_hmax2) ||
                contains_search(node_xmin1, node_xmax1, node_hmax1,
                                node_xmin2, node_xmax2)) {
                %(operation)s;
            }
            """ % dict(operation=operation),
                                   disable_unicode=disable_unicode).render(),
        'output_expr': output_expr,
        'args': Template("""
            ${data_t}${dim} *node_xmin_src, ${data_t}${dim} *node_xmax_src,
            ${data_t} *node_hmax_src,
            ${data_t}${dim} *node_xmin_dst, ${data_t}${dim} *node_xmax_dst,
            ${data_t} *node_hmax_dst,
            """ + args, disable_unicode=disable_unicode).render(data_t=data_t,
                                                                dim=dim)

    }
    return result


# Support for 1D
def register_custom_pyopencl_ctypes():
    cl.tools.get_or_register_dtype('float1', np.dtype([('s0', np.float32)]))
    cl.tools.get_or_register_dtype('double1', np.dtype([('s0', np.float64)]))


register_custom_pyopencl_ctypes()


class PointTree(Tree):
    def __init__(self, pa, dim=2, leaf_size=32, radius_scale=2.0,
                 use_double=False, c_type='float'):
        super(PointTree, self).__init__(pa.get_number_of_particles(), 2 ** dim,
                                        leaf_size)

        assert (1 <= dim <= 3)
        self.max_depth = None
        self.dim = dim
        self.powdim = 2 ** self.dim
        self.xvars = ('x', 'y', 'z')[:dim]

        self.c_type = c_type
        self.c_type_src = 'double' if use_double else 'float'

        if use_double and c_type == 'float':
            # Extend the search radius a little to account for rounding errors
            radius_scale = radius_scale * (1 + 2e-7)

        # y and z coordinates need to be present for 1D and z for 2D
        # This is because the NNPS implementation below assumes them to be
        # just set to 0.
        self.pa = ParticleArrayWrapper(pa, self.c_type_src,
                                       self.c_type, ('x', 'y', 'z', 'h'))

        self.radius_scale = radius_scale
        self.use_double = use_double

        self.helper = get_helper('tree/point_tree.mako', self.c_type)
        self.xmin = None
        self.xmax = None
        self.hmin = None
        self.make_vec = make_vec_dict[c_type][self.dim]
        self.ctx = get_context()

    def set_index_function_info(self):
        self.index_function_args = ["sfc"]
        self.index_function_arg_ctypes = ["ulong"]
        self.index_function_arg_dtypes = [np.uint64]
        self.index_function_consts = ['mask', 'rshift']
        self.index_function_const_ctypes = ['ulong', 'char']
        self.index_code = "((sfc[i] & mask) >> rshift)"

    def _calc_cell_size_and_depth(self):
        self.cell_size = self.hmin * self.radius_scale * (1. + 1e-3)

        # Logic from gpu_domain_manager.py
        if self.cell_size < 1e-6:
            self.cell_size = 1

        # This lets the tree grow up to log2(128) = 7 layers beyond what it
        # could have previously. Pretty arbitrary.
        self.cell_size /= 128
        max_width = max((self.xmax[i] - self.xmin[i]) for i in range(self.dim))
        self.max_depth = int(np.ceil(np.log2(max_width / self.cell_size))) + 1

    def _bin(self):
        dtype = ctype_to_dtype(self.c_type)
        fill_particle_data = self.helper.get_kernel("fill_particle_data",
                                                    dim=self.dim,
                                                    xvars=self.xvars)
        pa_gpu = self.pa.gpu
        args = [getattr(pa_gpu, v).dev for v in self.xvars]
        args += [dtype(self.cell_size),
                 self.make_vec(*[self.xmin[i] for i in range(self.dim)]),
                 self.sfc.dev, self.pids.dev]
        fill_particle_data(*args)

    def get_index_constants(self, depth):
        rshift = np.uint8(self.dim * (self.max_depth - depth - 1))
        mask = np.uint64((2 ** self.dim - 1) << rshift)
        return mask, rshift

    def _adjust_domain_width(self):
        # Convert width of domain to a power of 2 multiple of cell size
        # (Optimal width for cells)
        # Note that this makes sure the width in _all_ dimensions is the
        # same. We want our nodes to be cubes ideally.
        cell_size = self.hmin * self.radius_scale * (1. + 1e-5)
        max_width = np.max(self.xmax - self.xmin)

        new_width = cell_size * 2.0 ** int(
            np.ceil(np.log2(max_width / cell_size)))

        diff = (new_width - (self.xmax - self.xmin)) / 2

        self.xmin -= diff
        self.xmax += diff

    def setup_build(self, xmin, xmax, hmin):
        self._setup_build()
        self.pa.sync()
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.hmin = hmin
        self._adjust_domain_width()
        self._calc_cell_size_and_depth()
        self._bin()

    def build(self, fixed_depth=None):
        self._build(self.max_depth if fixed_depth is None else fixed_depth)
        self._get_unique_cids_and_count()

    def refresh(self, xmin, xmax, hmin, fixed_depth=None):
        self.setup_build(xmin, xmax, hmin)
        self.build(fixed_depth)

    def _sort(self):
        """Set tree as being sorted

        The particle array needs to be aligned by the caller!
        """
        if not self.sorted:
            self.pa.sync()
            self.sorted = 1

    ###########################################################################
    # General algorithms
    ###########################################################################
    def set_node_bounds(self):
        vector_data_t = get_vector_dtype(self.c_type, self.dim)
        dtype = ctype_to_dtype(self.c_type)

        self.node_xmin = self.allocate_node_prop(vector_data_t)
        self.node_xmax = self.allocate_node_prop(vector_data_t)
        self.node_hmax = self.allocate_node_prop(dtype)

        params = _get_node_bound_kernel_parameters(self.dim, self.c_type,
                                                   self.xvars)
        set_node_bounds = self.tree_bottom_up(
            params['args'], params['setup'], params['leaf_operation'],
            params['node_operation'], params['output_expr'],
            preamble=_get_macros_preamble(self.c_type, self.sorted, self.dim)
        )
        set_node_bounds = profile_kernel(set_node_bounds, 'set_node_bounds',
                                         backend='opencl')

        pa_gpu = self.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        args = [self, self.pids.dev]
        args += [getattr(pa_gpu, v).dev for v in self.xvars]
        args += [pa_gpu.h.dev,
                 dtype(self.radius_scale),
                 self.node_xmin.dev, self.node_xmax.dev,
                 self.node_hmax.dev]

        set_node_bounds(*args)

    ###########################################################################
    # Nearest Neighbor Particle Search (NNPS)
    ###########################################################################
    def _leaf_neighbor_operation(self, tree_src, args, setup, operation,
                                 output_expr):
        # Template for finding neighboring cids of a cell.
        params = _get_leaf_neighbor_kernel_parameters(self.c_type, self.dim,
                                                      args,
                                                      setup, operation,
                                                      output_expr)
        kernel = tree_src.leaf_tree_traverse(
            params['args'], params['setup'], params['node_operation'],
            params['leaf_operation'], params['output_expr'],
            preamble=_get_macros_preamble(self.c_type, self.sorted, self.dim)
        )

        def callable(*args):
            return kernel(tree_src, self,
                          tree_src.node_xmin.dev,
                          tree_src.node_xmax.dev,
                          tree_src.node_hmax.dev,
                          self.node_xmin.dev, self.node_xmax.dev,
                          self.node_hmax.dev,
                          *args)

        return callable

    def find_neighbor_cids(self, tree_src):
        neighbor_cid_count = Array(np.uint32, n=self.unique_cid_count + 1,
                                   backend='opencl')
        find_neighbor_cid_counts = self._leaf_neighbor_operation(
            tree_src,
            args="uint2 *pbounds, int *cnt",
            setup="int count=0",
            operation="""
                    if (pbounds[cid_src].s0 < pbounds[cid_src].s1)
                        count++;
                    """,
            output_expr="cnt[i] = count;"
        )
        find_neighbor_cid_counts = profile_kernel(
            find_neighbor_cid_counts, 'find_neighbor_cid_count',
            backend='opencl'
        )
        find_neighbor_cid_counts(tree_src.pbounds.dev,
                                 neighbor_cid_count.dev)

        neighbor_psum = _get_neighbor_count_prefix_sum_kernel(self.ctx)
        neighbor_psum(neighbor_cid_count.dev)

        total_neighbors = int(neighbor_cid_count.dev[-1].get())
        neighbor_cids = Array(np.uint32, n=total_neighbors,
                              backend='opencl')

        find_neighbor_cids = self._leaf_neighbor_operation(
            tree_src,
            args="uint2 *pbounds, int *cnt, int *neighbor_cids",
            setup="int offset=cnt[i];",
            operation="""
            if (pbounds[cid_src].s0 < pbounds[cid_src].s1)
                neighbor_cids[offset++] = cid_src;
            """,
            output_expr=""
        )
        find_neighbor_cids = profile_kernel(
            find_neighbor_cids, 'find_neighbor_cids', backend='opencl')
        find_neighbor_cids(tree_src.pbounds.dev,
                           neighbor_cid_count.dev, neighbor_cids.dev)
        return neighbor_cid_count, neighbor_cids

    # TODO?: 1D and 2D NNPS not properly supported here.
    # Just assuming the other spatial coordinates (y and z in case of 1D,
    # and z in case of 2D) are set to 0.
    def find_neighbor_lengths_elementwise(self, neighbor_cid_count,
                                          neighbor_cids, tree_src,
                                          neighbor_count):
        self.check_nnps_compatibility(tree_src)

        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = tree_src.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        find_neighbor_counts = self.helper.get_kernel(
            'find_neighbor_counts_elementwise', sorted=self.sorted
        )
        find_neighbor_counts(self.unique_cids_map.dev, tree_src.pids.dev,
                             self.pids.dev,
                             self.cids.dev,
                             tree_src.pbounds.dev, self.pbounds.dev,
                             pa_gpu_src.x.dev, pa_gpu_src.y.dev,
                             pa_gpu_src.z.dev,
                             pa_gpu_src.h.dev,
                             pa_gpu_dst.x.dev, pa_gpu_dst.y.dev,
                             pa_gpu_dst.z.dev,
                             pa_gpu_dst.h.dev,
                             dtype(self.radius_scale),
                             neighbor_cid_count.dev,
                             neighbor_cids.dev,
                             neighbor_count.dev)

    def find_neighbors_elementwise(self, neighbor_cid_count, neighbor_cids,
                                   tree_src, start_indices, neighbors):
        self.check_nnps_compatibility(tree_src)

        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = tree_src.pa.gpu

        dtype = ctype_to_dtype(self.c_type)

        find_neighbors = self.helper.get_kernel(
            'find_neighbors_elementwise', sorted=self.sorted)
        find_neighbors(self.unique_cids_map.dev, tree_src.pids.dev,
                       self.pids.dev,
                       self.cids.dev,
                       tree_src.pbounds.dev, self.pbounds.dev,
                       pa_gpu_src.x.dev, pa_gpu_src.y.dev, pa_gpu_src.z.dev,
                       pa_gpu_src.h.dev,
                       pa_gpu_dst.x.dev, pa_gpu_dst.y.dev, pa_gpu_dst.z.dev,
                       pa_gpu_dst.h.dev,
                       dtype(self.radius_scale),
                       neighbor_cid_count.dev,
                       neighbor_cids.dev,
                       start_indices.dev,
                       neighbors.dev)

    def _is_valid_nnps_wgs(self):
        # Max work group size can only be found by building the
        # kernel.
        try:
            find_neighbor_counts = self.helper.get_kernel(
                'find_neighbor_counts', sorted=self.sorted, wgs=self.leaf_size
            )

            find_neighbor = self.helper.get_kernel(
                'find_neighbors', sorted=self.sorted, wgs=self.leaf_size
            )
        except DeviceWGSException:
            return False
        else:
            return True

    def find_neighbor_lengths(self, neighbor_cid_count, neighbor_cids,
                              tree_src, neighbor_count,
                              use_partitions=False):
        self.check_nnps_compatibility(tree_src)

        wgs = self.leaf_size
        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = tree_src.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        def find_neighbor_counts_for_partition(partition_cids, partition_size,
                                               partition_wgs, q=None):
            find_neighbor_counts = self.helper.get_kernel(
                'find_neighbor_counts', sorted=self.sorted, wgs=wgs
            )
            find_neighbor_counts(partition_cids.dev, tree_src.pids.dev,
                                 self.pids.dev,
                                 self.cids.dev,
                                 tree_src.pbounds.dev, self.pbounds.dev,
                                 pa_gpu_src.x.dev, pa_gpu_src.y.dev,
                                 pa_gpu_src.z.dev,
                                 pa_gpu_src.h.dev,
                                 pa_gpu_dst.x.dev, pa_gpu_dst.y.dev,
                                 pa_gpu_dst.z.dev,
                                 pa_gpu_dst.h.dev,
                                 dtype(self.radius_scale),
                                 neighbor_cid_count.dev,
                                 neighbor_cids.dev,
                                 neighbor_count.dev,
                                 gs=(partition_wgs * partition_size,),
                                 ls=(partition_wgs,),
                                 queue=(get_queue() if q is None else q))

        if use_partitions and wgs > 32:
            if wgs < 128:
                wgs1 = 32
            else:
                wgs1 = 64

            m1, n1 = self.get_leaf_size_partitions(0, wgs1)

            find_neighbor_counts_for_partition(m1, n1, min(wgs, wgs1))
            m2, n2 = self.get_leaf_size_partitions(wgs1, wgs)
            find_neighbor_counts_for_partition(m2, n2, wgs)
        else:
            find_neighbor_counts_for_partition(
                self.unique_cids, self.unique_cid_count, wgs)

    def find_neighbors(self, neighbor_cid_count, neighbor_cids, tree_src,
                       start_indices, neighbors, use_partitions=False):
        self.check_nnps_compatibility(tree_src)

        wgs = self.leaf_size if self.leaf_size % 32 == 0 else \
            self.leaf_size + 32 - self.leaf_size % 32
        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = tree_src.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        def find_neighbors_for_partition(partition_cids, partition_size,
                                         partition_wgs, q=None):
            find_neighbors = self.helper.get_kernel('find_neighbors',
                                                    sorted=self.sorted,
                                                    wgs=wgs)
            find_neighbors(partition_cids.dev, tree_src.pids.dev,
                           self.pids.dev,
                           self.cids.dev,
                           tree_src.pbounds.dev, self.pbounds.dev,
                           pa_gpu_src.x.dev, pa_gpu_src.y.dev, pa_gpu_src.z.dev,
                           pa_gpu_src.h.dev,
                           pa_gpu_dst.x.dev, pa_gpu_dst.y.dev, pa_gpu_dst.z.dev,
                           pa_gpu_dst.h.dev,
                           dtype(self.radius_scale),
                           neighbor_cid_count.dev,
                           neighbor_cids.dev,
                           start_indices.dev,
                           neighbors.dev,
                           gs=(partition_wgs * partition_size,),
                           ls=(partition_wgs,),
                           queue=(get_queue() if q is None else q))

        if use_partitions and wgs > 32:
            if wgs < 128:
                wgs1 = 32
            else:
                wgs1 = 64

            m1, n1 = self.get_leaf_size_partitions(0, wgs1)
            fraction = (n1 / int(self.unique_cid_count))

            if fraction > 0.3:
                find_neighbors_for_partition(m1, n1, wgs1)
                m2, n2 = self.get_leaf_size_partitions(wgs1, wgs)
                assert (n1 + n2 == self.unique_cid_count)
                find_neighbors_for_partition(m2, n2, wgs)
                return
        else:
            find_neighbors_for_partition(
                self.unique_cids, self.unique_cid_count, wgs)

    def check_nnps_compatibility(self, tree):
        """Check if tree types and parameters are compatible for NNPS

        Two trees must satisfy a few conditions so that NNPS can be performed
        on one tree using the other as reference. In this case, the following
        conditions must be satisfied -

        1) Currently both should be instances of point_tree.PointTree
        2) Both must have the same sortedness
        3) Both must use the same floating-point datatype
        4) Both must have the same leaf sizes
        """
        if not isinstance(tree, PointTree):
            raise IncompatibleTreesException(
                "Both trees must be of the same type for NNPS"
            )

        if self.sorted != tree.sorted:
            raise IncompatibleTreesException(
                "Tree sortedness need to be the same for NNPS"
            )

        if self.c_type != tree.c_type or self.use_double != tree.use_double:
            raise IncompatibleTreesException(
                "Tree floating-point data types need to be the same for NNPS"
            )

        if self.leaf_size != tree.leaf_size:
            raise IncompatibleTreesException(
                "Tree leaf sizes need to be the same for NNPS (%d != %d)" % (
                    self.leaf_size, tree.leaf_size)
            )

        return
