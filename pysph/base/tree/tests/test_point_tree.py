import numpy as np
import unittest
from pytest import importorskip

cl = importorskip('pyopencl')

import pysph.base.particle_array
from pysph.base.device_helper import DeviceHelper   # noqa: E402
from pysph.base.utils import get_particle_array  # noqa: E402
from pysph.base.tree.point_tree import PointTree  # noqa: E402


def _gen_uniform_dataset_2d(n, h, seed=None):
    if seed is not None:
        np.random.seed(seed)
    u = np.random.uniform
    pa = get_particle_array(x=u(size=n), y=u(size=n), h=h)
    h = DeviceHelper(pa, backend='opencl')
    pa.set_device_helper(h)

    return pa


def _gen_uniform_dataset(n, h, seed=None):
    if seed is not None:
        np.random.seed(seed)
    u = np.random.uniform
    pa = get_particle_array(x=u(size=n), y=u(size=n), z=u(size=n), h=h)
    h = DeviceHelper(pa, backend='opencl')
    pa.set_device_helper(h)

    return pa


def _dfs_find_leaf(tree):
    leaf_id_count = tree.allocate_leaf_prop(np.int32)
    dfs_find_leaf = tree.leaf_tree_traverse(
        "int *leaf_id_count",
        setup="leaf_id_count[i] = 0;",
        node_operation="if (cid_dst == cid_src) leaf_id_count[i]++",
        leaf_operation="if (cid_dst == cid_src) leaf_id_count[i]++",
        output_expr=""
    )

    dfs_find_leaf(tree, tree, leaf_id_count.dev)
    return leaf_id_count.dev.get()


def _check_children_overlap_2d(node_xmin, node_xmax, child_offset):
    for j in range(4):
        nxmin1 = node_xmin[child_offset + j]
        nxmax1 = node_xmax[child_offset + j]
        for k in range(4):
            nxmin2 = node_xmin[child_offset + k]
            nxmax2 = node_xmax[child_offset + k]
            if j != k:
                assert (nxmax1[0] <= nxmin2[0] or nxmax2[0] <= nxmin1[0] or
                        nxmax1[1] <= nxmin2[1] or nxmax2[1] <= nxmin1[1])


def _check_children_overlap(node_xmin, node_xmax, child_offset):
    for j in range(8):
        nxmin1 = node_xmin[child_offset + j]
        nxmax1 = node_xmax[child_offset + j]
        for k in range(8):
            nxmin2 = node_xmin[child_offset + k]
            nxmax2 = node_xmax[child_offset + k]
            if j != k:
                assert (nxmax1[0] <= nxmin2[0] or nxmax2[0] <= nxmin1[0] or
                        nxmax1[1] <= nxmin2[1] or nxmax2[1] <= nxmin1[1] or
                        nxmax1[2] <= nxmin2[2] or nxmax2[2] <= nxmin1[2])


def _test_tree_structure(tree, k):
    # Traverse tree and check if max depth is correct
    # Additionally check if particle sets of siblings is disjoint
    # and union of particle sets of a nodes children = nodes own children
    #
    # This effectively also checks that no  particle is present in two nodes of
    # the same level

    s = [0, ]
    d = [0, ]

    offsets = tree.offsets.dev.get()
    pbounds = tree.pbounds.dev.get()

    max_depth = tree.depth
    max_depth_here = 0
    pids = set()

    while len(s) != 0:
        n = s[0]
        depth = d[0]
        max_depth_here = max(max_depth_here, depth)
        pbound = pbounds[n]
        assert (depth <= max_depth)

        del s[0]
        del d[0]

        if offsets[n] == -1:
            for i in range(pbound[0], pbound[1]):
                pids.add(i)
            continue

        # Particle ranges of children are contiguous
        # and are contained within parent's particle range
        start = pbound[0]
        for i in range(k):
            child_idx = offsets[n] + i
            assert (pbounds[child_idx][0] == start)
            assert (pbounds[child_idx][0] <= pbounds[child_idx][1])
            start = pbounds[child_idx][1]

            assert (child_idx < len(offsets))
            s.append(child_idx)
            d.append(depth + 1)
        assert (start == pbound[1])


class QuadtreeTestCase(unittest.TestCase):
    def setUp(self):
        use_double = False
        self.N = 3000
        pa = _gen_uniform_dataset_2d(self.N, 0.2, seed=0)
        self.quadtree = PointTree(pa, radius_scale=1., use_double=use_double,
                                  leaf_size=32, dim=2)
        self.leaf_size = 32
        self.quadtree.refresh(np.array([0., 0.]), np.array([1., 1.]),
                              np.min(pa.h))
        self.pa = pa

    def test_pids(self):
        pids = self.quadtree.pids.dev.get()
        s = set()
        for i in range(len(pids)):
            if 0 <= pids[i] < self.N:
                s.add(pids[i])

        assert (len(s) == self.N)

    def test_depth_and_inclusiveness(self):
        _test_tree_structure(self.quadtree, 4)
    
    @unittest.skip("Fails without oldest-supported-numpy")
    def test_node_bounds(self):

        self.quadtree.set_node_bounds()
        pids = self.quadtree.pids.dev.get()
        offsets = self.quadtree.offsets.dev.get()
        pbounds = self.quadtree.pbounds.dev.get()
        node_xmin = self.quadtree.node_xmin.dev.get()
        node_xmax = self.quadtree.node_xmax.dev.get()
        node_hmax = self.quadtree.node_hmax.dev.get()

        x = self.pa.x[pids]
        y = self.pa.y[pids]
        h = self.pa.h[pids]

        for i in range(len(offsets)):
            nxmin = node_xmin[i]
            nxmax = node_xmax[i]
            nhmax = node_hmax[i]

            for j in range(pbounds[i][0], pbounds[i][1]):
                assert (nxmin[0] <= np.float32(x[j]) <= nxmax[0])
                assert (nxmin[1] <= np.float32(y[j]) <= nxmax[1])
                assert (np.float32(h[j]) <= nhmax)
            # Check that children nodes don't overlap
            if offsets[i] != -1:
                _check_children_overlap_2d(node_xmin, node_xmax, offsets[i])

    def test_dfs_traversal(self):
        leaf_id_count = _dfs_find_leaf(self.quadtree)
        np.testing.assert_array_equal(
            np.ones(self.quadtree.unique_cid_count, dtype=np.int32),
            leaf_id_count
        )

    def test_get_leaf_size_partitions(self):
        a, b = np.random.randint(0, self.leaf_size, size=2)
        a, b = min(a, b), max(a, b)

        pbounds = self.quadtree.pbounds.dev.get()
        offsets = self.quadtree.offsets.dev.get()

        mapping, count = self.quadtree.get_leaf_size_partitions(a, b)
        mapping = mapping.dev.get()
        map_set_gpu = {mapping[i] for i in range(count)}
        map_set_here = {i for i in range(len(offsets))
                        if offsets[i] == -1 and
                        a < (pbounds[i][1] - pbounds[i][0]) <= b}
        assert (map_set_gpu == map_set_here)

    def tearDown(self):
        del self.quadtree


class OctreeTestCase(unittest.TestCase):
    def setUp(self):
        use_double = False
        self.N = 3000
        pa = _gen_uniform_dataset(self.N, 0.2, seed=0)
        self.octree = PointTree(pa, dim=3, radius_scale=1.,
                                use_double=use_double,
                                leaf_size=64)
        self.leaf_size = 64
        self.octree.refresh(np.array([0., 0., 0.]), np.array([1., 1., 1.]),
                            np.min(pa.h))
        self.pa = pa

    def test_pids(self):
        pids = self.octree.pids.dev.get()
        s = set()
        for i in range(len(pids)):
            if 0 <= pids[i] < self.N:
                s.add(pids[i])

        assert (len(s) == self.N)

    def test_depth_and_inclusiveness(self):
        _test_tree_structure(self.octree, 8)

    @unittest.skip("Fails without oldest-supported-numpy")
    def test_node_bounds(self):

        self.octree.set_node_bounds()
        print(self.octree.node_hmax.dev.get())
        pids = self.octree.pids.dev.get()
        offsets = self.octree.offsets.dev.get()
        pbounds = self.octree.pbounds.dev.get()
        node_xmin = self.octree.node_xmin.dev.get()
        node_xmax = self.octree.node_xmax.dev.get()
        node_hmax = self.octree.node_hmax.dev.get()

        x = self.pa.x[pids]
        y = self.pa.y[pids]
        z = self.pa.z[pids]
        h = self.pa.h[pids]

        for i in range(len(offsets)):
            nxmin = node_xmin[i]
            nxmax = node_xmax[i]
            nhmax = node_hmax[i]

            for j in range(pbounds[i][0], pbounds[i][1]):
                assert (nxmin[0] <= np.float32(x[j]) <= nxmax[0])
                assert (nxmin[1] <= np.float32(y[j]) <= nxmax[1])
                assert (nxmin[2] <= np.float32(z[j]) <= nxmax[2])
                assert (np.float32(h[j]) <= nhmax)
            # Check that children nodes don't overlap
            if offsets[i] != -1:
                _check_children_overlap(node_xmin, node_xmax, offsets[i])

    def test_dfs_traversal(self):
        leaf_id_count = _dfs_find_leaf(self.octree)
        np.testing.assert_array_equal(
            np.ones(self.octree.unique_cid_count, dtype=np.int32),
            leaf_id_count
        )

    def test_get_leaf_size_partitions(self):
        a, b = np.random.randint(0, self.leaf_size, size=2)
        a, b = min(a, b), max(a, b)

        pbounds = self.octree.pbounds.dev.get()
        offsets = self.octree.offsets.dev.get()

        mapping, count = self.octree.get_leaf_size_partitions(a, b)
        mapping = mapping.dev.get()
        map_set_gpu = {mapping[i] for i in range(count)}
        map_set_here = {i for i in range(len(offsets))
                        if offsets[i] == -1 and
                        a < (pbounds[i][1] - pbounds[i][0]) <= b}
        assert (map_set_gpu == map_set_here)

    def tearDown(self):
        del self.octree
