"""unittests for the serial NNPS

You can run the tests like so:

    $ pytest -v test_nnps.py
"""
import numpy
from numpy import random

# PySPH imports
from pysph.base.point import IntPoint, Point
from pysph.base.utils import get_particle_array
from pysph.base import nnps
from compyle.config import get_config

# Carrays from PyZoltan
from cyarray.carray import UIntArray, IntArray

# Python testing framework
import unittest
import pytest
from pytest import importorskip


class SimpleNNPSTestCase(unittest.TestCase):
    """Simplified NNPS test case

    We distribute particles manually and perform sanity checks on NNPS

    """

    def setUp(self):
        """Default set-up used by all the tests

        Particles with the following coordinates (x, y, z) are placed in a box

        0 : -1.5 , 0.25 , 0.5
        1 : 0.33 , -0.25, 0.25
        2 : 1.25 , -1.25, 1.25
        3 : 0.05 , 1.25 , -0.5
        4 : -0.5 , 0.5  , -1.25
        5 : -0.75, 0.75 , -1.25
        6 : -1.25, 0.5  , 0.5
        7 : 0.5  , 1.5  , -0.5
        8 : 0.5  , -0.5 , 0.5
        9 : 0.5  , 1.75 , -0.75

        The cell size is set to 1. Valid cell indices and the
        particles they contain are given below:

        (-2, 0, 0) : particle 0, 6
        (0, -1, 0) : particle 1, 8
        (1, -2, 1) : particle 2
        (0, 1, -1) : particle 3, 7, 9
        (-1, 0, -2): particle 4, 5

        """
        x = numpy.array([
            -1.5, 0.33, 1.25, 0.05, -0.5, -0.75, -1.25, 0.5, 0.5, 0.5])

        y = numpy.array([
            0.25, -0.25, -1.25, 1.25, 0.5, 0.75, 0.5, 1.5, -0.5, 1.75])

        z = numpy.array([
            0.5, 0.25, 1.25, -0.5, -1.25, -1.25, 0.5, -0.5, 0.5, -0.75])

        # using a degenrate (h=0) array will set cell size to 1 for NNPS
        h = numpy.zeros_like(x)

        pa = get_particle_array(x=x, y=y, z=z, h=h)

        self.dict_box_sort_nnps = nnps.DictBoxSortNNPS(
            dim=3, particles=[pa], radius_scale=1.0
        )

        self.box_sort_nnps = nnps.BoxSortNNPS(
            dim=3, particles=[pa], radius_scale=1.0
        )

        self.ll_nnps = nnps.LinkedListNNPS(
            dim=3, particles=[pa], radius_scale=1.0
        )

        self.sp_hash_nnps = nnps.SpatialHashNNPS(
            dim=3, particles=[pa], radius_scale=1.0
        )

        self.ext_sp_hash_nnps = nnps.ExtendedSpatialHashNNPS(
            dim=3, particles=[pa], radius_scale=1.0
        )

        self.strat_radius_nnps = nnps.StratifiedHashNNPS(
            dim=3, particles=[pa], radius_scale=1.0
        )

        # these are the expected cells
        self.expected_cells = {
            IntPoint(-2, 0, 0): [0, 6],
            IntPoint(0, -1, 0): [1, 8],
            IntPoint(1, -2, 1): [2],
            IntPoint(0, 1, -1): [3, 7, 9],
            IntPoint(-1, 0, -2): [4, 5]
        }

    def test_cell_size(self):
        "SimpleNNPS :: test cell_size"
        nnps = self.dict_box_sort_nnps
        self.assertAlmostEqual(nnps.cell_size, 1.0, 14)

        nnps = self.box_sort_nnps
        self.assertAlmostEqual(nnps.cell_size, 1.0, 14)

        nnps = self.ll_nnps
        self.assertAlmostEqual(nnps.cell_size, 1.0, 14)

        nnps = self.sp_hash_nnps
        self.assertAlmostEqual(nnps.cell_size, 1.0, 14)

        nnps = self.ext_sp_hash_nnps
        self.assertAlmostEqual(nnps.cell_size, 1.0, 14)

        nnps = self.strat_radius_nnps
        self.assertAlmostEqual(nnps.cell_size, 1.0, 14)

    def test_cells(self):
        "SimpleNNPS :: test cells"
        nnps = self.dict_box_sort_nnps
        cells = self.expected_cells

        # check each cell for it's contents
        for key in cells:
            self.assertTrue(key in nnps.cells)

            cell = nnps.cells.get(key)

            cell_indices = list(cell.lindices[0].get_npy_array())
            expected_indices = cells.get(key)

            self.assertTrue(cell_indices == expected_indices)


class NNPS2DTestCase(unittest.TestCase):
    def setUp(self):
        """Default set-up used by all the tests

        Two sets of particle arrays (a & b) are created and neighbors
        are checked from a -> b, b -> a , a -> a and b -> b

        """
        numpy.random.seed(123)
        self.numPoints1 = numPoints1 = 1 << 11
        self.numPoints2 = numPoints2 = 1 << 10

        self.pa1 = pa1 = self._create_random(numPoints1)
        self.pa2 = pa2 = self._create_random(numPoints2)

        # the list of particles
        self.particles = [pa1, pa2]

    def _create_random(self, numPoints):
        # average particle spacing and volume in the unit cube
        dx = pow(1.0 / numPoints, 1. / 2.)

        # create random points in the interval [-1, 1]^3
        x1, y1 = random.random((2, numPoints)) * 2.0 - 1.0
        z1 = numpy.zeros_like(x1)

        h1 = numpy.ones_like(x1) * 1.2 * dx
        gid1 = numpy.arange(numPoints).astype(numpy.uint32)

        # first particle array
        pa = get_particle_array(
            x=x1, y=y1, z=z1, h=h1, gid=gid1)

        return pa

    def _assert_neighbors(self, nbrs_nnps, nbrs_brute_force):
        # ensure that the lengths of the arrays are the same
        self.assertEqual(nbrs_nnps.length, nbrs_brute_force.length)
        nnbrs = nbrs_nnps.length

        _nbrs1 = nbrs_nnps.get_npy_array()
        _nbrs2 = nbrs_brute_force.get_npy_array()

        # sort the neighbors
        nbrs1 = _nbrs1[:nnbrs]
        nbrs1.sort()
        nbrs2 = _nbrs2
        nbrs2.sort()

        # check each neighbor
        for i in range(nnbrs):
            self.assertEqual(nbrs1[i], nbrs2[i])

    def _test_neighbors_by_particle(self, src_index, dst_index, dst_numPoints):
        # nnps and the two neighbor lists
        nps = self.nps
        nbrs1 = UIntArray()
        nbrs2 = UIntArray()

        nps.set_context(src_index, dst_index)

        # get the neighbors and sort the result
        for i in range(dst_numPoints):
            nps.get_nearest_particles(src_index, dst_index, i, nbrs1)
            nps.brute_force_neighbors(src_index, dst_index, i, nbrs2)

            # ensure that the neighbor lists are the same
            self._assert_neighbors(nbrs1, nbrs2)


class DictBoxSortNNPS2DTestCase(NNPS2DTestCase):
    """Test for the original box-sort algorithm"""

    def setUp(self):
        NNPS2DTestCase.setUp(self)
        self.nps = nnps.DictBoxSortNNPS(
            dim=2, particles=self.particles, radius_scale=2.0
        )

    def test_neighbors_aa(self):
        self._test_neighbors_by_particle(src_index=0, dst_index=0,
                                         dst_numPoints=self.numPoints1)

    def test_neighbors_ab(self):
        self._test_neighbors_by_particle(src_index=0, dst_index=1,
                                         dst_numPoints=self.numPoints2)

    def test_neighbors_ba(self):
        self._test_neighbors_by_particle(src_index=1, dst_index=0,
                                         dst_numPoints=self.numPoints1)

    def test_neighbors_bb(self):
        self._test_neighbors_by_particle(src_index=1, dst_index=1,
                                         dst_numPoints=self.numPoints2)

    def test_repeated(self):
        self.test_neighbors_aa()
        self.test_neighbors_ab()
        self.test_neighbors_ba()
        self.test_neighbors_bb()


class OctreeGPUNNPS2DTestCase(DictBoxSortNNPS2DTestCase):
    """Test for Z-Order SFC based OpenCL algorithm"""

    def setUp(self):
        NNPS2DTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps

        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = False

        self.nps = gpu_nnps.OctreeGPUNNPS(
            dim=2, particles=self.particles, radius_scale=2.0,
            backend='opencl'
        )

    def tearDown(self):
        super(OctreeGPUNNPS2DTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class OctreeGPUNNPSDouble2DTestCase(DictBoxSortNNPS2DTestCase):
    """Test for Z-Order SFC based OpenCL algorithm"""

    def setUp(self):
        NNPS2DTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = True

        self.nps = gpu_nnps.OctreeGPUNNPS(
            dim=2, particles=self.particles, radius_scale=2.0,
            backend='opencl'
        )

    def tearDown(self):
        super(OctreeGPUNNPSDouble2DTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class NNPSTestCase(unittest.TestCase):
    """Standard nearest neighbor queries and comparison with the brute
    force approach.

    We randomly distribute particles in 3-space and compare the list
    of neighbors using the NNPS algorithms and the brute force
    approach.

    The following particle arrays are set up for testing
    1) pa1, pa2: uniformly distributed distribution with a constant h. pa1
    and p2 have a different number of particles and hence, a different h.
    2) pa3: Uniformly distributed distribution for both the coordinates and
    for h.
    3) pa4: h varies along with spatial coordinates.
    """

    def setUp(self):
        """Default set-up used by all the tests
        """
        numpy.random.seed(123)
        # Datasets with constant h
        self.numPoints1 = numPoints1 = 1 << 11
        self.numPoints2 = numPoints2 = 1 << 10

        # Datasets with varying h
        self.numPoints3 = numPoints3 = 1 << 10

        # FIXME: Tets fail with m4=9
        # Looks like the issue arises due to rounding errors which should be
        # acceptable to a degree. Need to modify tests or brute force NNPS to
        # handle such cases appropriately
        m4 = 8
        self.numPoints4 = numPoints4 = m4 ** 3

        self.pa1 = pa1 = self._create_random(numPoints1)
        self.pa2 = pa2 = self._create_random(numPoints2)
        self.pa3 = pa3 = self._create_random_variable_h(numPoints3)
        self.pa4 = pa4 = self._create_linear_radius(0.1, 0.4, m4)

        # the list of particles
        self.particles = [pa1, pa2, pa3, pa4]

    def _create_random(self, numPoints):
        # average particle spacing and volume in the unit cube
        dx = pow(1.0 / numPoints, 1. / 3.)

        # create random points in the interval [-1, 1]^3
        x1, y1, z1 = random.random((3, numPoints)) * 2.0 - 1.0
        h1 = numpy.ones_like(x1) * 1.2 * dx
        gid1 = numpy.arange(numPoints).astype(numpy.uint32)

        # first particle array
        pa = get_particle_array(
            x=x1, y=y1, z=z1, h=h1, gid=gid1)

        return pa

    def _create_linear_radius(self, dx_min, dx_max, m):
        n = m ** 3
        base = numpy.linspace(1., (dx_max / dx_min), m)
        hl = base * dx_min
        xl = numpy.cumsum(hl)
        x, y, z = numpy.meshgrid(xl, xl, xl)
        x, y, z = x.ravel(), y.ravel(), z.ravel()
        h1, h2, h3 = numpy.meshgrid(hl, hl, hl)
        h = (h1 ** 2 + h2 ** 2 + h3 ** 2) ** 0.5
        h = h.ravel()
        gid = numpy.arange(n).astype(numpy.uint32)

        pa = get_particle_array(
            x=x, y=y, z=z, h=h, gid=gid
        )

        return pa

    def _create_random_variable_h(self, num_points):
        # average particle spacing and volume in the unit cube
        dx = pow(1.0 / num_points, 1. / 3.)

        # create random points in the interval [-1, 1]^3
        x1, y1, z1 = random.random((3, num_points)) * 2.0 - 1.0
        h1 = numpy.ones_like(x1) * \
            numpy.random.uniform(1, 4, size=num_points) * 1.2 * dx
        gid1 = numpy.arange(num_points).astype(numpy.uint32)

        # first particle array
        pa = get_particle_array(
            x=x1, y=y1, z=z1, h=h1, gid=gid1)

        return pa

    def _assert_neighbors(self, nbrs_nnps, nbrs_brute_force):
        # ensure that the lengths of the arrays are the same
        if nbrs_nnps.length != nbrs_brute_force.length:
            print(nbrs_nnps.get_npy_array(), nbrs_brute_force.get_npy_array())
        self.assertEqual(nbrs_nnps.length, nbrs_brute_force.length)
        nnbrs = nbrs_nnps.length

        _nbrs1 = nbrs_nnps.get_npy_array()
        _nbrs2 = nbrs_brute_force.get_npy_array()

        # sort the neighbors
        nbrs1 = _nbrs1[:nnbrs]
        nbrs1.sort()
        nbrs2 = _nbrs2
        nbrs2.sort()

        # check each neighbor
        for i in range(nnbrs):
            self.assertEqual(nbrs1[i], nbrs2[i])

    def _test_neighbors_by_particle(self, src_index, dst_index, dst_numPoints):
        # nnps and the two neighbor lists
        nps = self.nps
        nbrs1 = UIntArray()
        nbrs2 = UIntArray()

        nps.set_context(src_index, dst_index)

        # get the neighbors and sort the result
        for i in range(dst_numPoints):
            nps.get_nearest_particles(src_index, dst_index, i, nbrs1)
            nps.brute_force_neighbors(src_index, dst_index, i, nbrs2)

            # ensure that the neighbor lists are the same

            self._assert_neighbors(nbrs1, nbrs2)


class DictBoxSortNNPSTestCase(NNPSTestCase):
    """Test for the original box-sort algorithm"""

    def setUp(self):
        """
        Default setup and tests used for 3D NNPS tests

        We run the tests on the following pairs of particle arrays:

        Set 1) Same particle arrays. Both have constant h.
        1) a -> a
        2) b -> b

        Set 2) Different particle arrays with constant h.
        1) a -> b
        2) b -> a

        Set 3) Variable h
        1) c -> c
        2) d -> d

        We then repeat the above tests again to ensure that we get the
        correct results even when running NNPS repeatedly
        """
        NNPSTestCase.setUp(self)
        self.nps = nnps.DictBoxSortNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )

    def test_neighbors_aa(self):
        self._test_neighbors_by_particle(src_index=0, dst_index=0,
                                         dst_numPoints=self.numPoints1)

    def test_neighbors_ab(self):
        self._test_neighbors_by_particle(src_index=0, dst_index=1,
                                         dst_numPoints=self.numPoints2)

    def test_neighbors_ba(self):
        self._test_neighbors_by_particle(src_index=1, dst_index=0,
                                         dst_numPoints=self.numPoints1)

    def test_neighbors_bb(self):
        self._test_neighbors_by_particle(src_index=1, dst_index=1,
                                         dst_numPoints=self.numPoints2)

    def test_neighbors_cc(self):
        self._test_neighbors_by_particle(src_index=2, dst_index=2,
                                         dst_numPoints=self.numPoints3)

    def test_neighbors_dd(self):
        self._test_neighbors_by_particle(src_index=3, dst_index=3,
                                         dst_numPoints=self.numPoints4)

    def test_repeated(self):
        self.test_neighbors_aa()
        self.test_neighbors_ab()
        self.test_neighbors_ba()
        self.test_neighbors_bb()
        self.test_neighbors_cc()
        self.test_neighbors_dd()


class BoxSortNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for the original box-sort algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.BoxSortNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )


class SpatialHashNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Spatial Hash algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.SpatialHashNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )


class SingleLevelStratifiedHashNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Stratified hash algorithm with num_levels = 1"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.StratifiedHashNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )


class MultipleLevelsStratifiedHashNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Stratified hash algorithm with num_levels = 2"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.StratifiedHashNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            num_levels=2
        )


class SingleLevelStratifiedSFCNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Stratified SFC algorithm with num_levels = 1"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.StratifiedSFCNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )


class MultipleLevelsStratifiedSFCNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Stratified SFC algorithm with num_levels = 2"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.StratifiedSFCNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            num_levels=2
        )


class ExtendedSpatialHashNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Extended Spatial Hash algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.ExtendedSpatialHashNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )


class OctreeNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Octree based algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.OctreeNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )


class CellIndexingNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Cell Indexing based algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.CellIndexingNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )


class ZOrderNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Z-Order SFC based algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.ZOrderNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )


class ZOrderGPUNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Z-Order SFC based OpenCL algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = False

        self.nps = gpu_nnps.ZOrderGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            backend='opencl'
        )

    def tearDown(self):
        super(ZOrderGPUNNPSTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class ZOrderGPUNNPSTestCaseCUDA(ZOrderGPUNNPSTestCase):
    def setUp(self):
        NNPSTestCase.setUp(self)
        importorskip("pycuda")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = False

        self.nps = gpu_nnps.ZOrderGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            backend='cuda'
        )

    def tearDown(self):
        super(ZOrderGPUNNPSTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class BruteForceNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for OpenCL brute force algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = False

        self.nps = gpu_nnps.BruteForceNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            backend='opencl'
        )

    def tearDown(self):
        super(BruteForceNNPSTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class OctreeGPUNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Z-Order SFC based OpenCL algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = False

        self.nps = gpu_nnps.OctreeGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            backend='opencl'
        )

    def tearDown(self):
        super(OctreeGPUNNPSTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class ZOrderGPUDoubleNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Z-Order SFC based OpenCL algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = True
        self.nps = gpu_nnps.ZOrderGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            backend='opencl'
        )

    def tearDown(self):
        super(ZOrderGPUDoubleNNPSTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class ZOrderGPUDoubleNNPSTestCaseCUDA(ZOrderGPUDoubleNNPSTestCase):
    """Test for Z-Order SFC based OpenCL algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        importorskip("pycuda")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = True
        self.nps = gpu_nnps.ZOrderGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            backend='cuda'
        )

    def tearDown(self):
        super(ZOrderGPUDoubleNNPSTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class OctreeGPUDoubleNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Octree based OpenCL algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = True
        self.nps = gpu_nnps.OctreeGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            backend='opencl'
        )

    def tearDown(self):
        super(OctreeGPUDoubleNNPSTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class TestZOrderGPUNNPSWithSorting(DictBoxSortNNPSTestCase):
    def setUp(self):
        NNPSTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = False
        self.nps = gpu_nnps.ZOrderGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            backend='opencl'
        )
        self.nps.spatially_order_particles(0)
        self.nps.spatially_order_particles(1)

        for pa in self.particles:
            pa.gpu.pull()

    def tearDown(self):
        super(TestZOrderGPUNNPSWithSorting, self).tearDown()
        get_config().use_double = self._orig_use_double


class TestZOrderGPUNNPSWithSortingCUDA(TestZOrderGPUNNPSWithSorting):
    def setUp(self):
        NNPSTestCase.setUp(self)
        cl = importorskip("pycuda")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = False
        self.nps = gpu_nnps.ZOrderGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            backend='cuda'
        )
        self.nps.spatially_order_particles(0)
        self.nps.spatially_order_particles(1)

        for pa in self.particles:
            pa.gpu.pull()

    def tearDown(self):
        super(TestZOrderGPUNNPSWithSorting, self).tearDown()
        get_config().use_double = self._orig_use_double


class OctreeGPUNNPSWithSortingTestCase(DictBoxSortNNPSTestCase):
    def setUp(self):
        NNPSTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = False
        self.nps = gpu_nnps.OctreeGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            backend='opencl'
        )
        self.nps.spatially_order_particles(0)
        self.nps.spatially_order_particles(1)

        for pa in self.particles:
            pa.gpu.pull()

    def tearDown(self):
        super(OctreeGPUNNPSWithSortingTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class OctreeGPUNNPSWithPartitioningTestCase(DictBoxSortNNPSTestCase):
    def setUp(self):
        NNPSTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps
        cfg = get_config()
        self._orig_use_double = cfg.use_double
        cfg.use_double = False
        self.nps = gpu_nnps.OctreeGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            use_partitions=True, backend='opencl'
        )

        for pa in self.particles:
            pa.gpu.pull()

    def tearDown(self):
        super(OctreeGPUNNPSWithPartitioningTestCase, self).tearDown()
        get_config().use_double = self._orig_use_double


class StratifiedSFCGPUNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Stratified SFC based OpenCL algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        cl = importorskip("pyopencl")
        from pysph.base import gpu_nnps

        self.nps = gpu_nnps.StratifiedSFCGPUNNPS(
            dim=3, particles=self.particles, radius_scale=2.0,
            num_levels=2, backend='opencl'
        )

    @pytest.mark.xfail(reason="StratifiedSFCGPUNNPS failing for \
                       variable h cases")
    def test_neighbors_dd(self):
        self._test_neighbors_by_particle(src_index=3, dst_index=3,
                                         dst_numPoints=self.numPoints4)

    @pytest.mark.xfail(reason="StratifiedSFCGPUNNPS failing for \
                       variable h cases")
    def test_repeated(self):
        self.test_neighbors_aa()
        self.test_neighbors_ab()
        self.test_neighbors_ba()
        self.test_neighbors_bb()
        self.test_neighbors_cc()
        self.test_neighbors_dd()


class CompressedOctreeNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Compressed Octree based algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.CompressedOctreeNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )


class LinkedListNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for the original box-sort algorithm"""

    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.LinkedListNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )

    def test_cell_index_positivity(self):
        nps = self.nps
        ncells_tot = nps.ncells_tot
        ncells_per_dim = nps.ncells_per_dim
        dim = nps.dim

        # cell indices should be positive. We iterate over the
        # flattened indices, get the unflattened version and check
        # that each component remains positive
        for cell_index in range(ncells_tot):
            cid = nnps.py_unflatten(cell_index, ncells_per_dim, dim)

            self.assertTrue(cid.x > -1)
            self.assertTrue(cid.y > -1)
            self.assertTrue(cid.z > -1)


class TestNNPSOnLargeDomain(unittest.TestCase):
    def _make_particles(self, nx=20):
        x, y, z = numpy.random.random((3, nx, nx, nx))
        x = numpy.ravel(x)
        y = numpy.ravel(y)
        z = numpy.ravel(z)
        h = numpy.ones_like(x) * 1.3 / nx

        pa = get_particle_array(name='fluid', x=x, y=y, z=z, h=h)
        # Place one particle far far away
        # On Linux and OSX this works even if sz is 100000.
        # However, on Windows this fails but works with 1000,
        # hence we set it to 1000.
        sz = 1000.0
        pa.add_particles(x=[sz], y=[sz], z=[sz])
        return pa

    def test_linked_list_nnps_raises_exception_for_large_domain(self):
        # Given/When
        pa = self._make_particles(20)
        # Then
        self.assertRaises(
            RuntimeError, nnps.LinkedListNNPS, dim=3, particles=[pa],
            cache=True
        )

    def test_box_sort_works_for_large_domain(self):
        # Given
        pa = self._make_particles(20)
        # We turn on cache so it computes all the neighbors quickly for us.
        nps = nnps.BoxSortNNPS(dim=3, particles=[pa], cache=True)
        nbrs = UIntArray()
        direct = UIntArray()
        nps.set_context(0, 0)
        for i in range(pa.get_number_of_particles()):
            nps.get_nearest_particles(0, 0, i, nbrs)
            nps.brute_force_neighbors(0, 0, i, direct)
            x = nbrs.get_npy_array()
            y = direct.get_npy_array()
            x.sort()
            y.sort()
            assert numpy.all(x == y)

    def test_spatial_hash_works_for_large_domain(self):
        # Given
        pa = self._make_particles(20)
        # We turn on cache so it computes all the neighbors quickly for us.
        nps = nnps.SpatialHashNNPS(dim=3, particles=[pa], cache=True)
        nbrs = UIntArray()
        direct = UIntArray()
        nps.set_context(0, 0)
        for i in range(pa.get_number_of_particles()):
            nps.get_nearest_particles(0, 0, i, nbrs)
            nps.brute_force_neighbors(0, 0, i, direct)
            x = nbrs.get_npy_array()
            y = direct.get_npy_array()
            x.sort()
            y.sort()
            assert numpy.all(x == y)

    def test_extended_spatial_hash_works_for_large_domain(self):
        # Given
        pa = self._make_particles(20)
        # We turn on cache so it computes all the neighbors quickly for us.
        nps = nnps.ExtendedSpatialHashNNPS(dim=3, particles=[pa], cache=True)
        nbrs = UIntArray()
        direct = UIntArray()
        nps.set_context(0, 0)
        for i in range(pa.get_number_of_particles()):
            nps.get_nearest_particles(0, 0, i, nbrs)
            nps.brute_force_neighbors(0, 0, i, direct)
            x = nbrs.get_npy_array()
            y = direct.get_npy_array()
            x.sort()
            y.sort()
            assert numpy.all(x == y)

    def test_octree_works_for_large_domain(self):
        # Given
        pa = self._make_particles(20)
        # We turn on cache so it computes all the neighbors quickly for us.
        nps = nnps.OctreeNNPS(dim=3, particles=[pa], cache=True)
        nbrs = UIntArray()
        direct = UIntArray()
        nps.set_context(0, 0)
        for i in range(pa.get_number_of_particles()):
            nps.get_nearest_particles(0, 0, i, nbrs)
            nps.brute_force_neighbors(0, 0, i, direct)
            x = nbrs.get_npy_array()
            y = direct.get_npy_array()
            x.sort()
            y.sort()
            assert numpy.all(x == y)

    def test_compressed_octree_works_for_large_domain(self):
        # Given
        pa = self._make_particles(20)
        # We turn on cache so it computes all the neighbors quickly for us.
        nps = nnps.CompressedOctreeNNPS(dim=3, particles=[pa], cache=True)
        nbrs = UIntArray()
        direct = UIntArray()
        nps.set_context(0, 0)
        for i in range(pa.get_number_of_particles()):
            nps.get_nearest_particles(0, 0, i, nbrs)
            nps.brute_force_neighbors(0, 0, i, direct)
            x = nbrs.get_npy_array()
            y = direct.get_npy_array()
            x.sort()
            y.sort()
            assert numpy.all(x == y)


class TestLinkedListNNPSWithSorting(unittest.TestCase):
    def _make_particles(self, nx=20):
        x = numpy.linspace(0, 1, nx)
        h = numpy.ones_like(x) / (nx - 1)

        pa = get_particle_array(name='fluid', x=x, h=h)
        nps = nnps.LinkedListNNPS(dim=1, particles=[pa], sort_gids=True)
        return pa, nps

    def test_nnps_sorts_without_gids(self):
        # Given
        pa, nps = self._make_particles(10)

        # When
        nps.set_context(0, 0)
        # Test the that gids are actually huge and invalid.
        self.assertEqual(numpy.max(pa.gid), numpy.min(pa.gid))
        self.assertTrue(numpy.max(pa.gid) > pa.gid.size)

        # Then
        nbrs = UIntArray()
        for i in range(pa.get_number_of_particles()):
            nps.get_nearest_particles(0, 0, i, nbrs)
            nb = nbrs.get_npy_array()
            sorted_nbrs = nb.copy()
            sorted_nbrs.sort()
            self.assertTrue(numpy.all(nb == sorted_nbrs))

    def test_nnps_sorts_with_valid_gids(self):
        # Given
        pa, nps = self._make_particles(10)
        pa.gid[:] = numpy.arange(pa.x.size)
        nps.update()

        # When
        nps.set_context(0, 0)
        # Test the that gids are actually valid.
        self.assertEqual(numpy.max(pa.gid), pa.gid.size - 1)
        self.assertEqual(numpy.min(pa.gid), 0)

        # Then
        nbrs = UIntArray()
        for i in range(pa.get_number_of_particles()):
            nps.get_nearest_particles(0, 0, i, nbrs)
            nb = nbrs.get_npy_array()
            sorted_nbrs = nb.copy()
            sorted_nbrs.sort()
            self.assertTrue(numpy.all(nb == sorted_nbrs))


class TestSpatialHashNNPSWithSorting(TestLinkedListNNPSWithSorting):
    def _make_particles(self, nx=20):
        x = numpy.linspace(0, 1, nx)
        h = numpy.ones_like(x) / (nx - 1)

        pa = get_particle_array(name='fluid', x=x, h=h)
        nps = nnps.SpatialHashNNPS(dim=1, particles=[pa], sort_gids=True)
        return pa, nps


class TestMultipleLevelsStratifiedHashNNPSWithSorting(
        TestLinkedListNNPSWithSorting):
    def _make_particles(self, nx=20):
        x = numpy.linspace(0, 1, nx)
        h = numpy.ones_like(x) / (nx - 1)

        pa = get_particle_array(name='fluid', x=x, h=h)
        nps = nnps.StratifiedHashNNPS(dim=1, particles=[pa], num_levels=2,
                                      sort_gids=True)
        return pa, nps


class TestMultipleLevelsStratifiedSFCNNPSWithSorting(
        TestLinkedListNNPSWithSorting):
    def _make_particles(self, nx=20):
        x = numpy.linspace(0, 1, nx)
        h = numpy.ones_like(x) / (nx - 1)

        pa = get_particle_array(name='fluid', x=x, h=h)
        nps = nnps.StratifiedSFCNNPS(dim=1, particles=[pa], num_levels=2,
                                     sort_gids=True)
        return pa, nps


def test_large_number_of_neighbors_linked_list():
    x = numpy.random.random(1 << 14) * 0.1
    y = x.copy()
    z = x.copy()
    h = numpy.ones_like(x)
    pa = get_particle_array(name='fluid', x=x, y=y, z=z, h=h)

    nps = nnps.LinkedListNNPS(dim=3, particles=[pa], cache=False)
    nbrs = UIntArray()
    nps.get_nearest_particles(0, 0, 0, nbrs)
    # print(nbrs.length)
    assert nbrs.length == len(x)


def test_neighbor_cache_update_doesnt_leak():
    # Given
    x, y, z = numpy.random.random((3, 1000))
    pa = get_particle_array(name='fluid', x=x, y=y, z=z, h=0.05)

    nps = nnps.LinkedListNNPS(dim=3, particles=[pa], cache=True)
    nps.set_context(0, 0)
    nps.cache[0].find_all_neighbors()
    old_length = sum(x.length for x in nps.cache[0]._neighbor_arrays)

    # When
    nps.update()
    nps.set_context(0, 0)
    nps.cache[0].find_all_neighbors()

    # Then
    new_length = sum(x.length for x in nps.cache[0]._neighbor_arrays)
    assert new_length == old_length


nnps_classes = [
    nnps.BoxSortNNPS,
    nnps.CellIndexingNNPS,
    nnps.CompressedOctreeNNPS,
    nnps.ExtendedSpatialHashNNPS,
    nnps.LinkedListNNPS,
    nnps.OctreeNNPS,
    nnps.SpatialHashNNPS,
    nnps.StratifiedHashNNPS,
    nnps.StratifiedSFCNNPS,
    nnps.ZOrderNNPS
]


@pytest.mark.parametrize("cls", nnps_classes)
def test_corner_case_1d_few_cells(cls):
    x, y, z = [0.131, 0.359], [1.544, 1.809], [-3.6489999, -2.8559999]
    pa = get_particle_array(name='fluid', x=x, y=y, z=z, h=1.0)
    nbrs = UIntArray()
    bf_nbrs = UIntArray()
    nps = cls(dim=3, particles=[pa], radius_scale=0.7)
    for i in range(2):
        nps.get_nearest_particles(0, 0, i, nbrs)
        nps.brute_force_neighbors(0, 0, i, bf_nbrs)
        assert sorted(nbrs) == sorted(bf_nbrs), 'Failed for particle: %d' % i


def test_use_2d_for_1d_data_with_llnps():
    y = numpy.array([1.0, 1.5])
    h = numpy.ones_like(y)
    pa = get_particle_array(name='fluid', y=y, h=h)
    nps = nnps.LinkedListNNPS(dim=2, particles=[pa], cache=False)
    nbrs = UIntArray()
    nps.get_nearest_particles(0, 0, 0, nbrs)
    print(nbrs.length)
    assert nbrs.length == len(y)


def test_use_3d_for_1d_data_with_llnps():
    y = numpy.array([1.0, 1.5])
    h = numpy.ones_like(y)
    pa = get_particle_array(name='fluid', y=y, h=h)
    nps = nnps.LinkedListNNPS(dim=3, particles=[pa], cache=False)
    nbrs = UIntArray()
    nps.get_nearest_particles(0, 0, 0, nbrs)
    print(nbrs.length)
    assert nbrs.length == len(y)


def test_large_number_of_neighbors_spatial_hash():
    x = numpy.random.random(1 << 14) * 0.1
    y = x.copy()
    z = x.copy()
    h = numpy.ones_like(x)
    pa = get_particle_array(name='fluid', x=x, y=y, z=z, h=h)

    nps = nnps.SpatialHashNNPS(dim=3, particles=[pa], cache=False)
    nbrs = UIntArray()
    nps.get_nearest_particles(0, 0, 0, nbrs)
    # print(nbrs.length)
    assert nbrs.length == len(x)


def test_large_number_of_neighbors_octree():
    x = numpy.random.random(1 << 14) * 0.1
    y = x.copy()
    z = x.copy()
    h = numpy.ones_like(x)
    pa = get_particle_array(name='fluid', x=x, y=y, z=z, h=h)

    nps = nnps.OctreeNNPS(dim=3, particles=[pa], cache=False)
    nbrs = UIntArray()
    nps.get_nearest_particles(0, 0, 0, nbrs)
    # print(nbrs.length)
    assert nbrs.length == len(x)


def test_flatten_unflatten():
    # first consider the 2D case where we assume a 4 X 5 grid of cells
    dim = 2
    ncells_per_dim = IntArray(3)
    ncells_per_dim[0] = 4
    ncells_per_dim[1] = 5
    ncells_per_dim[2] = 0

    # valid un-flattened cell indices
    cids = [[i, j] for i in range(4) for j in range(5)]
    for _cid in cids:
        cid = IntPoint(_cid[0], _cid[1], 0)
        flattened = nnps.py_flatten(cid, ncells_per_dim, dim)
        unflattened = nnps.py_unflatten(flattened, ncells_per_dim, dim)

        # the unflattened index should match with cid
        assert (cid == unflattened)

    # 3D
    dim = 3
    ncells_per_dim = IntArray(3)
    ncells_per_dim[0] = 4
    ncells_per_dim[1] = 5
    ncells_per_dim[2] = 2

    # valid un-flattened indices
    cids = [[i, j, k] for i in range(4) for j in range(5) for k in range(2)]
    for _cid in cids:
        cid = IntPoint(_cid[0], _cid[1], _cid[2])
        flattened = nnps.py_flatten(cid, ncells_per_dim, dim)
        unflattened = nnps.py_unflatten(flattened, ncells_per_dim, dim)

        # the unflattened index should match with cid
        assert (cid == unflattened)


def test_1D_get_valid_cell_index():
    dim = 1

    # simulate a dummy distribution such that 10 cells are along the
    # 'x' direction
    n_cells = 10
    ncells_per_dim = IntArray(3)

    ncells_per_dim[0] = n_cells
    ncells_per_dim[1] = 1
    ncells_per_dim[2] = 1

    # target cell
    cx = 1
    cy = cz = 0

    # as long as cy and cz are 0, the function should return the valid
    # flattened cell index for the cell
    for i in [-1, 0, 1]:
        index = nnps.py_get_valid_cell_index(
            IntPoint(cx + i, cy, cz), ncells_per_dim, dim, n_cells)
        assert index != -1

    # index should be -1 whenever cy and cz are > 1. This is
    # specifically the case that was failing earlier.
    for j in [-1, 1]:
        for k in [-1, 1]:
            index = nnps.py_get_valid_cell_index(
                IntPoint(cx, cy + j, cz + k), ncells_per_dim, dim, n_cells)
            assert index == -1

    # When the cx > n_cells or < -1 it should be invalid
    for i in [-2, -1, n_cells, n_cells + 1]:
        index = nnps.py_get_valid_cell_index(
            IntPoint(i, cy, cz), ncells_per_dim, dim, n_cells)
        assert index == -1


def test_get_centroid():
    cell = nnps.Cell(IntPoint(0, 0, 0), cell_size=0.1, narrays=1)
    centroid = Point()
    cell.get_centroid(centroid)

    assert (abs(centroid.x - 0.05) < 1e-10)
    assert (abs(centroid.y - 0.05) < 1e-10)
    assert (abs(centroid.z - 0.05) < 1e-10)

    cell = nnps.Cell(IntPoint(1, 2, 3), cell_size=0.5, narrays=1)
    cell.get_centroid(centroid)

    assert (abs(centroid.x - 0.75) < 1e-10)
    assert (abs(centroid.y - 1.25) < 1e-10)
    assert (abs(centroid.z - 1.75) < 1e-10)


def test_get_bbox():
    cell_size = 0.1
    cell = nnps.Cell(IntPoint(0, 0, 0), cell_size=cell_size, narrays=1)
    centroid = Point()
    boxmin = Point()
    boxmax = Point()

    cell.get_centroid(centroid)
    cell.get_bounding_box(boxmin, boxmax)

    assert (abs(boxmin.x - (centroid.x - 1.5 * cell_size)) < 1e-10)
    assert (abs(boxmin.y - (centroid.y - 1.5 * cell_size)) < 1e-10)
    assert (abs(boxmin.z - (centroid.z - 1.5 * cell_size)) < 1e-10)

    assert (abs(boxmax.x - (centroid.x + 1.5 * cell_size)) < 1e-10)
    assert (abs(boxmax.y - (centroid.y + 1.5 * cell_size)) < 1e-10)
    assert (abs(boxmax.z - (centroid.z + 1.5 * cell_size)) < 1e-10)

    cell_size = 0.5
    cell = nnps.Cell(IntPoint(1, 2, 0), cell_size=cell_size, narrays=1)

    cell.get_centroid(centroid)
    cell.get_bounding_box(boxmin, boxmax)

    assert (abs(boxmin.x - (centroid.x - 1.5 * cell_size)) < 1e-10)
    assert (abs(boxmin.y - (centroid.y - 1.5 * cell_size)) < 1e-10)
    assert (abs(boxmin.z - (centroid.z - 1.5 * cell_size)) < 1e-10)

    assert (abs(boxmax.x - (centroid.x + 1.5 * cell_size)) < 1e-10)
    assert (abs(boxmax.y - (centroid.y + 1.5 * cell_size)) < 1e-10)
    assert (abs(boxmax.z - (centroid.z + 1.5 * cell_size)) < 1e-10)


if __name__ == '__main__':
    unittest.main()
