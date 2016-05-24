"""Unittests for the serial NNPS

You can run the tests like so:

    $ nosetests -v test_nnps.py
"""
import numpy
from numpy import random

# PySPH imports
from pysph.base.point import IntPoint, Point
from pysph.base.utils import get_particle_array
from pysph.base import nnps

# Carrays from PyZoltan
from pyzoltan.core.carray import UIntArray, IntArray

# Python testing framework
import unittest

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
            dim=3, particles=[pa,], radius_scale=1.0
        )

        self.box_sort_nnps = nnps.BoxSortNNPS(
            dim=3, particles=[pa,], radius_scale=1.0
        )

        self.ll_nnps = nnps.LinkedListNNPS(
            dim=3, particles=[pa,], radius_scale=1.0
        )

        self.sp_hash_nnps = nnps.SpatialHashNNPS(
            dim=3, particles=[pa,], radius_scale=1.0
        )

        self.ext_sp_hash_nnps = nnps.ExtendedSpatialHashNNPS(
                dim=3, particles=[pa,], radius_scale=1.0
        )

        # these are the expected cells
        self.expected_cells = {
            IntPoint(-2, 0, 0):[0,6],
            IntPoint(0, -1, 0):[1,8],
            IntPoint(1, -2, 1):[2,],
            IntPoint(0, 1, -1):[3, 7, 9],
            IntPoint(-1, 0, -2):[4, 5]
            }

    def test_cell_size(self):
        "SimpleNNPS :: test cell_size"
        nnps = self.dict_box_sort_nnps
        self.assertAlmostEqual( nnps.cell_size, 1.0, 14 )

        nnps = self.box_sort_nnps
        self.assertAlmostEqual( nnps.cell_size, 1.0, 14 )

        nnps = self.ll_nnps
        self.assertAlmostEqual( nnps.cell_size, 1.0, 14 )

        nnps = self.sp_hash_nnps
        self.assertAlmostEqual( nnps.cell_size, 1.0, 14 )

    def test_cells(self):
        "SimpleNNPS :: test cells"
        nnps = self.dict_box_sort_nnps
        cells = self.expected_cells

        # check each cell for it's contents
        for key in cells:
            self.assertTrue( key in nnps.cells )

            cell = nnps.cells.get(key)

            cell_indices = list( cell.lindices[0].get_npy_array() )
            expected_indices = cells.get(key)

            self.assertTrue( cell_indices == expected_indices )


class NNPSTestCase(unittest.TestCase):
    """Standard nearest neighbor queries and comparison with the brute
    force approach.

    We randomly distribute particles in 3-space and compare the list
    of neighbors using the NNPS algorithms and the brute force
    approach.

    """
    def setUp(self):
        """Default set-up used by all the tests

        Two sets of particle arrays (a & b) are created and neighbors
        are checked from a -> b, b -> a , a -> a and b -> b

        """
        self.numPoints1 = numPoints1 = 1 << 10
        self.numPoints2 = numPoints2 = 1 << 11

        self.pa1 = pa1 = self._create_random(numPoints1)
        self.pa2 = pa2 = self._create_random(numPoints2)

        # the list of particles
        self.particles = [pa1, pa2]

    def _create_random(self, numPoints):
        # average particle spacing and volume in the unit cube
        dx = pow( 1.0/numPoints, 1./3. )

        # create random points in the interval [-1, 1]^3
        x1, y1, z1 = random.random( (3, numPoints) ) * 2.0 - 1.0
        h1 = numpy.ones_like(x1) * 1.2 * dx
        gid1 = numpy.arange(numPoints).astype(numpy.uint32)

        # first particle array
        pa = get_particle_array(
            x=x1, y=y1, z=z1, h=h1, gid=gid1)

        return pa

    def _assert_neighbors(self, nbrs_nnps, nbrs_brute_force):
        # ensure that the lengths of the arrays are the same
        self.assertEqual( nbrs_nnps.length, nbrs_brute_force.length )
        nnbrs = nbrs_nnps.length

        _nbrs1 = nbrs_nnps.get_npy_array()
        _nbrs2 = nbrs_brute_force.get_npy_array()

        # sort the neighbors
        nbrs1 = _nbrs1[:nnbrs]; nbrs1.sort()
        nbrs2 = _nbrs2; nbrs2.sort()

        # check each neighbor
        for i in range(nnbrs):
            self.assertEqual( nbrs1[i], nbrs2[i] )

    def _test_neighbors_by_particle(
        self, src_index, dst_index, dst_numPoints):
        # nnps and the two neighbor lists
        nps = self.nps
        nbrs1 = UIntArray()
        nbrs2 = UIntArray()

        # get the neighbors and sort the result
        for i in range(dst_numPoints):
            nps.get_nearest_particles(src_index, dst_index, i, nbrs1)
            nps.brute_force_neighbors(src_index, dst_index, i, nbrs2)

            # ensure that the neighbor lists are the same
            self._assert_neighbors(nbrs1, nbrs2)


class DictBoxSortNNPSTestCase(NNPSTestCase):
    """Test for the original box-sort algorithm"""
    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.DictBoxSortNNPS(
            dim=3, particles=self.particles, radius_scale=2.0
        )

    def test_neighbors_aa(self):
        self._test_neighbors_by_particle(src_index=0, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_ab(self):
        self._test_neighbors_by_particle(src_index=0, dst_index=1, dst_numPoints=self.numPoints2)

    def test_neighbors_ba(self):
        self._test_neighbors_by_particle(src_index=1, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_bb(self):
        self._test_neighbors_by_particle(src_index=1, dst_index=1, dst_numPoints=self.numPoints2)


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

class ExtendedSpatialHashNNPSTestCase(DictBoxSortNNPSTestCase):
    """Test for Spatial Hash algorithm"""
    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.ExtendedSpatialHashNNPS(
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
            cid = nnps.py_unflatten( cell_index, ncells_per_dim, dim )

            self.assertTrue( cid.x > -1 )
            self.assertTrue( cid.y > -1 )
            self.assertTrue( cid.z > -1 )


class TestNNPSOnLargeDomain(unittest.TestCase):
    def _make_particles(self, nx=20):
        x, y, z = numpy.random.random((3, nx, nx, nx))
        x = numpy.ravel(x)
        y = numpy.ravel(y)
        z = numpy.ravel(z)
        h = numpy.ones_like(x)*1.3/nx

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
            RuntimeError, nnps.LinkedListNNPS, dim=3, particles=[pa], cache=True
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
            x.sort(); y.sort()
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
            x.sort(); y.sort()
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
            x.sort(); y.sort()
            assert numpy.all(x == y)


class TestLinkedListNNPSWithSorting(unittest.TestCase):
    def _make_particles(self, nx=20):
        x = numpy.linspace(0, 1, nx)
        h = numpy.ones_like(x)/(nx-1)

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
        self.assertEqual(numpy.max(pa.gid), pa.gid.size-1)
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
        h = numpy.ones_like(x)/(nx-1)

        pa = get_particle_array(name='fluid', x=x, h=h)
        nps = nnps.SpatialHashNNPS(dim=1, particles=[pa], sort_gids=True)
        return pa, nps

def test_large_number_of_neighbors_linked_list():
    x = numpy.random.random(1 << 14)*0.1
    y = x.copy()
    z = x.copy()
    h = numpy.ones_like(x)
    pa = get_particle_array(name='fluid', x=x, y=y, z=z, h=h)

    nps = nnps.LinkedListNNPS(dim=3, particles=[pa], cache=False)
    nbrs = UIntArray()
    nps.get_nearest_particles(0, 0, 0, nbrs)
    # print(nbrs.length)
    assert nbrs.length == len(x)

def test_large_number_of_neighbors_spatial_hash():
    x = numpy.random.random(1 << 14)*0.1
    y = x.copy()
    z = x.copy()
    h = numpy.ones_like(x)
    pa = get_particle_array(name='fluid', x=x, y=y, z=z, h=h)

    nps = nnps.SpatialHashNNPS(dim=3, particles=[pa], cache=False)
    nbrs = UIntArray()
    nps.get_nearest_particles(0, 0, 0, nbrs)
    # print(nbrs.length)
    assert nbrs.length == len(x)

def test_flatten_unflatten():
    # first consider the 2D case where we assume a 4 X 5 grid of cells
    dim = 2
    ncells_per_dim = IntArray(3)
    ncells_per_dim[0] = 4; ncells_per_dim[1] = 5; ncells_per_dim[2] = 0

    # valid un-flattened cell indices
    cids = [ [i, j] for i in range(4) for j in range(5) ]
    for _cid in cids:
        cid = IntPoint( _cid[0], _cid[1], 0)
        flattened = nnps.py_flatten( cid, ncells_per_dim, dim )
        unflattened = nnps.py_unflatten( flattened, ncells_per_dim, dim )

        # the unflattened index should match with cid
        assert( cid == unflattened )

    # 3D
    dim = 3
    ncells_per_dim = IntArray(3)
    ncells_per_dim[0] = 4; ncells_per_dim[1] = 5; ncells_per_dim[2] = 2

    # valid un-flattened indices
    cids = [ [i, j, k] for i in range(4) for j in range(5) for k in range(2) ]
    for _cid in cids:
        cid = IntPoint( _cid[0], _cid[1], _cid[2])
        flattened = nnps.py_flatten( cid, ncells_per_dim, dim )
        unflattened = nnps.py_unflatten( flattened, ncells_per_dim, dim )

        # the unflattened index should match with cid
        assert( cid == unflattened )


def test_1D_get_valid_cell_index():
    dim = 1

    # simulate a dummy distribution such that 10 cells are along the
    # 'x' direction
    n_cells = 10
    ncells_per_dim = IntArray(3)

    ncells_per_dim[0] = n_cells
    ncells_per_dim[1] = 0
    ncells_per_dim[2] = 0

    # target cell
    cx = 1; cy = cz = 0

    # as long as cy and cz are 0, the function should return the valid
    # flattened cell index for the cell
    for i in [-1, 0, 1]:
        index = nnps.py_get_valid_cell_index(
            IntPoint(cx+i, cy, cz), ncells_per_dim, dim, n_cells)
        assert index != -1

    # index should be -1 whenever cy and cz are > 1. This is
    # specifically the case that was failing earlier.
    for j in [-1,  1]:
        for k in [-1, 1]:
            index = nnps.py_get_valid_cell_index(
                IntPoint(cx,cy+j,cz+k), ncells_per_dim, dim, n_cells)
            assert index == -1


def test_get_centroid():
    cell = nnps.Cell(IntPoint(0, 0, 0), cell_size=0.1, narrays=1)
    centroid = Point()
    cell.get_centroid(centroid)

    assert(abs(centroid.x - 0.05) < 1e-10)
    assert(abs(centroid.y - 0.05) < 1e-10)
    assert(abs(centroid.z - 0.05) < 1e-10)

    cell = nnps.Cell(IntPoint(1, 2, 3), cell_size=0.5, narrays=1)
    cell.get_centroid(centroid)

    assert(abs(centroid.x - 0.75) < 1e-10)
    assert(abs(centroid.y - 1.25) < 1e-10)
    assert(abs(centroid.z - 1.75) < 1e-10)

def test_get_bbox():
    cell_size = 0.1
    cell = nnps.Cell(IntPoint(0, 0, 0), cell_size=cell_size, narrays=1)
    centroid = Point()
    boxmin = Point()
    boxmax = Point()

    cell.get_centroid(centroid)
    cell.get_bounding_box(boxmin, boxmax)

    assert(abs(boxmin.x - (centroid.x - 1.5*cell_size)) < 1e-10)
    assert(abs(boxmin.y - (centroid.y - 1.5*cell_size)) < 1e-10)
    assert(abs(boxmin.z - (centroid.z - 1.5*cell_size)) < 1e-10)

    assert(abs(boxmax.x - (centroid.x + 1.5*cell_size)) < 1e-10)
    assert(abs(boxmax.y - (centroid.y + 1.5*cell_size)) < 1e-10)
    assert(abs(boxmax.z - (centroid.z + 1.5*cell_size)) < 1e-10)

    cell_size = 0.5
    cell = nnps.Cell(IntPoint(1, 2, 0), cell_size=cell_size, narrays=1)

    cell.get_centroid(centroid)
    cell.get_bounding_box(boxmin, boxmax)

    assert(abs(boxmin.x - (centroid.x - 1.5*cell_size)) < 1e-10)
    assert(abs(boxmin.y - (centroid.y - 1.5*cell_size)) < 1e-10)
    assert(abs(boxmin.z - (centroid.z - 1.5*cell_size)) < 1e-10)

    assert(abs(boxmax.x - (centroid.x + 1.5*cell_size)) < 1e-10)
    assert(abs(boxmax.y - (centroid.y + 1.5*cell_size)) < 1e-10)
    assert(abs(boxmax.z - (centroid.z + 1.5*cell_size)) < 1e-10)

if __name__ == '__main__':
    unittest.main()
