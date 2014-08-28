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
from pyzoltan.core.carray import UIntArray, IntArray, DoubleArray

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
        
        self.box_sort_nnps = nnps.BoxSortNNPS(
            dim=3, particles=[pa,], radius_scale=1.0, warn=False)

        self.ll_nnps = nnps.LinkedListNNPS(
            dim=3, particles=[pa,], radius_scale=1.0, warn=False)

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
        nnps = self.box_sort_nnps
        self.assertAlmostEqual( nnps.cell_size, 1.0, 14 )
        
        nnps = self.ll_nnps
        self.assertAlmostEqual( nnps.cell_size, 1.0, 14 )

    def test_cells(self):
        "SimpleNNPS :: test cells"
        nnps = self.box_sort_nnps
        cells = self.expected_cells

        # check each cell for it's contents
        for key in cells:
            self.assertTrue( nnps.cells.has_key(key) )
            
            cell = nnps.cells.get(key)

            cell_indices = list( cell.lindices[0].get_npy_array() )
            expected_indices = cells.get(key)
            
            self.assertTrue( cell_indices == expected_indices )

    def test_n_part_per_cell(self):
        "SimpleNNPS :: test count_n_part_per_cell "
        nnps = self.box_sort_nnps

        # call the function to count the number of particles
        nnps.count_n_part_per_cell()
        expected_cells = self.expected_cells
        
        n_part_per_cell_array = nnps.n_part_per_cell[0]
        nnps_keys = nnps.cells.keys()
        n_cells = nnps.n_cells
        
        for i in range(n_cells):
            key = nnps_keys[i]

            self.assertTrue(
                n_part_per_cell_array[i] == len(expected_cells[key]))

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

class BoxSortNNPSTestCase(NNPSTestCase):
    """Test for the original box-sort algorithm"""
    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.BoxSortNNPS(dim=3, particles=self.particles, radius_scale=2.0, warn=False)

    def test_neighbors_aa(self):
        """NNPS :: neighbor test src = a, dst = a """
        self._test_neighbors_by_particle(src_index=0, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_ab(self):
        """NNPS :: neighbor test src = a, dst = b """
        self._test_neighbors_by_particle(src_index=0, dst_index=1, dst_numPoints=self.numPoints2)

    def test_neighbors_ba(self):
        """NNPS :: neighbor test src = b, dst = a """
        self._test_neighbors_by_particle(src_index=1, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_bb(self):
        """NNPS :: neighbor test src = b, dst = b """
        self._test_neighbors_by_particle(src_index=1, dst_index=1, dst_numPoints=self.numPoints2)

    def test_neighbors_filtered_aa(self):
        """NNPS :: neighbor test by cell src = a, dst = a """
        self._test_neighbors_filtered(src_index=0, dst_index=0)

    def test_neighbors_filtered_ab(self):
        """NNPS :: neighbor test by cell src = a, dst = b """
        self._test_neighbors_filtered(src_index=0, dst_index=1)

    def test_neighbors_filtered_ba(self):
        """NNPS :: neighbor test by cell src = b, dst = a"""
        self._test_neighbors_filtered(src_index=1, dst_index=0)

    def test_neighbors_filtered_bb(self):
        """NNPS :: neighbor test by cell src = b, dst = b"""
        self._test_neighbors_filtered(src_index=1, dst_index=1)

    def _test_neighbors_filtered(self, src_index, dst_index):
        # nnps and the two neighbor lists
        nps = self.nps
        nbrs1 = UIntArray(); nbrs2 = UIntArray()

        potential_neighbors = UIntArray()
        cell_indices = UIntArray()

        # get the neighbors for each particle and compare with brute force
        ncells_tot = nps.get_number_of_cells()
        for cell_index in range(ncells_tot):

            # get the dst particlces in this cell
            nps.get_particles_in_cell(
                cell_index, dst_index, cell_indices)

            # get the potential neighbors for this cell
            nps.get_particles_in_neighboring_cells(
                cell_index, src_index, potential_neighbors)

            # now iterate over the particles in this cell and get the
            # neighbors
            for indexi in range( cell_indices.length ):
                particle_index = cell_indices[indexi]

                # NNPS neighbors
                nps.get_nearest_particles_filtered(
                    src_index, dst_index, particle_index,
                    potential_neighbors, nbrs1)

                # brute force neighbors
                nps.brute_force_neighbors(
                    src_index, dst_index, particle_index, nbrs2)

                # check the neighbors
                self._assert_neighbors(nbrs1, nbrs2)

class LinkedListNNPSTestCase(BoxSortNNPSTestCase):
    """Test for the original box-sort algorithm"""
    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.LinkedListNNPS(dim=3, particles=self.particles, radius_scale=2.0, warn=False)

    def test_cell_indices(self):
        """LinkedListNNPS :: test positivity for cell indices"""
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

def test_flatten_unflatten():
    """Test the flattening and un-flattening functions"""
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
    """Test 'get_centroid'"""
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
    """Test 'get_bounding_box'"""
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
