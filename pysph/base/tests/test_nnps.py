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
        
        x1 = random.random( numPoints )
        y1 = random.random( numPoints )
        z1 = random.random( numPoints )
        h1 = numpy.ones_like(x1) * 1.2 * dx
        gid1 = numpy.arange(numPoints).astype(numpy.uint32)
        
        # first particle array
        pa = get_particle_array(
            x=x1, y=y1, z=z1, h=h1, gid=gid1)

        return pa

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

            # ensure that the lengths of the arrays are the same
            self.assertEqual( nbrs1._length, nbrs2.length )

            _nbrs1 = nbrs1.get_npy_array()
            _nbrs2 = nbrs2.get_npy_array()

            nnps_nbrs = _nbrs1[:nbrs1._length]; nnps_nbrs.sort()
            brut_nbrs = _nbrs2; brut_nbrs.sort()

            # check each neighbor
            for j in range(nbrs1._length):
                self.assertEqual( nnps_nbrs[j], brut_nbrs[j] )

    def _test_neighbors_filtered(self, src_index, dst_index):
        # nnps and the two neighbor lists
        nps = self.nps
        nbrs1 = UIntArray(); nbrs2 = UIntArray()

        potential_neighbors = UIntArray()
        cell_indices = UIntArray()

        # get the neighbors for each particle and compare with brute force
        ncells_tot = nps.ncells_tot
        for cell_index in range(ncells_tot):
            
            # get the dst particlces in this cell
            nps.get_particles_in_cell(
                cell_index, dst_index, cell_indices)

            # get the potential neighbors for this cell
            nps.get_particles_in_neighboring_cells(
                cell_index, src_index, potential_neighbors)

            # now iterate over the particles in this cell and get the
            # neighbors
            for indexi in range( cell_indices._length ):
                particle_index = cell_indices[indexi]

                # NNPS neighbors
                nps.get_nearest_particles_filtered(
                    src_index, dst_index, particle_index, 
                    potential_neighbors, nbrs1)

                # brute force neighbors
                nps.brute_force_neighbors(
                    src_index, dst_index, particle_index, nbrs2)

                # The rest is the same as before. We check for the
                # neighbor list for the particle 'particle_index'
                # ensure that the lengths of the arrays are the same
                self.assertEqual( nbrs1._length, nbrs2.length )
                
                _nbrs1 = nbrs1.get_npy_array()
                _nbrs2 = nbrs2.get_npy_array()

                nnps_nbrs = _nbrs1[:nbrs1._length]; nnps_nbrs.sort()
                brut_nbrs = _nbrs2; brut_nbrs.sort()

                # check each neighbor
                for j in range(nbrs1._length):
                    self.assertEqual( nnps_nbrs[j], brut_nbrs[j] )
        

class BoxSortNNPSTestCase(NNPSTestCase):
    """Test for the original box-sort algorithm"""
    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.BoxSortNNPS(dim=3, particles=self.particles, radius_scale=2.0)
        
    def test_neighbors_aa(self):
        """BoxSortNNPS :: neighbor test src = a, dst = a """
        self._test_neighbors_by_particle(src_index=0, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_ab(self):
        """BoxSortNNPS :: neighbor test src = a, dst = b """
        self._test_neighbors_by_particle(src_index=0, dst_index=1, dst_numPoints=self.numPoints2)

    def test_neighbors_ba(self):
        """BoxSortNNPS :: neighbor test src = b, dst = a """
        self._test_neighbors_by_particle(src_index=1, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_bb(self):
        """BoxSortNNPS :: neighbor test src = b, dst = b """
        self._test_neighbors_by_particle(src_index=1, dst_index=1, dst_numPoints=self.numPoints2)

class LinkedListNNPSTestCase(NNPSTestCase):
    """Test for the original box-sort algorithm"""
    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.LinkedListNNPS(dim=3, particles=self.particles, radius_scale=2.0)
        
    def test_neighbors_aa(self):
        """LinkedListNNPS :: neighbor test src = a, dst = a """
        self._test_neighbors_by_particle(src_index=0, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_ab(self):
        """LinkedListNNPS :: neighbor test src = a, dst = b """
        self._test_neighbors_by_particle(src_index=0, dst_index=1, dst_numPoints=self.numPoints2)

    def test_neighbors_ba(self):
        """LinkedListNNPS :: neighbor test src = b, dst = a """
        self._test_neighbors_by_particle(src_index=1, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_bb(self):
        """LinkedListNNPS :: neighbor test src = b, dst = b """
        self._test_neighbors_by_particle(src_index=1, dst_index=1, dst_numPoints=self.numPoints2)

    def test_neighbors_filtered_aa(self):
        """LinkedListNNPS :: neighbor test by cell src = a, dst = a """
        self._test_neighbors_filtered(src_index=0, dst_index=0)

    def test_neighbors_filtered_ab(self):
        """LinkedListNNPS :: neighbor test by cell src = a, dst = b """
        self._test_neighbors_filtered(src_index=0, dst_index=1)

    def test_neighbors_filtered_ba(self):
        """LinkedListNNPS :: neighbor test by cell src = b, dst = a"""
        self._test_neighbors_filtered(src_index=1, dst_index=0)

    def test_neighbors_filtered_bb(self):
        """LinkedListNNPS :: neighbor test by cell src = b, dst = b"""
        self._test_neighbors_filtered(src_index=1, dst_index=1)

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
