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
from pyzoltan.core.carray import UIntArray, DoubleArray

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

    def _test_neighbors(
        self, src_index, dst_index, dst_numPoints):
        """"""
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

class BoxSortNNPSTestCase(NNPSTestCase):
    """Test for the original box-sort algorithm"""
    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.BoxSortNNPS(dim=3, particles=self.particles, radius_scale=2.0)
        
    def test_neighbors_aa(self):
        """BoxSortNNPS :: neighbor test src = a, dst = a """
        self._test_neighbors(src_index=0, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_ab(self):
        """BoxSortNNPS :: neighbor test src = a, dst = b """
        self._test_neighbors(src_index=0, dst_index=1, dst_numPoints=self.numPoints2)

    def test_neighbors_ba(self):
        """BoxSortNNPS :: neighbor test src = b, dst = a """
        self._test_neighbors(src_index=1, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_bb(self):
        """BoxSortNNPS :: neighbor test src = b, dst = b """
        self._test_neighbors(src_index=1, dst_index=1, dst_numPoints=self.numPoints2)

class LinkedListNNPSTestCase(NNPSTestCase):
    """Test for the original box-sort algorithm"""
    def setUp(self):
        NNPSTestCase.setUp(self)
        self.nps = nnps.LinkedListNNPS(dim=3, particles=self.particles, radius_scale=2.0)
        
    def test_neighbors_aa(self):
        """LinkedListNNPS :: neighbor test src = a, dst = a """
        self._test_neighbors(src_index=0, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_ab(self):
        """LinkedListNNPS :: neighbor test src = a, dst = b """
        self._test_neighbors(src_index=0, dst_index=1, dst_numPoints=self.numPoints2)

    def test_neighbors_ba(self):
        """LinkedListNNPS :: neighbor test src = b, dst = a """
        self._test_neighbors(src_index=1, dst_index=0, dst_numPoints=self.numPoints1)

    def test_neighbors_bb(self):
        """LinkedListNNPS :: neighbor test src = b, dst = b """
        self._test_neighbors(src_index=1, dst_index=1, dst_numPoints=self.numPoints2)

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
