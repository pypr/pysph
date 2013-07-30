"""Simple test for the NNPS class"""
import numpy

from numpy import random

from pysph.base.point import IntPoint, Point
from pysph.base.utils import get_particle_array
from pysph.base import nnps

from pyzoltan.core.carray import UIntArray, DoubleArray

numPoints = 1<<10
dx = numpy.sqrt( 1.0/numPoints )

xa = random.random(numPoints)
ya = random.random(numPoints)
za = random.random(numPoints)
ha = numpy.ones_like(xa) * 2*dx
gida = numpy.arange(numPoints).astype(numpy.uint32)

x = DoubleArray(numPoints); x.set_data(xa)
y = DoubleArray(numPoints); y.set_data(ya)
z = DoubleArray(numPoints); z.set_data(za)
h = DoubleArray(numPoints); h.set_data(ha)
gid = UIntArray(numPoints); gid.set_data(gida)

# Create the NNPS object
pa = get_particle_array(x=xa, y=ya, z=za, h=ha, gid=gida)
nps = nnps.NNPS(dim=3, particles=[pa,], radius_scale=2.0)

nbrs1 = UIntArray()
nbrs2 = UIntArray()

for i in range(numPoints):
    nps.get_nearest_particles(0, 0, i, nbrs1)
    nps.brute_force_neighbors(0, 0, i, nbrs2)

    _nbrs1 = nbrs1.get_npy_array()
    _nbrs2 = nbrs2.get_npy_array()

    # make sure the size of the neighbor list is the same
    assert( nbrs1._length == nbrs2.length )

    # sort the neighbor lists
    nnps_nbrs = _nbrs1[:nbrs1._length]; nnps_nbrs.sort()
    brut_nbrs = _nbrs2; brut_nbrs.sort()

    # check each neighbor
    for j in range(nbrs1._length):
        assert( nnps_nbrs[j] == brut_nbrs[j] )

def test_get_centroid():
    """Tests the get_centroid function."""
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
    """Tests the get_centroid function."""
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

test_get_centroid()
test_get_bbox()

print "OK"
