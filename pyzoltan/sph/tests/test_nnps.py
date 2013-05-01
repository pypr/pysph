"""Simple test for the NNPS class"""
import numpy
from numpy import random

from pyzoltan.core.carray import UIntArray, DoubleArray
from pyzoltan.core.point import IntPoint, Point
from pyzoltan.sph.nnps import NNPSParticleGeometric, Cell
from pyzoltan.sph.nnps_utils import get_particle_array

import mpi4py.MPI as mpi
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

numPoints = 1<<10
dx = numpy.sqrt( 1.0/numPoints )

xa = random.random(numPoints)
ya = random.random(numPoints)
ha = numpy.ones_like(xa) * 2*dx
gida = numpy.arange(numPoints).astype(numpy.uint32)

x = DoubleArray(numPoints); x.set_data(xa)
y = DoubleArray(numPoints); y.set_data(ya)
h = DoubleArray(numPoints); h.set_data(ha)
gid = UIntArray(numPoints); gid.set_data(gida)

# Create the NNPS object
pa = get_particle_array(x=xa, y=ya, h=ha, gid=gida)
nps = NNPSParticleGeometric(dim=2, comm=comm, pa=pa)

nbrs1 = UIntArray(100)
nbrs2 = UIntArray(100)

for i in range(numPoints):
    nps.get_nearest_particles(i, nbrs1)
    nps.brute_force_neighbors(i, nbrs2)

    _nbrs1 = nbrs1.get_npy_array()
    _nbrs2 = nbrs2.get_npy_array()

    assert( nbrs1._length == nbrs2.length )
    for j in range(nbrs1._length):
        assert( _nbrs1[j] in _nbrs2 )
        #assert( abs( _nbrs1[j] - _nbrs2[j] ) < 1e-10 )

def test_get_centroid():
    """Tests the get_centroid function."""
    cell = Cell(IntPoint(0, 0, 0), cell_size=0.1, narrays=1)
    centroid = Point()
    cell.get_centroid(centroid)

    assert(abs(centroid.x - 0.05) < 1e-10)
    assert(abs(centroid.y - 0.05) < 1e-10)
    assert(abs(centroid.z - 0.00) < 1e-10)
    
    cell = Cell(IntPoint(1, 2, 3), cell_size=0.5)
    cell.get_centroid(centroid)

    assert(abs(centroid.x - 0.75) < 1e-10)
    assert(abs(centroid.y - 1.25) < 1e-10)
    assert(abs(centroid.z - 0.00) < 1e-10)

def test_get_bbox():
    """Tests the get_centroid function."""
    cell_size = 0.1
    cell = Cell(IntPoint(0, 0, 0), cell_size=cell_size)
    centroid = Point()
    boxmin = Point()
    boxmax = Point()

    cell.get_centroid(centroid)
    cell.get_bounding_box(boxmin, boxmax)

    layers = nps.ghost_layers

    assert(abs(boxmin.x - (centroid.x - (layers+0.5)*cell_size)) < 1e-10)
    assert(abs(boxmin.y - (centroid.y - (layers+0.5)*cell_size)) < 1e-10)

    assert(abs(boxmax.x - (centroid.x + (layers+0.5)*cell_size)) < 1e-10)
    assert(abs(boxmax.y - (centroid.y + (layers+0.5)*cell_size)) < 1e-10)

    cell_size = 0.5
    cell = Cell(IntPoint(1, 2, 0), cell_size=cell_size)

    cell.get_centroid(centroid)
    cell.get_bounding_box(boxmin, boxmax)

    assert(abs(boxmin.x - (centroid.x - (layers+0.5)*cell_size)) < 1e-10)
    assert(abs(boxmin.y - (centroid.y - (layers+0.5)*cell_size)) < 1e-10)

    assert(abs(boxmax.x - (centroid.x + (layers+0.5)*cell_size)) < 1e-10)
    assert(abs(boxmax.y - (centroid.y + (layers+0.5)*cell_size)) < 1e-10)

test_get_centroid()
test_get_bbox()

print "OK"
