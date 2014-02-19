"""Benchmark numbers for the base.nnps module."""
import numpy
from time import time
from numpy import random

from pyzoltan.core.carray import UIntArray, DoubleArray

from pysph.base.point import IntPoint, Point
from pysph.base.utils import get_particle_array
from pysph.base.nnps import BoxSortNNPS, LinkedListNNPS

_numPoints = [1<<15, 1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22]
bs_times = []
ll_times = []

for numPoints in _numPoints:
    dx = numpy.power( 1./numPoints, 1.0/3.0 )
    xa = random.random(numPoints)
    ya = random.random(numPoints)
    za = random.random(numPoints)
    ha = numpy.ones_like(xa) * 2*dx
    gida = numpy.arange(numPoints).astype(numpy.uint32)
    
    # create the particle array
    pa = get_particle_array(x=xa, y=ya, z=za, h=ha, gid=gida)

    # the box-sort NNPS
    nnps_boxs = BoxSortNNPS(
        dim=3, particles=[pa,], radius_scale=2.0)

    # calculate the time to get neighbors
    t1 = time()
    nnps_boxs.update()
    bs_times.append(time() - t1)
    
    # the linked list NNPS
    nnps_llist = LinkedListNNPS(
        dim=3, particles=[pa,], radius_scale=2.0)

    # calculate the time to get neighbors
    t1 = time()
    nnps_llist.update()
    ll_times.append(time() - t1)

scale_factors = numpy.array(bs_times[1:])/numpy.array(bs_times[:-1])
scale_factors = list(scale_factors)
scale_factors.insert(0, "---")

print "Summation density benchmarks for BoxSortNNPS"
for i, numPoints in enumerate(_numPoints):
    print "N_p = %d, time = %3.05g s, %3.03g s/particle, scale factor = %2.04s"%(numPoints,
                                                                                 bs_times[i],
                                                                                 bs_times[i]/numPoints,
                                                                                 scale_factors[i])

scale_factors = numpy.array(ll_times[1:])/numpy.array(ll_times[:-1])
scale_factors = list(scale_factors)
scale_factors.insert(0, "---")

print "\n\nSummation density benchmarks for LinkedListNNPS"
for i, numPoints in enumerate(_numPoints):
    print "N_p = %d, time = %3.05g s, %3.03g s/particle, scale factor = %2.04s"%(numPoints,
                                                                                 ll_times[i],
                                                                                 ll_times[i]/numPoints,
                                                                                 scale_factors[i])

