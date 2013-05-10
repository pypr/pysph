"""Benchmark numbers for the base.nnps module."""
import numpy
from time import time
from numpy import random

from pyzoltan.core.carray import UIntArray, DoubleArray

from pysph.base.point import IntPoint, Point
from pysph.base.utils import get_particle_array
from pysph.base.nnps import NNPS

times = []
_numPoints = [1<<15, 1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22]
for numPoints in _numPoints:

    dx = numpy.power( 1./numPoints, 1.0/3.0 )
    
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
    pa = get_particle_array(x=xa, y=ya, h=ha, gid=gida)
    nps = NNPS(dim=2, particles=[pa,], radius_scale=2.0)

    # calculate the time to get neighbors
    t1 = time()
    nps.update()
    times.append(time() - t1)

scale_factors = numpy.array(times[1:])/numpy.array(times[:-1])
scale_factors = list(scale_factors)
scale_factors.insert(0, "---")

print "Summation density benchmarks"
for i, numPoints in enumerate(_numPoints):
    print "N_p = %d, time = %3.05g s, %3.03g s/particle, scale factor = %2.04s"%(numPoints,
                                                                                 times[i],
                                                                                 times[i]/numPoints,
                                                                                 scale_factors[i])

