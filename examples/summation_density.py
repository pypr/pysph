"""A simple example demonstrating the use of PySPH as a library for
working with particles.

The fundamental summation density operation is used as the
example. Specifically, a uniform distribution of particles is
generated and the expected density via summation is compared with the
numerical result.

This tutorial illustrates the following:

   - Creating particles : ParticleArray
   - Setting up a periodic domain : DomainLimits
   - Nearest Neighbor Particle Searching : NNPS

"""

# PySPH imports
from pyzoltan.core.carray import UIntArray
from pysph.base import utils
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import DomainLimits, LinkedListNNPS

# NumPy
import numpy

# Python timer
from time import time

# Create a uniform particle distribution in the plane [0,1] X [0,1]
dx = 0.01; dxb2 = 0.5 * dx
x, y = numpy.mgrid[dxb2:1:dx, dxb2:1:dx]

x = x.ravel(); y = y.ravel()
h = numpy.ones_like(x) * 1.3*dx
m = numpy.ones_like(x) * dx*dx

# use the helper function get_particle_array to create a ParticleArray
pa = utils.get_particle_array(x=x,y=y,h=h,m=m)

# the simulation domain used to request periodicity
domain = DomainLimits(
    xmin=0., xmax=1., ymin=0., ymax=1.,periodic_in_x=True, periodic_in_y=True)

# NNPS object for nearest neighbor queries
nps = LinkedListNNPS(dim=2, particles=[pa,], radius_scale=2.0, domain=domain)

# SPH kernel
k = CubicSpline(dim=2)

# container for neighbors
nbrs = UIntArray()

# arrays including ghosts
x, y, h, m  = pa.get('x', 'y', 'h', 'm', only_real_particles=False)

# iterate over destination particles
t1 = time()
max_ngb = -1
for i in range( pa.num_real_particles ):
    xi = x[i]; yi = y[i]; hi = h[i]
    
    # get list of neighbors
    nps.get_nearest_particles(0, 0, i, nbrs)
    neighbors = nbrs.get_npy_array()

    max_ngb = max( neighbors.size, max_ngb )

    # iterate over the neighbor set
    rho = 0.0
    for j in neighbors:
        xij = xi - x[j]
        yij = yi - y[j]

        rij = numpy.sqrt( xij**2 + yij**2 )
        hij = 0.5 * (h[i] + h[j])

        wij = k.kernel( [xij, yij, 0.0], rij, hij)

        # contribution from this neibhor
        rho += m[j] * wij

    # total contribution to the density
    pa.rho[i] = rho
t2 = time()-t1

avg_density = numpy.sum(pa.rho)/pa.num_real_particles
print '2D Summation density: %d particles %g seconds, Max %d neighbors'%(pa.num_real_particles, t2, max_ngb)
print """Average density = %g, Relative error = %g"""%(avg_density, (1-avg_density)*100), '%'
