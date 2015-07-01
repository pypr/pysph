"""A simple example demonstrating the use of PySPH as a library for
working with particles.

The fundamental summation density operation is used as the
example. Specifically, a uniform distribution of particles is
generated and the expected density via summation is compared with the
numerical result.

This tutorial illustrates the following:

   - Creating particles : ParticleArray
   - Setting up a periodic domain : DomainManager
   - Nearest Neighbor Particle Searching : NNPS

"""

# PySPH imports
from pyzoltan.core.carray import UIntArray
from pysph.base import utils
from pysph.base.nnps import DomainManager, LinkedListNNPS
from pysph.base.kernels import CubicSpline, Gaussian, QuinticSpline, WendlandQuintic
from pysph.tools.uniform_distribution import uniform_distribution_cubic2D, \
    uniform_distribution_hcp2D, get_number_density_hcp

# NumPy
import numpy

# Python timer
from time import time

# particle spacings
dx = 0.01; dxb2 = 0.5 * dx
h0 = 1.3*dx

# Uniform lattice distribution of particles
#x, y, dx, dy, xmin, xmax, ymin, ymax = uniform_distribution_cubic2D(
#    dx, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

# Uniform hexagonal close packing arrangement of particles
x, y, dx, dy, xmin, xmax, ymin, ymax = uniform_distribution_hcp2D(
    dx, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, adjust=True)

# SPH kernel
#k = CubicSpline(dim=2)
#k = Gaussian(dim=2)
#k = QuinticSpline(dim=2)
k = WendlandQuintic(dim=2)

# for the hexagonal particle spacing, dx*dy is only an approximate
# expression for the particle volume. As far as the summation density
# test is concerned, the value will be uniform but not equal to 1. To
# reproduce a density profile of 1, we need to estimate the kernel sum
# or number density of the distribution based on the kernel
wij_sum_estimate = get_number_density_hcp(dx, dy, k, h0)
volume = 1./wij_sum_estimate
print('Volume estimates :: dx^2 = %g, Number density = %g'%(dx*dy, volume))

x = x.ravel(); y = y.ravel()
h = numpy.ones_like(x) * h0
m = numpy.ones_like(x) * volume
wij = numpy.zeros_like(x)

# use the helper function get_particle_array to create a ParticleArray
pa = utils.get_particle_array(x=x,y=y,h=h,m=m,wij=wij)

# the simulation domain used to request periodicity
domain = DomainManager(
    xmin=0., xmax=1., ymin=0., ymax=1.,periodic_in_x=True, periodic_in_y=True)

# NNPS object for nearest neighbor queries
nps = LinkedListNNPS(dim=2, particles=[pa,], radius_scale=k.radius_scale, domain=domain)

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
    rho_sum = 0.0; wij_sum = 0.0
    for j in neighbors:
        xij = xi - x[j]
        yij = yi - y[j]

        rij = numpy.sqrt( xij**2 + yij**2 )
        hij = 0.5 * (h[i] + h[j])

        _wij = k.kernel( [xij, yij, 0.0], rij, hij)

        # contribution from this neibhor
        wij_sum += _wij
        rho_sum += m[j] * _wij

    # total contribution to the density and number density
    pa.rho[i] = rho_sum
    pa.wij[i] = wij_sum

t2 = time()-t1

avg_density = numpy.sum(pa.rho)/pa.num_real_particles
print('2D Summation density: %d particles %g seconds, Max %d neighbors'%(pa.num_real_particles, t2, max_ngb))
print("""Average density = %g, Relative error = %g"""%(avg_density, (1-avg_density)*100), '%')
