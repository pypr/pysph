"""Tests for the periodicity algorithms in NNPS"""

import numpy as np

from pysph.base.nnps import DomainLimits, NNPS
from pysph.base.utils import get_particle_array
from pysph.parallel._kernels import CubicSpline, Gaussian
from pysph.base.point import Point

from pyzoltan.core.carray import UIntArray

# create the particle arrays
L = 1.0; n = 100; dx = L/n; hdx = 1.5
_x = np.arange(dx/2, L, dx)
vol = dx*dx

# fluid particles
xx, yy = np.meshgrid(_x, _x)

x = xx.ravel(); y = yy.ravel()
h = np.ones_like(x) * hdx*dx
m = np.ones_like(x) * vol
V = np.zeros_like(x)

fluid = get_particle_array(name='fluid', x=x, y=y, h=h, m=m, V=V)

# channel particles
_y = np.arange(L+dx/2, L+dx/2 + 10*dx, dx)
xx, yy = np.meshgrid(_x, _y)

xtop = xx.ravel(); ytop = yy.ravel()

_y = np.arange(-dx/2, -dx/2-10*dx, -dx)
xx, yy = np.meshgrid(_x, _y)

xbot = xx.ravel(); ybot = yy.ravel()

x = np.concatenate( (xtop, xbot) )
y = np.concatenate( (ytop, ybot) )

h = np.ones_like(x) * hdx*dx
m = np.ones_like(x) * vol
V = np.zeros_like(x)

channel = get_particle_array(name='channel', x=x, y=y, h=h, m=m, V=V)

# particles
particles = [fluid, channel]

# domain
domain = DomainLimits(xmin=0, xmax=L, periodic_in_x=True)

assert( domain.is_periodic )

# kernel ane nnps
kernel = Gaussian(dim=2)

# nnps
nnps = NNPS(dim=2, particles=particles, domain=domain, radius_scale=kernel.radius)

assert (nnps.is_periodic)

# test summation density on the fluid. It should be approximately 1
fx, fy, fh = fluid.get('x', 'y', 'h', only_real_particles=False)

frho = fluid.rho
frho[:] = 0.0

assert( frho.size == fluid.num_real_particles )

nbrs = UIntArray()
for i in range(frho.size):
    xi = Point( fx[i], fy[i] )
    hi = fh[i]

    # compute density from the fluid
    nnps.get_nearest_particles(src_index=0, dst_index=0, d_idx=i, nbrs=nbrs)
    nnbrs = nbrs._length

    sx, sy, sh, sm = fluid.get('x', 'y', 'h', 'm', only_real_particles=False)

    for indexj in range(nnbrs):
        j = nbrs[indexj]

        xj = Point( sx[j], sy[j] )
        hij = 0.5 * (hi + sh[j])

        frho[i] += sm[j] * kernel.py_function(xi, xj, hij)
        fluid.V[i] += kernel.py_function(xi, xj, hij)

    # compute density from the channel
    nnps.get_nearest_particles(src_index=1, dst_index=0, d_idx=i, nbrs=nbrs)
    nnbrs = nbrs._length

    sx, sy, sh, sm = channel.get('x', 'y', 'h', 'm', only_real_particles=False)

    for indexj in range(nnbrs):
        j = nbrs[indexj]

        xj = Point( sx[j], sy[j] )
        hij = 0.5 * (hi + sh[j])

        frho[i] += sm[j] * kernel.py_function(xi, xj, hij)
        fluid.V[i] += kernel.py_function(xi, xj, hij)

# volume computed by SPH
vol = 1./fluid.V
expected_rho = fluid.m/vol

diff = max(abs(expected_rho-frho))
assert (diff < 1e-14)

print "OK"
