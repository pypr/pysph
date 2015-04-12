"""Tests for the periodicity algorithms in the DomainManager.
"""

# NumPy
import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager, BoxSortNNPS, LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian, get_compiled_kernel

# PyZoltan CArrays
from pyzoltan.core.carray import UIntArray

# Python unit testing framework
import unittest

class PeriodicBox2DTestCase(unittest.TestCase):
    """Test the periodicity algorithms in the Domain Manager.

    We create a 2D box with periodicity along x and y.  We check if this
    produces a constant density with summation density.

    """
    def setUp(self):
        # create the particle arrays
        L = 1.0; n = 10; dx = L/n; hdx = 1.5
        self.L = L
        _x = np.arange(dx/2, L, dx)
        self.vol = vol = dx*dx

        # fluid particles
        xx, yy = np.meshgrid(_x, _x)

        x = xx.ravel(); y = yy.ravel() # particle positions
        p = self._get_pressure(x, y, 0.0)
        h = np.ones_like(x) * hdx*dx   # smoothing lengths
        m = np.ones_like(x) * vol      # mass
        V = np.zeros_like(x)           # volumes

        fluid = get_particle_array(name='fluid', x=x, y=y, h=h, m=m, V=V, p=p)

        # particles and domain
        self.fluid = fluid
        self.domain = DomainManager(xmin=0, xmax=L, ymin=0, ymax=L,
            periodic_in_x=True, periodic_in_y=True
        )
        self.kernel = get_compiled_kernel(Gaussian(dim=2))

    def _get_pressure(self, x, y, z=0.0):
        L = self.L
        return np.mod(x, L) + np.mod(y, L) + np.mod(z, L)

    def _check_summation_density(self):
        fluid = self.fluid
        nnps = self.nnps
        kernel = self.kernel
        nnps.update_domain()
        nnps.update()

        # get the fluid arrays
        fx, fy, fz, fh, frho, fV, fm = fluid.get(
            'x', 'y', 'z', 'h', 'rho', 'V', 'm', only_real_particles=True
        )

        # the source arrays. First source is also the fluid
        sx, sy, sz, sh, sm = fluid.get('x', 'y', 'z', 'h', 'm',
            only_real_particles=False
        )

        # initialize the fluid density and volume
        frho[:] = 0.0
        fV[:] = 0.0

        # compute density on the fluid
        nbrs = UIntArray()
        for i in range( fluid.num_real_particles ):
            hi = fh[i]

            # compute density from the fluid from the source arrays
            nnps.get_nearest_particles(src_index=0, dst_index=0, d_idx=i, nbrs=nbrs)
            nnbrs = nbrs.length

            for indexj in range(nnbrs):
                j = nbrs[indexj]
                hij = 0.5 * (hi + sh[j])

                frho[i] += sm[j] * kernel.kernel(
                    fx[i], fy[i], fz[i], sx[j], sy[j], sz[j], hij
                )
                fV[i] += kernel.kernel(
                    fx[i], fy[i], fz[i], sx[j], sy[j], sz[j], hij
                )

            # check the number density and density by summation
            voli = 1./fV[i]
            #print voli, frho[i], fm[i], self.vol
            self.assertAlmostEqual( voli, self.vol, 5 )
            self.assertAlmostEqual( frho[i], fm[i]/voli, 14)


class BoxSortPeriodicBox2D(PeriodicBox2DTestCase):
    def setUp(self):
        PeriodicBox2DTestCase.setUp(self)
        self.orig_n = self.fluid.get_number_of_particles()
        self.nnps = BoxSortNNPS(
            dim=2, particles=[self.fluid],
            domain=self.domain,
            radius_scale=self.kernel.radius_scale)

    def test_summation_density(self):
        self._check_summation_density()

    def test_box_wrapping(self):
        # Given
        fluid = self.fluid
        fluid.x += 0.35
        fluid.y += 0.35
        self._check_summation_density()

    def test_periodicity(self):
        # Given.
        fluid = self.fluid

        # When
        self.domain.update()

        # Then.
        x, y, p = fluid.get('x', 'y', 'p', only_real_particles=False)
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        new_n = fluid.get_number_of_particles()
        self.assertTrue(new_n > self.orig_n)
        self.assertTrue(xmin < 0.0)
        self.assertTrue(xmax > self.L)
        self.assertTrue(ymin < 0.0)
        self.assertTrue(ymax > self.L)

        p_expect = self._get_pressure(x, y, 0)
        diff = np.abs(p - p_expect).max()
        message = "Pressure not equal, max diff: %s"%diff
        self.assertTrue(np.allclose(p, p_expect, atol=1e-14), message)


class LinkedListPeriodicBox2D(BoxSortPeriodicBox2D):
    def setUp(self):
        PeriodicBox2DTestCase.setUp(self)
        self.orig_n = self.fluid.get_number_of_particles()
        self.nnps = LinkedListNNPS(
            dim=2, particles=[self.fluid],
            domain=self.domain,
            radius_scale=self.kernel.radius_scale)


class PeriodicBox3DTestCase(PeriodicBox2DTestCase):
    """Test the periodicity algorithms in the Domain Manager.

    We create a 3D box with periodicity along x, y and z.  We check if this
    produces a constant density with summation density.

    """
    def setUp(self):
        # create the particle arrays
        L = 1.0; n = 5; dx = L/n; hdx = 1.5
        self.L = L
        self.vol = vol = dx*dx*dx

        # fluid particles
        xx, yy, zz = np.mgrid[dx/2:L:dx,dx/2:L:dx,dx/2:L:dx]

        x = xx.ravel(); y = yy.ravel(); z = zz.ravel() # particle positions
        p = self._get_pressure(x, y, z)
        h = np.ones_like(x) * hdx*dx   # smoothing lengths
        m = np.ones_like(x) * vol      # mass
        V = np.zeros_like(x)           # volumes

        fluid = get_particle_array(name='fluid', x=x, y=y, z=z,
            h=h, m=m, V=V, p=p
        )

        # particles and domain
        self.fluid = fluid
        self.domain = DomainManager(
            xmin=0, xmax=L, ymin=0, ymax=L, zmin=0, zmax=L,
            periodic_in_x=True, periodic_in_y=True, periodic_in_z=True
        )
        self.kernel = get_compiled_kernel(Gaussian(dim=3))

        self.orig_n = self.fluid.get_number_of_particles()
        self.nnps = LinkedListNNPS(
            dim=3, particles=[self.fluid],
            domain=self.domain,
            radius_scale=self.kernel.radius_scale)

    def test_summation_density(self):
        self._check_summation_density()

    def test_box_wrapping(self):
        # Given
        fluid = self.fluid
        fluid.x += 0.35
        fluid.y += 0.35
        fluid.z += 0.35
        self._check_summation_density()

    def test_periodicity(self):
        # Given.
        fluid = self.fluid

        # When
        self.domain.update()

        # Then.
        x, y, z, p = fluid.get('x', 'y', 'z', 'p', only_real_particles=False)
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        zmin, zmax = z.min(), z.max()
        new_n = fluid.get_number_of_particles()
        self.assertTrue(new_n > self.orig_n)
        self.assertTrue(xmin < 0.0)
        self.assertTrue(xmax > self.L)
        self.assertTrue(ymin < 0.0)
        self.assertTrue(ymax > self.L)
        self.assertTrue(zmin < 0.0)
        self.assertTrue(zmax > self.L)

        p_expect = self._get_pressure(x, y, z)
        diff = np.abs(p - p_expect).max()
        message = "Pressure not equal, max diff: %s"%diff
        self.assertTrue(np.allclose(p, p_expect, atol=1e-14), message)


if __name__ == '__main__':
    unittest.main()
