# Author: Prabhu Ramachandran
# Copyright (c) 2014 Prabhu Ramachandran
# License: BSD Style.

# Standard library imports
try:
    # This is for Python-2.6.x
    import unittest2 as unittest
except ImportError:
    import unittest

# Library imports.
import numpy as np

# Local imports
from pysph.tools.interpolator import get_nx_ny_nz, Interpolator
from pysph.base.nnps_base import DomainManager
from pysph.base.utils import get_particle_array


class TestGetNxNyNz(unittest.TestCase):
    def test_should_work_for_1d_data(self):
        # When
        num_points = 100
        bounds = (0.0, 1.5, 0.0, 0.0, 0.0, 0.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [100, 1, 1])

        # When
        num_points = 100
        bounds = (0.0, 0.0, 0.0, 1.123, 0.0, 0.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [1, 100, 1])

        # When
        num_points = 100
        bounds = (0.0, 0.0, 0.0, 0.0, 0.0, 3.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [1, 1, 100])

    def test_should_work_for_2d_data(self):
        # When
        num_points = 100
        bounds = (0.0, 1.0, 0.5, 1.5, 0.0, 0.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [10, 10, 1])

        # When
        num_points = 100
        bounds = (0.0, 0.0, 0, 1, 0.5, 1.5)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [1, 10, 10])

    def test_should_work_for_3d_data(self):
        # When
        num_points = 1000
        bounds = (0.0, 1.0, 0.5, 1.5, -1.0, 0.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [10, 10, 10])

        # When
        num_points = 1000
        bounds = (0.0, 1.0, 0.5, 1.5, -1.0, 2.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [7, 7, 21])


class TestInterpolator(unittest.TestCase):
    def _make_2d_grid(self, name='fluid'):
        n = 101
        dx = 2.0/(n-1)
        x, y = np.mgrid[-1.+dx/2:1.:dx, -1.+dx/2:1.:dx]
        z = np.zeros_like(x)
        x, y, z = x.ravel(), y.ravel(), z.ravel()
        m = np.ones_like(x)
        p = np.sin(x * np.pi)
        h = np.ones_like(x)*2.*dx
        u = np.cos(x * np.pi)
        rho = np.ones_like(x)*m/(dx*dx)
        pa = get_particle_array(name=name, x=x, y=y, z=z, h=h, m=m, p=p, u=u,
                                rho=rho)
        return pa

    @property
    def _domain(self):
        x0 = 1.0
        domain = DomainManager(
            xmin=-x0, xmax=x0, ymin=-x0, ymax=x0,
            periodic_in_x=True, periodic_in_y=True
        )
        return domain

    def test_should_work_on_2d_data(self):
        # Given
        pa = self._make_2d_grid()

        # When.
        ip = Interpolator([pa], num_points=1000, domain_manager=self._domain)
        p = ip.interpolate('p')
        u = ip.interpolate('u')

        # Then.
        expect = np.sin(ip.x * np.pi)
        np.testing.assert_allclose(p, expect, rtol=5e-3)
        expect = np.cos(ip.x * np.pi)
        np.testing.assert_allclose(u, expect, rtol=5e-3)

    def test_should_work_with_multiple_arrays(self):
        # Given
        pa1 = self._make_2d_grid()
        pa2 = self._make_2d_grid('solid')
        pa2.p[:] = 4.0
        pa2.u[:] = 0.2

        # When.
        ip = Interpolator([pa1, pa2], num_points=1000,
                          domain_manager=self._domain)
        p = ip.interpolate('p')
        u = ip.interpolate('u')

        # Then.
        expect = (np.sin(ip.x * np.pi) + 4.0) * 0.5
        np.testing.assert_allclose(p, expect, rtol=5e-2)
        expect = (np.cos(ip.x * np.pi) + 0.2) * 0.5
        np.testing.assert_allclose(u, expect, rtol=5e-2)

    def test_should_work_with_ghost_particles(self):
        # Given
        pa = self._make_2d_grid()
        # Make half the particles ghosts.
        n = pa.get_number_of_particles()
        pa.tag[int(n//2):] = 1
        pa.align_particles()

        # When.
        ip = Interpolator([pa], num_points=1000, domain_manager=self._domain)
        p = ip.interpolate('p')
        u = ip.interpolate('u')

        # Then.
        expect = np.sin(ip.x * np.pi)
        np.testing.assert_allclose(p, expect, rtol=5e-3)
        expect = np.cos(ip.x * np.pi)
        np.testing.assert_allclose(u, expect, rtol=5e-3)

    def test_should_work_with_changed_data(self):
        # Given
        pa = self._make_2d_grid()
        ip = Interpolator([pa], num_points=1000, domain_manager=self._domain)
        p = ip.interpolate('p')

        # When.
        pa.p *= 2.0
        ip.update()
        p = ip.interpolate('p')

        # Then.
        expect = np.sin(ip.x * np.pi) * 2.0
        np.testing.assert_allclose(p, expect, rtol=5e-2)

    def test_should_be_able_to_update_particle_arrays(self):
        # Given
        pa = self._make_2d_grid()
        pa_new = self._make_2d_grid()
        pa_new.p[:] = np.cos(pa_new.x * np.pi)

        ip = Interpolator([pa], num_points=1000, domain_manager=self._domain)
        p = ip.interpolate('p')

        # When.
        ip.update_particle_arrays([pa_new])
        p = ip.interpolate('p')

        # Then.
        expect = np.cos(ip.x * np.pi)
        np.testing.assert_allclose(p, expect, rtol=5e-3)

    def test_should_correctly_update_domain(self):
        # Given
        pa = self._make_2d_grid()
        ip = Interpolator([pa], num_points=1000, domain_manager=self._domain)
        p = ip.interpolate('p')

        # When.
        ip.set_domain((0.1, 1.0, 0.1, 1.0, 0.0, 0.0), (11, 11, 1))
        p = ip.interpolate('p')

        # Then.
        expect = np.sin(ip.x * np.pi)
        print(p - expect)
        np.testing.assert_allclose(p, expect, atol=5e-3)

    def test_should_work_when_arrays_have_different_props(self):
        # Given
        pa1 = self._make_2d_grid()
        pa1.add_property('junk', default=2.0, type='int')
        pa2 = self._make_2d_grid('solid')

        # When.
        ip = Interpolator([pa1, pa2], num_points=1000,
                          domain_manager=self._domain)
        junk = ip.interpolate('junk')

        # Then.
        expect = np.ones_like(junk)*1.0
        self.assertTrue(np.allclose(junk, expect))

    def test_should_work_with_explicit_points_in_constructor(self):
        # Given
        pa = self._make_2d_grid()
        x, y = np.random.random((2, 5, 5))
        z = np.zeros_like(x)

        # When.
        ip = Interpolator([pa], x=x, y=y, z=z, domain_manager=self._domain)
        p = ip.interpolate('p')

        # Then.
        self.assertEqual(p.shape, x.shape)
        expect = np.sin(ip.x * np.pi)
        np.testing.assert_allclose(p, expect, rtol=5e-3)

    def test_should_work_with_explicit_points_without_z(self):
        # Given
        pa = self._make_2d_grid()
        x, y = np.random.random((2, 5, 5))

        # When.
        ip = Interpolator([pa], x=x, y=y, domain_manager=self._domain)
        p = ip.interpolate('p')

        # Then.
        self.assertEqual(p.shape, x.shape)
        expect = np.sin(ip.x * np.pi)
        np.testing.assert_allclose(p, expect, rtol=5e-3)

    def test_that_set_interpolation_points_works(self):
        # Given
        pa = self._make_2d_grid()
        ip = Interpolator([pa], num_points=1000, domain_manager=self._domain)

        # When.
        x, y = np.random.random((2, 5, 5))
        ip.set_interpolation_points(x=x, y=y)
        p = ip.interpolate('p')

        # Then.
        self.assertEqual(p.shape, x.shape)
        expect = np.sin(ip.x * np.pi)
        np.testing.assert_allclose(p, expect, rtol=5e-3)

    def test_should_work_with_method_order1(self):
        # Given
        pa = self._make_2d_grid()

        # When.
        ip = Interpolator(
            [pa], domain_manager=self._domain, method='order1'
        )
        x, y = np.random.random((2, 5, 5))
        ip.set_interpolation_points(x=x, y=y)
        p = ip.interpolate('p')

        # Then.
        expect = np.sin(ip.x * np.pi)
        np.testing.assert_allclose(p, expect, rtol=5e-3)

    def test_should_work_with_method_sph(self):
        # Given
        pa = self._make_2d_grid()

        # When.
        ip = Interpolator(
            [pa], domain_manager=self._domain, method='sph'
        )
        x, y = np.random.random((2, 5, 5))
        ip.set_interpolation_points(x=x, y=y)
        p = ip.interpolate('p')

        # Then.
        expect = np.sin(ip.x * np.pi)
        np.testing.assert_allclose(p, expect, rtol=5e-3)

    def test_gradient_calculation_2d(self):
        # Given
        pa = self._make_2d_grid()

        # When.
        ip = Interpolator(
            [pa], domain_manager=self._domain, method='order1'
        )
        x, y = np.random.random((2, 5, 5))
        ip.set_interpolation_points(x=x, y=y)
        p = ip.interpolate('p', 1)

        # Then.
        expect = np.pi * np.cos(ip.x * np.pi)
        np.testing.assert_allclose(p, expect, rtol=1e-2)

    def test_gradient_with_sph(self):
        # Given
        pa = self._make_2d_grid()

        # When.
        ip = Interpolator(
            [pa], domain_manager=self._domain, method='sph'
        )
        x, y = np.random.random((2, 5, 5))
        ip.set_interpolation_points(x=x, y=y)

        # Then.
        self.assertRaises(RuntimeError, ip.interpolate, 'p', 1)

    def test_gradient_with_shepard(self):
        # Given
        pa = self._make_2d_grid()

        # When.
        ip = Interpolator(
            [pa], domain_manager=self._domain, method='shepard'
        )
        x, y = np.random.random((2, 5, 5))
        ip.set_interpolation_points(x=x, y=y)

        # Then.
        self.assertRaises(RuntimeError, ip.interpolate, 'p', 1)


if __name__ == '__main__':
    unittest.main()
