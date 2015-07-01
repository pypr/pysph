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
        n = 11
        x, y = np.mgrid[-1:1:n*1j,-1:1:n*1j]
        dx = 2.0/(n-1)
        z = np.zeros_like(x)
        x, y, z = x.ravel(), y.ravel(), z.ravel()
        m = np.ones_like(x)
        p = np.ones_like(x)*2.0
        h = np.ones_like(x)*2*dx
        u = np.ones_like(x)*0.1
        pa = get_particle_array(name=name, x=x, y=y, z=z, h=h, m=m, p=p, u=u)
        return pa

    def test_should_work_on_2d_data(self):
        # Given
        pa = self._make_2d_grid()

        # When.
        ip = Interpolator([pa], num_points=1000)
        p = ip.interpolate('p')
        u = ip.interpolate('u')

        # Then.
        expect = np.ones_like(p)*2.0
        self.assertTrue(np.allclose(p, expect))
        expect = np.ones_like(u)*0.1
        self.assertTrue(np.allclose(u, expect))

    def test_should_work_with_multiple_arrays(self):
        # Given
        pa1 = self._make_2d_grid()
        pa2 = self._make_2d_grid('solid')
        pa2.p[:] = 4.0
        pa2.u[:] = 0.2

        # When.
        ip = Interpolator([pa1, pa2], num_points=1000)
        p = ip.interpolate('p')
        u = ip.interpolate('u')

        # Then.
        expect = np.ones_like(p)*3.0
        self.assertTrue(np.allclose(p, expect))
        expect = np.ones_like(u)*0.15
        self.assertTrue(np.allclose(u, expect))

    def test_should_work_with_ghost_particles(self):
        # Given
        pa = self._make_2d_grid()
        # Make half the particles ghosts.
        n = pa.get_number_of_particles()
        pa.tag[n/2:] = 1
        pa.align_particles()

        # When.
        ip = Interpolator([pa], num_points=1000)
        p = ip.interpolate('p')
        u = ip.interpolate('u')

        # Then.
        expect = np.ones_like(p)*2.0
        self.assertTrue(np.allclose(p, expect))
        expect = np.ones_like(u)*0.1
        self.assertTrue(np.allclose(u, expect))

    def test_should_work_with_changed_data(self):
        # Given
        pa = self._make_2d_grid()
        ip = Interpolator([pa], num_points=1000)
        p = ip.interpolate('p')

        # When.
        pa.p *= 2.0
        p = ip.interpolate('p')

        # Then.
        expect = np.ones_like(p)*4.0
        self.assertTrue(np.allclose(p, expect))

    def test_should_be_able_to_update_particle_arrays(self):
        # Given
        pa = self._make_2d_grid()
        pa_new = self._make_2d_grid()
        pa_new.p[:] = 10.0

        ip = Interpolator([pa], num_points=1000)
        p = ip.interpolate('p')

        # When.
        ip.update_particle_arrays([pa_new])
        p = ip.interpolate('p')

        # Then.
        expect = np.ones_like(p)*10.0
        self.assertTrue(np.allclose(p, expect))

    def test_should_correctly_update_domain(self):
        # Given
        pa = self._make_2d_grid()
        ip = Interpolator([pa], num_points=1000)
        p = ip.interpolate('p')

        # When.
        ip.set_domain((0.0, 1.0, 0.0, 1.0, 0.0, 0.0), (11, 11, 1))
        p = ip.interpolate('p')

        # Then.
        expect = np.ones_like(p)*2.0
        self.assertTrue(np.allclose(p, expect))

    def test_should_work_when_arrays_have_different_props(self):
        # Given
        pa1 = self._make_2d_grid()
        pa1.add_property('junk', default=2.0)
        pa2 = self._make_2d_grid('solid')

        # When.
        ip = Interpolator([pa1, pa2], num_points=1000)
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
        ip = Interpolator([pa], x=x, y=y, z=z)
        p = ip.interpolate('p')

        # Then.
        self.assertEqual(p.shape, x.shape)
        expect = np.ones_like(x)*2.0
        self.assertTrue(np.allclose(p, expect))

    def test_should_work_with_explicit_points_without_z(self):
        # Given
        pa = self._make_2d_grid()
        x, y = np.random.random((2, 5, 5))

        # When.
        ip = Interpolator([pa], x=x, y=y)
        p = ip.interpolate('p')

        # Then.
        self.assertEqual(p.shape, x.shape)
        expect = np.ones_like(x)*2.0
        self.assertTrue(np.allclose(p, expect))

    def test_that_set_interpolation_points_works(self):
        # Given
        pa = self._make_2d_grid()
        ip = Interpolator([pa], num_points=1000)

        # When.
        x, y = np.random.random((2, 5, 5))
        ip.set_interpolation_points(x=x, y=y)
        p = ip.interpolate('p')

        # Then.
        self.assertEqual(p.shape, x.shape)
        expect = np.ones_like(x)*2.0
        self.assertTrue(np.allclose(p, expect))



if __name__ == '__main__':
    unittest.main()
