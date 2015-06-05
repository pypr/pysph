"""Tests for the simple inlet.

Copyright (c) 2015, Prabhu Ramachandran
License: BSD
"""
import numpy as np
try:
    # This is for Python-2.6.x
    import unittest2 as unittest
except ImportError:
    import unittest

from pysph.base.utils import get_particle_array
from pysph.sph.simple_inlet_outlet import SimpleInlet, SimpleOutlet


class TestSimpleInlet1D(unittest.TestCase):
    def setUp(self):
        dx = 0.1
        x = np.array([0.0])
        m = np.ones_like(x)
        h = np.ones_like(x)*dx*1.5
        p = np.ones_like(x)*5.0
        self.inlet_pa = get_particle_array(x=x, m=m, h=h, p=p)
        # Empty particle array.
        self.dest_pa = get_particle_array()
        self.dx = dx

    def test_inlet_block_has_correct_copies(self):
        # Given
        inlet = SimpleInlet(self.inlet_pa, self.dest_pa, spacing=self.dx, n=5,
                            xmin=-0.5, xmax=0.0)
        # When
        x = self.inlet_pa.x
        p = self.inlet_pa.p
        h = self.inlet_pa.h

        # Then
        self.assertEqual(len(x), 5)
        x_expect = -np.arange(0, 5)*self.dx
        self.assertListEqual(list(x), list(x_expect))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14))

        self.assertEqual(self.dest_pa.get_number_of_particles(), 0)

    def test_inlet_raises_error_on_incorrect_domain(self):
        # Given

        # When
        n = 5
        xmin = -0.4

        # Then
        self.assertRaises(RuntimeError, SimpleInlet,
                          self.inlet_pa, self.dest_pa,
                          spacing=self.dx, n=n, xmin=xmin, xmax=0.0)

    def test_update_creates_particles_in_destination(self):
        # Given
        inlet = SimpleInlet(self.inlet_pa, self.dest_pa, spacing=self.dx, n=5,
                            xmin=-0.5, xmax=0.0)
        # Two rows of particles should move out.
        self.inlet_pa.x += 0.15

        # When
        inlet.update()

        # Then
        x = self.inlet_pa.x
        p = self.inlet_pa.p
        h = self.inlet_pa.h
        self.assertEqual(len(x), 5)
        x_expect = (-np.arange(0, 5)*self.dx + 0.15)
        x_expect[x_expect > 0.0] -= 0.5
        self.assertListEqual(list(x), list(x_expect))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14))

        # The destination particle array should now have two particles.
        x = self.dest_pa.x
        p = self.dest_pa.p
        h = self.dest_pa.h
        self.assertEqual(self.dest_pa.get_number_of_particles(), 2)
        x_expect = (-np.arange(0, 2)*self.dx + 0.15)
        self.assertListEqual(list(x), list(x_expect))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14))


class TestSimpleInletGenericMotion2D(unittest.TestCase):
    def setUp(self):
        dx = 0.1
        y = np.arange(5, dtype=float)*0.1
        m = np.ones_like(y)
        h = np.ones_like(y)*dx*1.5
        p = np.ones_like(y)*5.0
        self.inlet_pa = get_particle_array(y=y, m=m, h=h, p=p)
        # Empty particle array.
        self.dest_pa = get_particle_array()
        self.dx = dx

    def test_update_correctly_handles_general_motion(self):
        # Given
        inlet_pa = self.inlet_pa
        dest_pa = self.dest_pa
        inlet = SimpleInlet(inlet_pa, dest_pa, spacing=self.dx, n=5,
                            xmin=-0.5, xmax=0.0, ymin=0.0, ymax=0.5)
        x0 = inlet_pa.x.copy()
        y0 = inlet_pa.y.copy()
        # One row and one column of particles should move out.
        dx = 0.15
        dy = 0.1
        inlet_pa.x -= dx
        inlet_pa.y -= dy

        # When
        inlet.update()

        # Then
        x = inlet_pa.x
        y = inlet_pa.y
        p = inlet_pa.p
        h = inlet_pa.h

        # The number of particles should be the same in the inlet.
        self.assertEqual(len(x), 25)
        x_expect = (x0 - dx)
        x_expect[x_expect < -0.5] += 0.5
        self.assertListEqual(list(x), list(x_expect))

        y_expect = (y0 - dy)
        y_expect[y_expect < 0.0] += 0.5
        self.assertListEqual(list(y), list(y_expect))

        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14))

        # The destination particle array should now have 9 particles.
        x = dest_pa.x
        y = dest_pa.y
        p = dest_pa.p
        h = dest_pa.h
        print(x, y)
        self.assertEqual(self.dest_pa.get_number_of_particles(), 9)

        # Calculate the expected positions.
        x_new = (x0 - dx)
        y_new = (y0 - dy)
        idx = np.where((x_new < -0.5) | (y_new < 0.0))[0]
        x_expect = x_new[idx]
        y_expect = y_new[idx]

        self.assertListEqual(list(x), list(x_expect))
        self.assertListEqual(list(y), list(y_expect))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14))


class TestSimpleOutlet1D(unittest.TestCase):
    def setUp(self):
        dx = 0.1
        x = np.arange(0, 10, dtype=float)*dx
        m = np.ones_like(x)
        h = np.ones_like(x)*dx*1.5
        p = np.ones_like(x)*5.0
        self.source_pa = get_particle_array(x=x, m=m, h=h, p=p)
        # Empty particle array.
        self.outlet_pa = get_particle_array()
        self.dx = dx

    def test_outlet_absorbs_particles_from_source(self):
        # Given
        outlet = SimpleOutlet(self.outlet_pa, self.source_pa, xmin=1.0,
                              xmax=1.5)

        # When
        self.source_pa.x += 0.45
        outlet.update()

        # Then
        x = self.source_pa.x
        p = self.source_pa.p
        h = self.source_pa.h
        self.assertEqual(len(x), 6)
        x_expect = np.arange(0, 6, dtype=float)*self.dx + 0.45
        self.assertListEqual(list(x), list(x_expect))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14))

        x = self.outlet_pa.x
        p = self.outlet_pa.p
        h = self.outlet_pa.h
        self.assertEqual(len(x), 4)

    def test_outlet_removes_exiting_particles_3D(self):
        # Given
        # Note that we treat the self.source_pa as the outlet in this case.
        outlet_pa = self.source_pa
        source_pa = self.outlet_pa
        outlet = SimpleOutlet(
            outlet_pa=outlet_pa, source_pa=source_pa, xmin=0,
            xmax=1.0, ymin=-0.5, ymax=0.5, zmin=-0.5, zmax=0.5
        )

        # When
        outlet_pa.x += 0.61
        outlet_pa.y[0] += 0.51
        outlet_pa.y[1] -= 0.51
        outlet_pa.z[2] += 0.51
        outlet_pa.z[3] -= 0.51
        outlet.update()

        # Then
        self.assertEqual(len(outlet_pa.x), 0)
        self.assertEqual(len(source_pa.x), 0)

if __name__ == '__main__':
    unittest.main()
