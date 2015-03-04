"""Tests for the simple inlet.

Copyright (c) 2015, Prabhu Ramachandran
License: BSD
"""
import numpy as np
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
        inlet = SimpleInlet(self.inlet_pa, self.dest_pa, dx=self.dx, n=5,
                            xmin=-0.4, xmax=0.0)
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

    def test_update_creates_particles_in_destination(self):
        # Given
        inlet = SimpleInlet(self.inlet_pa, self.dest_pa, dx=self.dx, n=5,
                            xmin=-0.4, xmax=0.0)
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
        x_expect[x_expect > 0.0] -= 0.4
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
