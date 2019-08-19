"""Tests for the simple_inlet_outlet.

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
from pysph.base.kernels import QuinticSpline
from pysph.sph.bc.inlet_outlet_manager import (
    InletInfo, OutletInfo, InletBase, OutletBase)


class TestSimpleInlet1D(unittest.TestCase):
    def setUp(self):
        dx = 0.1
        x = np.arange(-5*dx, 0.0, dx)
        m = np.ones_like(x)
        h = np.ones_like(x)*dx*1.5
        p = np.ones_like(x)*5.0
        self.inlet_pa = get_particle_array(name='inlet', x=x, m=m, h=h, p=p)
        # Empty particle array.
        self.dest_pa = get_particle_array(name='fluid')
        props = ['ioid', 'disp']
        for p in props:
            for pa_arr in [self.dest_pa, self.inlet_pa]:
                pa_arr.add_property(p)
        self.dx = dx
        self.kernel = QuinticSpline(dim=1)

        self.inletinfo = InletInfo('inlet', normal=[-1., 0., 0.],
                                   refpoint=[-dx/2, 0., 0.])
        self.inletinfo.length = 0.5

    def test_update_creates_particles_in_destination(self):
        # Given
        inlet = InletBase(
            self.inlet_pa, self.dest_pa, self.inletinfo,
            dim=1, kernel=self.kernel)
        # Two rows of particles should move out.
        self.inlet_pa.x += 0.12

        # When
        inlet.update(time=0.0, dt=0.0, stage=1)

        # Then
        x = self.inlet_pa.x
        p = self.inlet_pa.p
        h = self.inlet_pa.h
        self.assertEqual(len(x), 5)
        x_expect = (-np.arange(5, 0, -1)*self.dx + 0.12)
        x_expect[x_expect > 0.0] -= 0.5
        self.assertTrue(np.allclose(list(x), list(x_expect)))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(
            np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14)
        )

        # The destination particle array should now have one particles.
        x = self.dest_pa.x
        p = self.dest_pa.p
        h = self.dest_pa.h
        self.assertEqual(self.dest_pa.get_number_of_particles(), 1)
        x_expect = (-np.arange(1, 0, -1)*self.dx + 0.12)
        self.assertTrue(np.allclose(list(x), list(x_expect)))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(
            np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14)
        )

    def test_particles_should_update_in_given_stage(self):
        # Given
        inlet = InletBase(
            self.inlet_pa, self.dest_pa, self.inletinfo,
            dim=1, kernel=self.kernel)
        # Two rows of particles should move out.
        self.inlet_pa.x += 0.15
        inlet.active_stages = [1]

        # When
        inlet.update(time=0.0, dt=0.0, stage=2)

        # Then
        x = self.inlet_pa.x
        p = self.inlet_pa.p
        h = self.inlet_pa.h
        self.assertEqual(len(x), 5)
        x_expect = (-np.arange(5, 0, -1)*self.dx + 0.15)
        self.assertTrue(np.allclose(list(x), list(x_expect)))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(
            np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14)
        )

        # The destination particle array should not have particles.
        self.assertEqual(self.dest_pa.get_number_of_particles(), 0)

    def test_inlet_calls_callback(self):
        # Given
        calls = []

        def _callback(d, i):
            calls.append((d, i))

        inlet = InletBase(
            self.inlet_pa, self.dest_pa, self.inletinfo,
            dim=1.0, kernel=self.kernel, callback=_callback)

        # When
        self.inlet_pa.x += 0.5
        inlet.update(time=0.0, dt=0.0, stage=1)

        # Then
        self.assertEqual(len(calls), 1)
        d_pa, i_pa = calls[0]
        self.assertEqual(d_pa, self.dest_pa)
        self.assertEqual(i_pa, self.inlet_pa)


class TestSimpleOutlet1D(unittest.TestCase):
    def setUp(self):
        dx = 0.1
        x = np.arange(-5*dx, 0.0, dx)
        m = np.ones_like(x)
        h = np.ones_like(x)*dx*1.5
        p = np.ones_like(x)*5.0
        self.source_pa = get_particle_array(name='fluid', x=x, m=m, h=h, p=p)
        # Empty particle array.
        self.outlet_pa = get_particle_array(name='outlet')
        props = ['ioid', 'disp']
        for p in props:
            for pa_arr in [self.source_pa, self.outlet_pa]:
                pa_arr.add_property(p)
        self.dx = dx
        self.kernel = QuinticSpline(dim=1)

        self.outletinfo = OutletInfo(
            'outlet', normal=[1., 0., 0.], refpoint=[-dx/2, 0., 0.],
            props_to_copy=self.source_pa.get_lb_props())
        self.outletinfo.length = 0.5

    def test_outlet_absorbs_particles_from_source(self):
        # Given
        outlet = OutletBase(
            self.outlet_pa, self.source_pa, self.outletinfo,
            dim=1, kernel=self.kernel)
        # Two rows of particles should move out.
        self.source_pa.x += 0.12

        # When
        outlet.update(time=0.0, dt=0.0, stage=1)

        # Then
        x = self.source_pa.x
        p = self.source_pa.p
        h = self.source_pa.h
        print(x)
        self.assertEqual(len(x), 4)
        x_expect = (-np.arange(5, 1, -1)*self.dx + 0.12)
        x_expect[x_expect > 0.0] -= 0.5
        self.assertTrue(np.allclose(list(x), list(x_expect)))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(
            np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14)
        )

        # The outlet particle array should now have one particles.
        x = self.outlet_pa.x
        p = self.outlet_pa.p
        h = self.outlet_pa.h
        self.assertEqual(self.outlet_pa.get_number_of_particles(), 1)
        x_expect = (-np.arange(1, 0, -1)*self.dx + 0.12)
        self.assertTrue(np.allclose(list(x), list(x_expect)))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(
            np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14)
        )

    def test_particles_should_update_in_given_stage(self):
        # Given
        outlet = OutletBase(
            self.outlet_pa, self.source_pa, self.outletinfo,
            dim=1, kernel=self.kernel)
        # Two rows of particles should move out.
        self.source_pa.x += 0.15
        outlet.active_stages = [1]

        # When
        outlet.update(time=0.0, dt=0.0, stage=2)

        # Then
        x = self.source_pa.x
        p = self.source_pa.p
        h = self.source_pa.h
        self.assertEqual(len(x), 5)
        x_expect = (-np.arange(5, 0, -1)*self.dx + 0.15)
        self.assertTrue(np.allclose(list(x), list(x_expect)))
        self.assertTrue(np.allclose(p, np.ones_like(x)*5, atol=1e-14))
        self.assertTrue(
            np.allclose(h, np.ones_like(x)*self.dx*1.5, atol=1e-14)
        )

        # The outlet particle array should not have particles.
        self.assertEqual(self.outlet_pa.get_number_of_particles(), 0)

    def test_outlet_deletes_particles(self):
        # Given
        outlet = OutletBase(
            self.outlet_pa, self.source_pa, self.outletinfo,
            dim=1, kernel=self.kernel)
        # Two rows of particles should move out.
        self.source_pa.x += 0.5

        # When
        outlet.update(time=0.0, dt=0.0, stage=1)

        # Then
        self.assertEqual(self.source_pa.get_number_of_particles(), 0)
        self.assertEqual(self.outlet_pa.get_number_of_particles(), 5)

        # When
        self.outlet_pa.x += 0.12
        outlet.update(time=0.0, dt=0.0, stage=1)

        # The outlet particle array should delete one particle.
        self.assertEqual(self.outlet_pa.get_number_of_particles(), 4)

    def test_outlet_calls_callback(self):
        # Given
        calls = []

        def _callback(s, o):
            calls.append((s, o))

        outlet = OutletBase(
            self.outlet_pa, self.source_pa, self.outletinfo,
            dim=1.0, kernel=self.kernel, callback=_callback)

        # When
        self.source_pa.x += 0.5
        outlet.update(time=0.0, dt=0.0, stage=1)

        # Then
        self.assertEqual(len(calls), 1)
        s_pa, o_pa = calls[0]
        self.assertEqual(o_pa, self.outlet_pa)
        self.assertEqual(s_pa, self.source_pa)


if __name__ == '__main__':
    unittest.main()
