# Standard library imports.
import unittest

# Library imports.
import numpy as np

# Local imports.
from pysph.base.utils import get_particle_array, get_particle_array_wcsph
from pysph.sph.equation import Equation
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.integrator import (LeapFrogIntegrator, PECIntegrator,
                                  PEFRLIntegrator)
from pysph.sph.integrator_step import (LeapFrogStep, PEFRLStep,
                                       TwoStageRigidBodyStep)

class SHM(Equation):
    """Simple harmonic oscillator equation.
    """
    def initialize(self, d_idx, d_x, d_au):
        d_au[d_idx] = -d_x[d_idx]


class TestIntegrator(unittest.TestCase):
    def test_detection_of_missing_arrays_for_integrator(self):
        # Given.
        x = np.asarray([1.0])
        u = np.asarray([0.0])
        h = np.ones_like(x)
        pa = get_particle_array(name='fluid', x=x, u=u, h=h, m=h)
        arrays = [pa]

        # When
        integrator = LeapFrogIntegrator(fluid=LeapFrogStep())
        equations = [SHM(dest="fluid", sources=None)]
        kernel = CubicSpline(dim=1)
        a_eval = AccelerationEval(
            particle_arrays=arrays, equations=equations, kernel=kernel
        )
        comp = SPHCompiler(a_eval, integrator=integrator)

        # Then
        self.assertRaises(RuntimeError, comp.compile)

    def test_detect_missing_arrays_for_many_particle_arrays(self):
        # Given.
        x = np.asarray([1.0])
        u = np.asarray([0.0])
        h = np.ones_like(x)
        fluid = get_particle_array_wcsph(name='fluid', x=x, u=u, h=h, m=h)
        solid = get_particle_array(name='solid', x=x, u=u, h=h, m=h)
        arrays = [fluid, solid]

        # When
        integrator = PECIntegrator(
            fluid=TwoStageRigidBodyStep(), solid=TwoStageRigidBodyStep()
        )
        equations = [SHM(dest="fluid", sources=None)]
        kernel = CubicSpline(dim=1)
        a_eval = AccelerationEval(
            particle_arrays=arrays, equations=equations, kernel=kernel
        )
        comp = SPHCompiler(a_eval, integrator=integrator)

        # Then
        self.assertRaises(RuntimeError, comp.compile)


class TestIntegratorBase(unittest.TestCase):
    def setUp(self):
        x = np.asarray([1.0])
        u = np.asarray([0.0])
        h = np.ones_like(x)
        pa = get_particle_array(name='fluid', x=x, u=u, h=h, m=h)
        for prop in ('ax', 'ay', 'az', 'ae', 'arho', 'e'):
            pa.add_property(prop)
        self.pa = pa

    def _setup_integrator(self, equations, integrator):
        kernel = CubicSpline(dim=1)
        arrays = [self.pa]
        a_eval = AccelerationEval(
            particle_arrays=arrays, equations=equations, kernel=kernel
        )
        comp = SPHCompiler(a_eval, integrator=integrator)
        comp.compile()
        nnps = LinkedListNNPS(dim=kernel.dim, particles=arrays)
        a_eval.set_nnps(nnps)
        integrator.set_nnps(nnps)

    def _integrate(self, integrator, dt, tf, post_step_callback):
        """The post_step_callback is called after each step and is passed the
        current time.
        """
        t = 0.0
        while t < tf:
            integrator.step(t, dt)
            post_step_callback(t+dt)
            t += dt


class TestLeapFrogIntegrator(TestIntegratorBase):
    def test_leapfrog(self):
        # Given.
        integrator = LeapFrogIntegrator(fluid=LeapFrogStep())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)
        tf = np.pi
        dt = 0.02*tf

        # When
        energy = []
        def callback(t):
            x, u = self.pa.x[0], self.pa.u[0]
            energy.append(0.5*(x*x + u*u))

        callback(0.0)
        self._integrate(integrator, dt, tf, callback)

        # Then
        energy = np.asarray(energy)
        self.assertAlmostEqual(np.max(np.abs(energy - 0.5)), 0.0, places=3)

    def test_leapfrog_is_second_order(self):
        # Given.
        integrator = LeapFrogIntegrator(fluid=LeapFrogStep())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)

        # Take a dt, find the error, halve dt, and see that error is drops as
        # desired.

        # When
        tf = np.pi
        dt = 0.02*tf
        energy = []
        def callback(t):
            x, u = self.pa.x[0], self.pa.u[0]
            energy.append(0.5*(x*x + u*u))

        callback(0.0)
        self._integrate(integrator, dt, tf, callback)
        energy = np.asarray(energy)
        err1 = np.max(np.abs(energy - 0.5))

        # When
        self.pa.x[0] = 1.0
        self.pa.u[0] = 0.0
        energy = []
        dt *= 0.5
        callback(0.0)
        self._integrate(integrator, dt, tf, callback)
        energy = np.asarray(energy)
        err2 = np.max(np.abs(energy - 0.5))

        # Then
        self.assertTrue(err2 < err1)
        self.assertAlmostEqual(err1/err2, 4.0, places=2)


class TestPEFRLIntegrator(TestIntegratorBase):
    def test_pefrl(self):
        # Given.
        integrator = PEFRLIntegrator(fluid=PEFRLStep())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)
        tf = np.pi
        dt = 0.1*tf

        # When
        energy = []
        def callback(t):
            x, u = self.pa.x[0], self.pa.u[0]
            energy.append(0.5*(x*x + u*u))

        callback(0.0)
        self._integrate(integrator, dt, tf, callback)

        # Then
        energy = np.asarray(energy)
        self.assertAlmostEqual(np.max(np.abs(energy - 0.5)), 0.0, places=4)

    def test_pefrl_is_fourth_order(self):
        # Given.
        integrator = PEFRLIntegrator(fluid=PEFRLStep())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)

        # Take a dt, find the error, halve dt, and see that error is drops as
        # desired.

        # When
        tf = np.pi
        dt = 0.1*tf
        energy = []
        def callback(t):
            x, u = self.pa.x[0], self.pa.u[0]
            energy.append(0.5*(x*x + u*u))

        callback(0.0)
        self._integrate(integrator, dt, tf, callback)
        energy = np.asarray(energy)
        err1 = np.max(np.abs(energy - 0.5))

        # When
        self.pa.x[0] = 1.0
        self.pa.u[0] = 0.0
        energy = []
        dt *= 0.5
        callback(0.0)
        self._integrate(integrator, dt, tf, callback)
        energy = np.asarray(energy)
        err2 = np.max(np.abs(energy - 0.5))

        # Then
        self.assertTrue(err2 < err1)
        self.assertTrue(err1/err2 > 16.0)


if __name__ == '__main__':
    unittest.main()
