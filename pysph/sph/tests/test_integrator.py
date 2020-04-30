# Standard library imports.
import unittest

# Library imports.
import numpy as np
import pytest

# Local imports.
from pysph.base.utils import get_particle_array, get_particle_array_wcsph
from compyle.config import get_config
from pysph.sph.equation import Equation
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.integrator import (LeapFrogIntegrator, PECIntegrator,
                                  PEFRLIntegrator, EulerIntegrator)
from pysph.sph.integrator_step import (
    IntegratorStep, LeapFrogStep, PEFRLStep, TwoStageRigidBodyStep
)


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


class EulerStep(IntegratorStep):
    def stage1(self, d_idx, d_x, d_u, dt):
        d_x[d_idx] += dt*d_u[d_idx]


class TestIntegratorAdaptiveTimestep(TestIntegratorBase):
    def test_compute_timestep_without_adaptive(self):
        # Given.
        integrator = EulerIntegrator(fluid=EulerStep())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)

        # When
        dt = integrator.compute_time_step(0.1, 0.5)

        # Then
        self.assertEqual(dt, None)

    def test_compute_timestep_with_dt_adapt(self):
        # Given.
        self.pa.extend(1)
        self.pa.align_particles()
        self.pa.add_property('dt_adapt')
        self.pa.dt_adapt[:] = [0.1, 0.2]

        integrator = EulerIntegrator(fluid=EulerStep())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)

        # When
        dt = integrator.compute_time_step(0.1, 0.5)

        # Then
        self.assertEqual(dt, 0.1)

    def test_compute_timestep_with_dt_adapt_with_invalid_values(self):
        # Given.
        self.pa.extend(1)
        self.pa.align_particles()
        self.pa.add_property('dt_adapt')
        self.pa.dt_adapt[:] = [0.0, -2.0]

        integrator = EulerIntegrator(fluid=EulerStep())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)

        # When
        dt = integrator.compute_time_step(0.1, 0.5)

        # Then
        self.assertEqual(dt, None)

    def test_compute_timestep_with_dt_adapt_trumps_dt_cfl(self):
        # Given.
        self.pa.extend(1)
        self.pa.align_particles()
        self.pa.add_property('dt_adapt')
        self.pa.add_property('dt_cfl')
        self.pa.h[:] = 1.0
        self.pa.dt_adapt[:] = [0.1, 0.2]
        self.pa.dt_cfl[:] = 1.0

        integrator = EulerIntegrator(fluid=EulerStep())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)

        # When
        cfl = 0.5
        dt = integrator.compute_time_step(0.1, cfl)

        # Then
        self.assertEqual(dt, 0.1)

    def test_compute_timestep_with_dt_cfl(self):
        # Given.
        self.pa.extend(1)
        self.pa.align_particles()
        self.pa.add_property('dt_cfl')
        self.pa.h[:] = 1.0
        self.pa.dt_cfl[:] = [1.0, 2.0]

        integrator = EulerIntegrator(fluid=EulerStep())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)

        # When
        cfl = 0.5
        dt = integrator.compute_time_step(0.1, cfl)

        # Then
        expect = cfl*1.0/(2.0)
        self.assertEqual(dt, expect)


class S1Step(IntegratorStep):

    def py_stage1(self, dest, t, dt):
        self.called_with1 = t, dt
        dest.u[:] = 1.0

    def stage1(self, d_idx, d_u, d_au, dt):
        d_u[d_idx] += d_au[d_idx] * dt * 0.5

    def stage2(self, d_idx, d_x, d_u, d_au, dt):
        d_u[d_idx] += 0.5*dt * d_au[d_idx]
        d_x[d_idx] += dt * d_u[d_idx]


class S12Step(IntegratorStep):

    def py_stage1(self, dest, t, dt):
        self.called_with1 = t, dt
        dest.u[:] = 1.0
        if dest.gpu:
            dest.gpu.push('u')

    def stage1(self, d_idx, d_u, d_au, dt):
        d_u[d_idx] += d_au[d_idx] * dt * 0.5

    def py_stage2(self, dest, t, dt):
        self.called_with2 = t, dt
        if dest.gpu:
            dest.gpu.pull('u')
        dest.u += 0.5
        if dest.gpu:
            dest.gpu.push('u')

    def stage2(self, d_idx, d_x, d_u, d_au, dt):
        d_u[d_idx] += 0.5*dt * d_au[d_idx]
        d_x[d_idx] += dt * d_u[d_idx]


class OnlyPyStep(IntegratorStep):

    def py_stage1(self, dest, t, dt):
        self.called_with1 = t, dt
        dest.x[:] = 0.0
        dest.u[:] = 1.0

    def py_stage2(self, dest, t, dt):
        self.called_with2 = t, dt
        dest.u += 0.5
        dest.x += 0.5


def my_helper(dt=0.0):
    return dt*2.0


class StepWithHelper(IntegratorStep):

    def _get_helpers_(self):
        return [my_helper]

    def stage1(self, d_idx, d_u, d_au, dt):
        d_u[d_idx] += d_au[d_idx] * my_helper(dt)


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

    def test_integrator_calls_py_stage1(self):
        # Given.
        stepper = S1Step()
        integrator = LeapFrogIntegrator(fluid=stepper)
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)
        tf = 1.0
        dt = tf

        # When
        call_data = []

        def callback(t):
            call_data.append(t)

        self._integrate(integrator, dt, tf, callback)

        # Then
        self.assertEqual(len(call_data), 1)
        self.assertTrue(hasattr(stepper, 'called_with1'))
        self.assertEqual(stepper.called_with1, (0.0, dt))
        # These are not physically significant as the main purpose is to see
        # if the py_stage* methods are called.
        np.testing.assert_array_almost_equal(self.pa.x, [1.5])
        np.testing.assert_array_almost_equal(self.pa.u, [0.5])

    def test_integrator_calls_py_stage1_stage2(self):
        # Given.
        stepper = S12Step()
        integrator = LeapFrogIntegrator(fluid=stepper)
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)
        tf = 1.0
        dt = tf

        # When
        def callback(t):
            pass

        self._integrate(integrator, dt, tf, callback)

        # Then
        self.assertTrue(hasattr(stepper, 'called_with1'))
        self.assertEqual(stepper.called_with1, (0.0, dt))
        self.assertTrue(hasattr(stepper, 'called_with2'))
        self.assertEqual(stepper.called_with2, (0.5*dt, dt))
        # These are not physically significant as the main purpose is to see
        # if the py_stage* methods are called.
        np.testing.assert_array_almost_equal(self.pa.x, [2.0])
        np.testing.assert_array_almost_equal(self.pa.u, [1.0])

    def test_integrator_calls_only_py_when_no_stage(self):
        # Given.
        stepper = OnlyPyStep()
        integrator = LeapFrogIntegrator(fluid=stepper)
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)
        tf = 1.0
        dt = tf

        # When
        def callback(t):
            pass

        self._integrate(integrator, dt, tf, callback)

        # Then
        self.assertTrue(hasattr(stepper, 'called_with1'))
        self.assertEqual(stepper.called_with1, (0.0, dt))
        self.assertTrue(hasattr(stepper, 'called_with2'))
        self.assertEqual(stepper.called_with2, (0.5*dt, dt))
        np.testing.assert_array_almost_equal(self.pa.x, [0.5])
        np.testing.assert_array_almost_equal(self.pa.u, [1.5])

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

    def test_helper_can_be_used_with_stepper(self):
        # Given.
        integrator = EulerIntegrator(fluid=StepWithHelper())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)

        # When
        tf = 1.0
        dt = tf/2

        def callback(t):
            pass

        self._integrate(integrator, dt, tf, callback)

        # Then
        if self.pa.gpu is not None:
            self.pa.gpu.pull('u')
        u = self.pa.u
        self.assertEqual(u, -2.0*self.pa.x)


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


class TestLeapFrogIntegratorGPU(TestIntegratorBase):
    def _setup_integrator(self, equations, integrator):
        pytest.importorskip('pysph.base.gpu_nnps')
        kernel = CubicSpline(dim=1)
        arrays = [self.pa]
        from pysph.base.gpu_nnps import BruteForceNNPS as GPUNNPS
        a_eval = AccelerationEval(
             particle_arrays=arrays, equations=equations, kernel=kernel,
             backend='opencl'
        )
        comp = SPHCompiler(a_eval, integrator=integrator)
        comp.compile()
        nnps = GPUNNPS(dim=kernel.dim, particles=arrays, cache=True,
                       backend='opencl')
        nnps.update()
        a_eval.set_nnps(nnps)
        integrator.set_nnps(nnps)

    def test_leapfrog(self):
        # Given.
        integrator = LeapFrogIntegrator(fluid=LeapFrogStep())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)
        tf = np.pi/5
        dt = 0.05*tf

        # When
        energy = []

        def callback(t):
            self.pa.gpu.pull('x', 'u')
            x, u = self.pa.x[0], self.pa.u[0]
            energy.append(0.5*(x*x + u*u))

        callback(0.0)
        self._integrate(integrator, dt, tf, callback)

        # Then
        energy = np.asarray(energy)
        self.assertAlmostEqual(np.max(np.abs(energy - 0.5)), 0.0, places=3)

    def test_py_stage_is_called_on_gpu(self):
        # Given.
        stepper = S12Step()
        integrator = LeapFrogIntegrator(fluid=stepper)
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)
        dt = 1.0
        tf = dt

        # When
        def callback(t):
            pass

        self._integrate(integrator, dt, tf, callback)
        self.pa.gpu.pull('x', 'u')

        # Then
        self.assertTrue(hasattr(stepper, 'called_with1'))
        self.assertEqual(stepper.called_with1, (0.0, dt))
        self.assertTrue(hasattr(stepper, 'called_with2'))
        self.assertEqual(stepper.called_with2, (0.5*dt, dt))
        # These are not physically significant as the main purpose is to see
        # if the py_stage* methods are called.
        np.testing.assert_array_almost_equal(self.pa.x, [2.0])
        np.testing.assert_array_almost_equal(self.pa.u, [1.0])

    def test_leapfrog_with_double(self):
        orig = get_config().use_double

        def _cleanup():
            get_config().use_double = orig
        get_config().use_double = True
        self.addCleanup(_cleanup)
        self.test_leapfrog()

    def test_helper_can_be_used_with_stepper_on_gpu(self):
        # Given.
        integrator = EulerIntegrator(fluid=StepWithHelper())
        equations = [SHM(dest="fluid", sources=None)]
        self._setup_integrator(equations=equations, integrator=integrator)

        # When
        tf = 1.0
        dt = tf/2

        def callback(t):
            pass

        self._integrate(integrator, dt, tf, callback)

        # Then
        if self.pa.gpu is not None:
            self.pa.gpu.pull('u')
        u = self.pa.u
        self.assertEqual(u, -2.0*self.pa.x)


class TestLeapFrogIntegratorCUDA(TestLeapFrogIntegratorGPU):
    def _setup_integrator(self, equations, integrator):
        pytest.importorskip('pycuda')
        pytest.importorskip('pysph.base.gpu_nnps')
        kernel = CubicSpline(dim=1)
        arrays = [self.pa]
        from pysph.base.gpu_nnps import BruteForceNNPS as GPUNNPS
        a_eval = AccelerationEval(
             particle_arrays=arrays, equations=equations, kernel=kernel,
             backend='cuda'
        )
        comp = SPHCompiler(a_eval, integrator=integrator)
        comp.compile()
        nnps = GPUNNPS(dim=kernel.dim, particles=arrays, cache=True,
                       backend='cuda')
        nnps.update()
        a_eval.set_nnps(nnps)
        integrator.set_nnps(nnps)


if __name__ == '__main__':
    unittest.main()
