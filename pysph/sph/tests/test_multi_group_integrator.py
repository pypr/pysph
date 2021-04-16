'''Tests for integrator having different acceleration evaluators for different
stages.

'''
import unittest

import pytest
import numpy as np

from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation, MultiStageEquations
from pysph.sph.acceleration_eval import make_acceleration_evals
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS as NNPS
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep


class Eq1(Equation):
    def initialize(self, d_idx, d_au):
        d_au[d_idx] = 1.0


class Eq2(Equation):
    def initialize(self, d_idx, d_au):
        d_au[d_idx] += 1.0


class MyStepper(IntegratorStep):
    def stage1(self, d_idx, d_u, d_au, dt):
        d_u[d_idx] += d_au[d_idx]*dt

    def stage2(self, d_idx, d_u, d_au, dt):
        d_u[d_idx] += d_au[d_idx]*dt


class MyIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.compute_accelerations(0, update_nnps=False)
        self.stage1()
        self.do_post_stage(dt, 1)
        self.compute_accelerations(1, update_nnps=False)
        self.stage2()
        self.update_domain()
        self.do_post_stage(dt, 2)


class TestMultiGroupIntegrator(unittest.TestCase):
    def setUp(self):
        self.dim = 1
        n = 10
        dx = 1.0/(n-1)
        x = np.linspace(0, 1, n)
        m = np.ones_like(x)
        h = np.ones_like(x)*dx*1.05
        pa = get_particle_array(name='fluid', x=x, h=h, m=m, au=0.0, u=0.0)
        self.pa = pa
        self.NNPS_cls = NNPS
        self.backend = 'cython'

    def _make_integrator(self):
        arrays = [self.pa]
        kernel = CubicSpline(dim=self.dim)
        eqs = [
            [Eq1(dest='fluid', sources=['fluid'])],
            [Eq2(dest='fluid', sources=['fluid'])]
        ]
        meqs = MultiStageEquations(eqs)
        a_evals = make_acceleration_evals(
            arrays, meqs, kernel, backend=self.backend
        )
        integrator = MyIntegrator(fluid=MyStepper())
        comp = SPHCompiler(a_evals, integrator=integrator)
        comp.compile()
        nnps = self.NNPS_cls(dim=kernel.dim, particles=arrays)
        nnps.update()
        for ae in a_evals:
            ae.set_nnps(nnps)
        integrator.set_nnps(nnps)
        return integrator

    def test_different_accels_per_integrator(self):
        # Given
        pa = self.pa
        integrator = self._make_integrator()

        # When
        integrator.step(0.0, 0.1)

        # Then
        if pa.gpu:
            pa.gpu.pull('u', 'au')
        one = np.ones_like(pa.x)
        np.testing.assert_array_almost_equal(pa.au, 2.0*one)
        np.testing.assert_array_almost_equal(pa.u, 0.3*one)


class TestMultiGroupIntegratorGPU(TestMultiGroupIntegrator):
    def setUp(self):
        pytest.importorskip('pyopencl')
        pytest.importorskip('pysph.base.gpu_nnps')
        super(TestMultiGroupIntegratorGPU, self).setUp()
        from pysph.base.gpu_nnps import ZOrderGPUNNPS
        self.NNPS_cls = ZOrderGPUNNPS
        self.backend = 'opencl'


if __name__ == '__main__':
    unittest.main()
