import unittest

import numpy as np

from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline
from pysph.sph.equation import Equation, Group
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.basic_equations import SummationDensity
from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep,  MixedKernelCorrectionPreStep,
    GradientCorrection, MixedGradientCorrection
)


class GradPhi(Equation):
    def initialize(self, d_idx, d_gradu):
        d_gradu[2*d_idx] = 0.0
        d_gradu[2*d_idx + 1] = 0.0

    def loop(self, d_idx, d_gradu, d_u, s_idx, s_m, s_rho, s_u, DWIJ):
        fac = s_m[s_idx]/s_rho[s_idx]*(s_u[s_idx] - d_u[d_idx])
        d_gradu[2*d_idx] += fac*DWIJ[0]
        d_gradu[2*d_idx + 1] += fac*DWIJ[1]


class TestKernelCorrection(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        x, y = np.mgrid[0.5:1:2j, 0.5:1:2j]
        u = x + y
        pa = get_particle_array(
            name='fluid', x=x, y=y, h=0.5, m=1.0, u=u
        )
        pa.add_property('gradu', stride=2)
        pa.add_property('cwij')
        pa.add_property('dw_gamma', stride=3)
        pa.add_property('m_mat', stride=9)
        pa.cwij[:] = 1.0
        self.pa = pa

    def _make_accel_eval(self, equations):
        kernel = CubicSpline(dim=self.dim)
        seval = SPHEvaluator(
            arrays=[self.pa], equations=equations,
            dim=self.dim, kernel=kernel
        )
        return seval

    def test_gradient_correction(self):
        # Given
        pa = self.pa
        dest = 'fluid'
        sources = ['fluid']
        eqs = [
            Group(equations=[
                SummationDensity(dest=dest, sources=sources),
            ]),
            Group(equations=[
                GradientCorrectionPreStep(dest=dest, sources=sources, dim=2)
            ]),
            Group(equations=[
                GradientCorrection(dest=dest, sources=sources,
                                   dim=2, tol=10.0),
                GradPhi(dest=dest, sources=sources)
            ])
        ]
        a_eval = self._make_accel_eval(eqs)

        # When
        a_eval.evaluate(0.0, 0.1)

        # Then
        np.testing.assert_array_almost_equal(pa.gradu, 1.0)

    def test_gradient_correction_perturbed(self):
        # Given
        pa = self.pa
        pa.x[:] = pa.x + [0.1, 0.05, -0.1, -0.05]
        pa.y[:] = pa.y + [0.1, 0.05, -0.1, -0.05]
        pa.u[:] = pa.x + self.pa.y

        self.test_gradient_correction()

    def test_mixed_gradient_correction(self):
        # Given
        pa = self.pa
        dest = 'fluid'
        sources = ['fluid']
        eqs = [
            Group(equations=[
                SummationDensity(dest=dest, sources=sources),
            ]),
            Group(equations=[
                MixedKernelCorrectionPreStep(
                    dest=dest, sources=sources, dim=2
                )
            ]),
            Group(equations=[
                MixedGradientCorrection(dest=dest, sources=sources,
                                        dim=2, tol=20.0),
                GradPhi(dest=dest, sources=sources)
            ])
        ]
        a_eval = self._make_accel_eval(eqs)

        # When
        a_eval.evaluate(0.0, 0.1)

        # Then
        np.testing.assert_array_almost_equal(pa.gradu, 1.0)

    def test_mixed_gradient_correction_perturbed(self):
        # Given
        pa = self.pa
        pa.x[:] = pa.x + [0.1, 0.05, -0.1, -0.05]
        pa.y[:] = pa.y + [0.1, 0.05, -0.1, -0.05]
        pa.u[:] = pa.x + self.pa.y

        self.test_mixed_gradient_correction()
