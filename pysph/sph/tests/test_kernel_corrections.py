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
from pysph.sph.wc.crksph import CRKSPHPreStep, CRKSPH, CRKSPHSymmetric


class GradPhi(Equation):
    def initialize(self, d_idx, d_gradu):
        d_gradu[2*d_idx] = 0.0
        d_gradu[2*d_idx + 1] = 0.0

    def loop(self, d_idx, d_gradu, d_u, s_idx, s_m, s_rho, s_u, DWIJ):
        fac = s_m[s_idx]/s_rho[s_idx]*(s_u[s_idx] - d_u[d_idx])
        d_gradu[2*d_idx] += fac*DWIJ[0]
        d_gradu[2*d_idx + 1] += fac*DWIJ[1]


class GradPhiSymm(Equation):
    def initialize(self, d_idx, d_gradu):
        d_gradu[2*d_idx] = 0.0
        d_gradu[2*d_idx + 1] = 0.0

    def loop(self, d_idx, d_rho, d_m, d_gradu, d_u, s_idx, s_m, s_rho, s_u,
             DWIJ):
        fac = s_m[s_idx]/s_rho[s_idx]*(s_u[s_idx] + d_u[d_idx])/d_rho[d_idx]
        d_gradu[2*d_idx] += fac*DWIJ[0]
        d_gradu[2*d_idx + 1] += fac*DWIJ[1]


class VerifyCRKSPH(Equation):
    def initialize(self, d_idx, d_zero_mom, d_first_mom):
        d_zero_mom[d_idx] = 0.0
        d_first_mom[3*d_idx] = 0.0
        d_first_mom[3*d_idx + 1] = 0.0
        d_first_mom[3*d_idx + 2] = 0.0

    def loop(self, d_idx, d_zero_mom, d_first_mom, d_cwij, s_idx, s_m,
             s_rho, WIJ, XIJ):
        vjwijp = s_m[s_idx]/s_rho[s_idx]*WIJ/d_cwij[d_idx]
        d_zero_mom[d_idx] += vjwijp
        d_first_mom[3*d_idx] += vjwijp * XIJ[0]
        d_first_mom[3*d_idx + 1] += vjwijp * XIJ[1]
        d_first_mom[3*d_idx + 2] += vjwijp * XIJ[2]


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
        # for crksph
        pa.add_property('ai')
        pa.add_property('gradai', stride=2)
        pa.add_property('bi', stride=2)
        pa.add_property('gradbi', stride=4)
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

    def test_crksph(self):
        # Given
        pa = self.pa
        dest = 'fluid'
        sources = ['fluid']
        pa.add_property('zero_mom')
        pa.add_property('first_mom', stride=3)
        pa.rho[:] = 1.0
        eqs = [
            Group(equations=[
                SummationDensity(dest=dest, sources=sources),
            ]),
            Group(equations=[
                CRKSPHPreStep(dest=dest, sources=sources, dim=2)
            ]),
            Group(equations=[
                CRKSPH(dest=dest, sources=sources,
                       dim=2, tol=1000.0),
                GradPhi(dest=dest, sources=sources),
                VerifyCRKSPH(dest=dest, sources=sources)
            ])
        ]
        a_eval = self._make_accel_eval(eqs)

        # When
        a_eval.evaluate(0.0, 0.1)

        # Then
        np.testing.assert_array_almost_equal(pa.zero_mom, 1.0)
        np.testing.assert_array_almost_equal(pa.first_mom, 0.0)
        np.testing.assert_array_almost_equal(pa.gradu, 1.0)

    def test_crksph_perturbed(self):
        # Given
        pa = self.pa
        pa.x[:] = pa.x + [0.1, 0.05, -0.1, -0.05]
        pa.y[:] = pa.y + [0.1, 0.05, -0.1, -0.05]
        pa.u[:] = pa.x + self.pa.y

        self.test_crksph()

    def test_crksph_symmetric(self):
        # Given
        pa = self.pa
        dest = 'fluid'
        sources = ['fluid']
        pa.add_property('zero_mom')
        pa.add_property('first_mom', stride=3)
        pa.rho[:] = 1.0
        eqs = [
            Group(equations=[
                SummationDensity(dest=dest, sources=sources),
            ]),
            Group(equations=[
                CRKSPHPreStep(dest=dest, sources=sources, dim=2)
            ]),
            Group(equations=[
                CRKSPHSymmetric(dest=dest, sources=sources,
                                dim=2, tol=1000.0),
                GradPhiSymm(dest=dest, sources=sources),
                VerifyCRKSPH(dest=dest, sources=sources)
            ])
        ]
        a_eval = self._make_accel_eval(eqs)

        # When
        a_eval.evaluate(0.0, 0.1)

        # Then
        np.testing.assert_array_almost_equal(pa.zero_mom, 1.0)
        np.testing.assert_array_almost_equal(pa.first_mom, 0.0)
        # Here all we can test is that the total acceleration is zero.
        print(pa.gradu)
        self.assertAlmostEqual(np.sum(pa.gradu[::2]), 0.0)
        self.assertAlmostEqual(np.sum(pa.gradu[1::2]), 0.0)

    def test_crksph_symmetric_perturbed(self):
        # Given
        pa = self.pa
        pa.x[:] = pa.x + [0.1, 0.05, -0.1, -0.05]
        pa.y[:] = pa.y + [0.1, 0.05, -0.1, -0.05]
        pa.u[:] = pa.x + self.pa.y

        self.test_crksph_symmetric()
