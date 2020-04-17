'''
Kernel Corrections
###################

These are the equations for the kernel corrections that are mentioned in the
paper by Bonet and Lok [BonetLok1999].

References
-----------

    .. [BonetLok1999] Bonet, J. and Lok T.-S.L. (1999)
        Variational and Momentum Preservation Aspects of Smoothed
        Particle Hydrodynamic Formulations.

'''

from math import sqrt
from compyle.api import declare
from pysph.sph.equation import Equation
from pysph.sph.wc.density_correction import gj_solve


class KernelCorrection(Equation):
    r"""**Kernel Correction**

    From [BonetLok1999], equation (53):

    .. math::
            \mathbf{f}_{a} = \frac{\sum_{b}\frac{m_{b}}{\rho_{b}}
            \mathbf{f}_{b}W_{ab}}{\sum_{b}\frac{m_{b}}{\rho_{b}}W_{ab}}
    """

    def initialize(self, d_idx, d_cwij):
        d_cwij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_cwij, s_m, s_rho, WIJ):
        d_cwij[d_idx] += s_m[s_idx] * WIJ / s_rho[s_idx]


class GradientCorrectionPreStep(Equation):

    def __init__(self, dest, sources, dim=2):
        self.dim = dim
        super(GradientCorrectionPreStep, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m_mat):
        i = declare('int')
        for i in range(9):
            d_m_mat[9 * d_idx + i] = 0.0

    def loop_all(self, d_idx, d_m_mat, s_m, s_rho, d_x, d_y, d_z, d_h, s_x,
                 s_y, s_z, s_h, SPH_KERNEL, NBRS, N_NBRS):
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        h = d_h[d_idx]
        i, j, s_idx, n = declare('int', 4)
        xij = declare('matrix(3)')
        dwij = declare('matrix(3)')
        n = self.dim
        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            hij = (h + s_h[s_idx]) * 0.5
            r = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            SPH_KERNEL.gradient(xij, r, hij, dwij)
            V = s_m[s_idx] / s_rho[s_idx]
            if r > 1.0e-12:
                for i in range(n):
                    for j in range(n):
                        xj = xij[j]
                        d_m_mat[9 * d_idx + 3 * i + j] -= V * dwij[i] * xj


class GradientCorrection(Equation):
    r"""**Kernel Gradient Correction**

    From [BonetLok1999], equations (42) and (45)

    .. math::
            \nabla \tilde{W}_{ab} = L_{a}\nabla W_{ab}

    .. math::
            L_{a} = \left(\sum \frac{m_{b}}{\rho_{b}} \nabla W_{ab}
            \mathbf{\otimes}x_{ba} \right)^{-1}
    """

    def _get_helpers_(self):
        return [gj_solve]

    def __init__(self, dest, sources, dim=2, tol=0.1):
        self.dim = dim
        self.tol = tol
        super(GradientCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_m_mat, DWIJ, HIJ):
        i, j, n, nt = declare('int', 4)
        n = self.dim
        nt = n + 1
        # Note that we allocate enough for a 3D case but may only use a
        # part of the matrix.
        temp = declare('matrix(12)')
        res = declare('matrix(3)')
        eps = 1.0e-04 * HIJ
        for i in range(n):
            for j in range(n):
                temp[nt * i + j] = d_m_mat[9 * d_idx + 3 * i + j]
            # Augmented part of matrix
            temp[nt*i + n] = DWIJ[i]

        gj_solve(temp, n, 1, res)

        res_mag = 0.0
        dwij_mag = 0.0
        for i in range(n):
            res_mag += abs(res[i])
            dwij_mag += abs(DWIJ[i])
        change = abs(res_mag - dwij_mag)/(dwij_mag + eps)
        if change < self.tol:
            for i in range(n):
                DWIJ[i] = res[i]


class MixedKernelCorrectionPreStep(Equation):
    r"""**Mixed Kernel Correction**

    From [BonetLok1999], equations (54), (57) and (58)

    .. math::
            \tilde{W}_{ab} = \frac{W_{ab}}{\sum_{b} V_{b}W_{ab}}

    .. math::
            \nabla \tilde{W}_{ab} = L_{a}\nabla \bar{W}_{ab}

    where,

    .. math::
            L_{a} = \left(\sum_{b} V_{b} \nabla \bar{W}_{ab}
            \mathbf{\otimes}x_{ba} \right)^{-1}

    .. math::
            \nabla \bar{W}_{ab} = \frac{\nabla W_{ab} - \gamma}
            {\sum_{b} V_{b}W_{ab}}

    .. math::
            \gamma = \frac{\sum_{b} V_{b}\nabla W_{ab}}
            {\sum_{b} V_{b}W_{ab}}

    """

    def __init__(self, dest, sources, dim=2):
        self.dim = dim
        super(MixedKernelCorrectionPreStep, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m_mat):
        i = declare('int')
        for i in range(9):
            d_m_mat[9 * d_idx + i] = 0.0

    def loop_all(self, d_idx, d_x, d_y, d_z, d_h, s_x, s_y, s_z, s_h,
                 SPH_KERNEL, N_NBRS, NBRS, d_m_mat, s_m, s_rho, d_cwij,
                 d_dw_gamma):
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        h = d_h[d_idx]
        i, j, n, k, s_idx = declare('int', 5)
        n = self.dim
        xij = declare('matrix(3)')
        dwij = declare('matrix(3)')
        dwij1 = declare('matrix(3)')
        numerator = declare('matrix(3)')

        for i in range(3):
            numerator[i] = 0.0
        den = 0.0

        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            V = s_m[s_idx] / s_rho[s_idx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            hij = (h + s_h[s_idx]) * 0.5
            SPH_KERNEL.gradient(xij, rij, hij, dwij)
            wij = SPH_KERNEL.kernel(xij, rij, hij)
            den += V * wij
            for i in range(n):
                numerator[i] += V * dwij[i]

        for i in range(n):
            d_dw_gamma[3*d_idx + i] = numerator[i]/den
        d_cwij[d_idx] = den

        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            hij = (h + s_h[s_idx]) * 0.5
            r = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            SPH_KERNEL.gradient(xij, r, hij, dwij)
            for i in range(n):
                dwij1[i] = (dwij[i] - numerator[i] / den) / den
            V = s_m[s_idx] / s_rho[s_idx]
            if r > 1.0e-12:
                for i in range(n):
                    for j in range(n):
                        xj = xij[j]
                        d_m_mat[9 * d_idx + 3 * i + j] -= V * dwij1[i] * xj


class MixedGradientCorrection(Equation):
    r"""**Mixed Kernel Gradient Correction**

    This is as per [BonetLok1999]. See the MixedKernelCorrectionPreStep for the
    equations.

    """

    def _get_helpers_(self):
        return [gj_solve]

    def __init__(self, dest, sources, dim=2, tol=0.1):
        self.dim = dim
        self.tol = tol
        super(MixedGradientCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_m_mat, d_dw_gamma, d_cwij, DWIJ, HIJ):
        i, j, n, nt = declare('int', 4)
        n = self.dim
        nt = n + 1
        temp = declare('matrix(12)')  # The augmented matrix
        res = declare('matrix(3)')
        dwij = declare('matrix(3)')
        eps = 1.0e-04 * HIJ
        for i in range(n):
            dwij[i] = (DWIJ[i] - d_dw_gamma[3*d_idx + i])/d_cwij[d_idx]
            for j in range(n):
                temp[nt * i + j] = d_m_mat[9 * d_idx + 3 * i + j]
            temp[nt*i + n] = dwij[i]
        gj_solve(temp, n, 1, res)

        res_mag = 0.0
        dwij_mag = 0.0
        for i in range(n):
            res_mag += abs(res[i])
            dwij_mag += abs(dwij[i])
        change = abs(res_mag - dwij_mag)/(dwij_mag + eps)
        if change < self.tol:
            for i in range(n):
                DWIJ[i] = res[i]
