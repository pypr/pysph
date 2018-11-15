'''
CRKSPH corrections
###################

These are equations for the basic kernel corrections in [CRKSPH2017].

References
-----------

    .. [CRKSPH2017] Nicholas Frontiere, Cody D. Raskin, J. Michael Owen (2017)
        CRKSPH - A Conservative Reproducing Kernel Smoothed Particle
        Hydrodynamics Scheme.

'''

from math import sqrt
from pysph.cpy.api import declare
from pysph.sph.equation import Equation
from pysph.sph.wc.linalg import (
    augmented_matrix, dot, gj_solve, identity, mat_vec_mult
)


class CRKSPHPreStep(Equation):

    def __init__(self, dest, sources, dim=2):
        self.dim = dim
        super(CRKSPHPreStep, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [augmented_matrix, gj_solve, identity, dot, mat_vec_mult]

    def loop_all(self, d_idx, d_x, d_y, d_z, d_h, s_x, s_y, s_z, s_h, s_m,
                 s_rho, SPH_KERNEL, NBRS, N_NBRS, d_ai, d_gradai, d_bi,
                 d_gradbi):
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        h = d_h[d_idx]
        i, j, k, s_idx, d, d2 = declare('int', 6)
        alp, bet, gam, phi, psi = declare('int', 5)
        xij = declare('matrix(3)')
        dwij = declare('matrix(3)')
        d = self.dim
        d2 = d*d

        m0 = 0.0
        m1 = declare('matrix(3)')
        m2 = declare('matrix(9)')
        temp_vec = declare('matrix(3)')
        temp_aug_m2 = declare('matrix(18)')
        m2inv = declare('matrix(9)')
        grad_m0 = declare('matrix(3)')
        grad_m1 = declare('matrix(9)')
        grad_m2 = declare('matrix(27)')
        ai = 0.0
        bi = declare('matrix(3)')
        grad_ai = declare('matrix(3)')
        grad_bi = declare('matrix(9)')

        for i in range(3):
            m1[i] = 0.0
            grad_m0[i] = 0.0
            bi[i] = 0.0
            grad_ai[i] = 0.0
            for j in range(3):
                m2[3*i + j] = 0.0
                grad_m1[3*i + j] = 0.0
                grad_bi[3*i + j] = 0.0
                for k in range(3):
                    grad_m2[9*i + 3*j + k] = 0.0

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            hij = (h + s_h[s_idx]) * 0.5
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            wij = SPH_KERNEL.kernel(xij, rij, hij)
            SPH_KERNEL.gradient(xij, rij, hij, dwij)
            V = s_m[s_idx] / s_rho[s_idx]

            m0 += V * wij
            for alp in range(d):
                m1[alp] += V * wij * xij[alp]
                for bet in range(d):
                    m2[d*alp + bet] += V * wij * xij[alp] * xij[bet]
            for gam in range(d):
                grad_m0[gam] += V * dwij[gam]
                for alp in range(d):
                    fac = 1.0 if alp == gam else 0.0
                    temp = (xij[alp] * dwij[gam] + fac * wij)
                    grad_m1[d*gam + alp] += V * temp
                    for bet in range(d):
                        fac2 = 1.0 if bet == gam else 0.0
                        temp = xij[alp] * fac2 + xij[bet] * fac
                        temp2 = (xij[alp] * xij[bet] * dwij[gam] + temp * wij)
                        grad_m2[d2*gam + d*alp + bet] += V * temp2

        identity(m2inv, d)
        augmented_matrix(m2, m2inv, d, d, temp_aug_m2)

        # If is_singular > 0 then matrix was singular
        is_singular = gj_solve(temp_aug_m2, d, d, m2inv)

        if is_singular > 0.0:
            # Cannot do much if the matrix is singular.  Perhaps later
            # we can tag such particles to see if the user can do something.
            pass
        else:
            mat_vec_mult(m2inv, m1, d, temp_vec)

            # Eq. 12.
            ai = 1.0/(m0 - dot(temp_vec, m1, d))
            # Eq. 13.
            mat_vec_mult(m2inv, m1, d, bi)
            for gam in range(d):
                bi[gam] = -bi[gam]

            # Eq. 14. and 15.
            for gam in range(d):
                temp1 = grad_m0[gam]
                for alp in range(d):
                    temp2 = 0.0
                    for bet in range(d):
                        temp1 -= m2inv[d*alp + bet] * (
                            m1[bet] * grad_m1[d*gam + alp] +
                            m1[alp] * grad_m1[d*gam + bet]
                        )
                        temp2 -= (
                            m2inv[d*alp + bet] * grad_m1[d*gam + bet]
                        )
                        for phi in range(d):
                            for psi in range(d):
                                temp1 += (
                                    m2inv[d*alp + phi] * m2inv[d*psi + bet] *
                                    grad_m2[d2*gam + d*phi + psi] *
                                    m1[bet] * m1[alp]
                                )
                                temp2 += (
                                    m2inv[d*alp + phi] * m2inv[d*psi + bet] *
                                    grad_m2[d2*gam + d*phi + psi] * m1[bet]
                                )
                    grad_bi[d*gam + alp] = temp2
                grad_ai[gam] = -ai*ai*temp1

        if N_NBRS < 2 or is_singular > 0.0:
            d_ai[d_idx] = 1.0
            for i in range(d):
                d_gradai[d * d_idx + i] = 0.0
                d_bi[d * d_idx + i] = 0.0
                for j in range(d):
                    d_gradbi[d2 * d_idx + d * i + j] = 0.0
        else:
            d_ai[d_idx] = ai
            for i in range(d):
                d_gradai[d * d_idx + i] = grad_ai[i]
                d_bi[d * d_idx + i] = bi[i]
                for j in range(d):
                    d_gradbi[d2 * d_idx + d * i + j] = grad_bi[d*i + j]


class CRKSPH(Equation):
    r"""**Conservative Reproducing Kernel SPH**

    Equations from the paper [CRKSPH2017].

    .. math::
            W_{ij}^{R} = A_{i}\left(1+B_{i}^{\alpha}x_{ij}^{\alpha}
            \right)W_{ij}
    .. math::
            \partial_{\gamma}W_{ij}^{R} = A_{i}\left(1+B_{i}^{\alpha}
            x_{ij}^{\alpha}\right)\partial_{\gamma}W_{ij} +
            \partial_{\gamma}A_{i}\left(1+B_{i}^{\alpha}x_{ij}^{\alpha}
            \right)W_{ij} + A_{i}\left(\partial_{\gamma}B_{i}^{\alpha}
            x_{ij}^{\alpha} + B_{i}^{\gamma}\right)W_{ij}
    .. math::
            \nabla\tilde{W}_{ij} = 0.5 * \left(\nabla W_{ij}^{R}-\nabla
             W_{ji}^{R} \right)

    where,

    .. math::
            A_{i} = \left[m_{0} - \left(m_{2}^{-1}\right)^{\alpha \beta}
            m_1^{\beta}m_1^{\alpha}\right]^{-1}
    .. math::
            B_{i}^{\alpha} = -\left(m_{2}^{-1}\right)^{\alpha \beta}
            m_{1}^{\beta}
    .. math::
            \partial_{\gamma}A_{i} = -A_{i}^{2}\left(\partial_{\gamma}
            m_{0}-\left(m_{2}^{-1}\right)^{\alpha \beta}\left(
            m_{1}^{\beta}\partial_{\gamma}m_{1}^{\alpha} +
            \partial_{\gamma}m_{1}^{\beta}m_{1}^{\alpha}\right) +
            \left(m_{2}^{-1}\right)^{\alpha \phi}\partial_{\gamma}
            m_{2}^{\phi \psi}\left(m_{2}^{-1}\right)^{\psi \beta}
            m_{1}^{\beta}m_{1}^{\alpha} \right)
    .. math::
            \partial_{\gamma}B_{i}^{\alpha} = -\left(m_{2}^{-1}\right)^
            {\alpha \beta}\partial_{\gamma}m_{1}^{\beta} +
            \left(m_{2}^{-1}\right)^
            {\alpha \phi}\partial_{\gamma}m_{2}^{\phi \psi}\left(m_{2}^
            {-1}\right)^{\psi \beta}m_{1}^{\beta}
    .. math::
            m_{0} = \sum_{j}V_{j}W_{ij}
    .. math::
            m_{1}^{\alpha} = \sum_{j}V_{j}x_{ij}^{\alpha}W_{ij}
    .. math::
            m_{2}^{\alpha \beta} = \sum_{j}V_{j}x_{ij}^{\alpha}
            x_{ij}^{\beta}W_{ij}
    .. math::
            \partial_{\gamma}m_{0} = \sum_{j}V_{j}\partial_{\gamma}
            W_{ij}
    .. math::
            \partial_{\gamma}m_{1}^{\alpha} = \sum_{j}V_{j}\left[
            x_{ij}^{\alpha}\partial_{\gamma}W_{ij}+\delta^
            {\alpha \gamma}W_{ij} \right]
    .. math::
            \partial_{\gamma}m_{2}^{\alpha \beta} = \sum_{j}V_{j}\left[
            x_{ij}^{\alpha}x_{ij}^{\beta}\partial_{\gamma}W_{ij} +
            \left(x_{ij}^{\alpha}\delta^{\beta \gamma} + x_{ij}^{\beta}
            \delta^{\alpha \gamma} \right)W_{ij} \right]
    """

    def __init__(self, dest, sources, dim=2, tol=0.5):
        self.dim = dim
        self.tol = tol
        super(CRKSPH, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_ai, d_gradai, d_cwij, d_bi, d_gradbi,
             WIJ, DWIJ, XIJ, HIJ):
        alp, gam, d = declare('int', 3)
        res = declare('matrix(3)')
        dbxij = declare('matrix(3)')
        d = self.dim
        ai = d_ai[d_idx]
        eps = 1.0e-04 * HIJ
        bxij = 0.0
        for alp in range(d):
            bxij += d_bi[d*d_idx + alp] * XIJ[alp]
        for gam in range(d):
            temp = 0.0
            for alp in range(d):
                temp += d_gradbi[d*d*d_idx + d*gam + alp]*XIJ[alp]
            dbxij[gam] = temp

        d_cwij[d_idx] = 1.0/(ai*(1 + bxij))

        for gam in range(d):
            res[gam] = ((ai * DWIJ[gam] + d_gradai[d * d_idx + gam] * WIJ) *
                        (1 + bxij))
            res[gam] += ai * (dbxij[gam] + d_bi[d * d_idx + gam]) * WIJ

        res_mag = 0.0
        dwij_mag = 0.0
        for i in range(d):
            res_mag += abs(res[i])
            dwij_mag += abs(DWIJ[i])
        change = abs(res_mag - dwij_mag)/(dwij_mag + eps)
        if change < self.tol:
            for i in range(d):
                DWIJ[i] = res[i]


class CRKSPHSymmetric(Equation):
    r"""**Conservative Reproducing Kernel SPH**

    This is symmetric and will only work for particles of the same array.

    Equations from the paper [CRKSPH2017].

    .. math::
            W_{ij}^{R} = A_{i}\left(1+B_{i}^{\alpha}x_{ij}^{\alpha}
            \right)W_{ij}
    .. math::
            \partial_{\gamma}W_{ij}^{R} = A_{i}\left(1+B_{i}^{\alpha}
            x_{ij}^{\alpha}\right)\partial_{\gamma}W_{ij} +
            \partial_{\gamma}A_{i}\left(1+B_{i}^{\alpha}x_{ij}^{\alpha}
            \right)W_{ij} + A_{i}\left(\partial_{\gamma}B_{i}^{\alpha}
            x_{ij}^{\alpha} + B_{i}^{\gamma}\right)W_{ij}
    .. math::
            \nabla\tilde{W}_{ij} = 0.5 * \left(\nabla W_{ij}^{R}-\nabla
             W_{ji}^{R} \right)

    where,

    .. math::
            A_{i} = \left[m_{0} - \left(m_{2}^{-1}\right)^{\alpha \beta}
            m1_{\beta}m1_{\alpha}\right]^{-1}
    .. math::
            B_{i}^{\alpha} = -\left(m_{2}^{-1}\right)^{\alpha \beta}
            m_{1}^{\beta}
    .. math::
            \partial_{\gamma}A_{i} = -A_{i}^{2}\left(\partial_{\gamma}
            m_{0}-\left[m_{2}^{-1}\right]^{\alpha \beta}\left[
            m_{1}^{\beta}\partial_{\gamma}m_{1}^{\beta}m_{1}^{\alpha} +
            \partial_{\gamma}m_{1}^{\alpha}m_{1}^{\beta}\right] +
            \left[m_{2}^{-1}\right]^{\alpha \phi}\partial_{\gamma}
            m_{2}^{\phi \psi}\left[m_{2}^{-1}\right]^{\psi \beta}
            m_{1}^{\beta}m_{1}^{\alpha} \right)
    .. math::
            \partial_{\gamma}B_{i}^{\alpha} = -\left[m_{2}^{-1}\right]^
            {\alpha \beta}\left[m_{1}^{\beta} + \left(m_{2}^{-1}\right)^
            {\alpha \phi}\partial_{\gamma}m_{2}^{\phi \psi}\left(m_{2}^
            {-1}\right)^{\psi \beta}m_{1}^{\beta}
    .. math::
            m_{0} = \sum_{j}V_{j}W_{ij}
    .. math::
            m_{1}^{\alpha} = \sum_{j}V_{j}x_{ij}^{\alpha}W_{ij}
    .. math::
            m_{2}^{\alpha \beta} = \sum_{j}V_{j}x_{ij}^{\alpha}
            x_{ij}^{\beta}W_{ij}
    .. math::
            \partial_{\gamma}m_{0} = \sum_{j}V_{j}\partial_{\gamma}
            W_{ij}
    .. math::
            \partial_{\gamma}m_{1}^{\alpha} = \sum_{j}V_{j}\left[
            x_{ij}^{\alpha}\partial_{\gamma}W_{ij}+\delta^
            {\alpha \gamma}W_{ij} \right]
    .. math::
            \partial_{\gamma}m_{2}^{\alpha \beta} = \sum_{j}V_{j}\left[
            x_{ij}^{\alpha}x_{ij}^{\beta}\partial_{\gamma}W_{ij} +
            \left(x_{ij}^{\alpha}\delta^{\beta \gamma} + x_{ij}^{\beta}
            \delta^{\alpha \gamma} \right)W_{ij} \right]
    """

    def __init__(self, dest, sources, dim=2, tol=0.5):
        self.dim = dim
        self.tol = tol
        super(CRKSPHSymmetric, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_ai, d_gradai, d_cwij, d_bi, d_gradbi, s_ai,
             s_gradai, s_bi, s_gradbi, WIJ, DWIJ, XIJ, HIJ):
        alp, gam, d = declare('int', 3)
        res = declare('matrix(3)')
        dbxij = declare('matrix(3)')
        dbxji = declare('matrix(3)')
        d = self.dim
        ai = d_ai[d_idx]
        aj = s_ai[s_idx]
        eps = 1.0e-04 * HIJ
        bxij = 0.0
        bxji = 0.0
        for alp in range(d):
            bxij += d_bi[d*d_idx + alp] * XIJ[alp]
            bxji -= s_bi[d*s_idx + alp] * XIJ[alp]
        for gam in range(d):
            temp = 0.0
            temp1 = 0.0
            for alp in range(d):
                temp += d_gradbi[d*d*d_idx + d*gam + alp]*XIJ[alp]
                temp1 -= s_gradbi[d*d*s_idx + d*gam + alp]*XIJ[alp]
            dbxij[gam] = temp
            dbxji[gam] = temp1

        d_cwij[d_idx] = 1.0/(ai*(1 + bxij))

        for gam in range(d):
            temp = ((ai * DWIJ[gam] + d_gradai[d * d_idx + gam] * WIJ) *
                    (1 + bxij))
            temp += ai * (dbxij[gam] + d_bi[d * d_idx + gam]) * WIJ
            temp += ((aj * DWIJ[gam] - s_gradai[d * s_idx + gam] * WIJ) *
                     (1 + bxji))
            temp -= aj * (dbxji[gam] + s_bi[d * s_idx + gam]) * WIJ
            res[gam] = 0.5*temp

        res_mag = 0.0
        dwij_mag = 0.0
        for i in range(d):
            res_mag += abs(res[i])
            dwij_mag += abs(DWIJ[i])
        change = abs(res_mag - dwij_mag)/(dwij_mag + eps)
        if change < self.tol:
            for i in range(d):
                DWIJ[i] = res[i]
