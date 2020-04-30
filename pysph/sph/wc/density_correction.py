from math import sqrt
from pysph.sph.equation import Equation
from compyle.api import declare
from pysph.sph.wc.linalg import gj_solve, augmented_matrix


class ShepardFilter(Equation):
    r"""**Shepard Filter density reinitialization**
    This is a zeroth order density reinitialization

    .. math::
            \tilde{W_{ab}} = \frac{W_{ab}}{\sum_{b} W_{ab}\frac{m_{b}}
            {\rho_{b}}}

    .. math::
            \rho_{a} = \sum_{b} \m_{b}\tilde{W_{ab}}

    References
    ----------
    .. [Panizzo, 2004] Panizzo, Physical and Numerical Modelling of
        Subaerial Landslide Generated Waves, PhD thesis.
    """

    def initialize(self, d_idx, d_rho, d_rhotmp):
        d_rhotmp[d_idx] = d_rho[d_idx]

    def loop_all(self, d_idx, d_rho, d_x, d_y, d_z, s_m, s_rhotmp, s_x, s_y,
                 s_z, d_h, s_h, SPH_KERNEL, NBRS, N_NBRS):
        i, s_idx = declare('int', 2)
        xij = declare('matrix(3)')
        tmp_w = 0.0
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        d_rho[d_idx] = 0.0
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            hij = (d_h[d_idx] + s_h[s_idx]) * 0.5
            wij = SPH_KERNEL.kernel(xij, rij, hij)
            tmp_w += wij * s_m[s_idx] / s_rhotmp[s_idx]
            d_rho[d_idx] += wij * s_m[s_idx]
        d_rho[d_idx] /= tmp_w


class MLSFirstOrder2D(Equation):
    r"""**Moving Least Squares density reinitialization**
    This is a first order density reinitialization

    .. math::
            W_{ab}^{MLS} = \beta\left(\mathbf{r_{a}}\right)\cdot\left(
            \mathbf{r}_a - \mathbf{r}_b\right)W_{ab}

    .. math::
            \beta\left(\mathbf{r_{a}}\right) = A^{-1}
            \left[1 0 0\right]^{T}

    where

    .. math::
            A = \sum_{b}W_{ab}\tilde{A}\frac{m_{b}}{\rho_{b}}

    .. math::
            \tilde{A} = pp^{T}

    where

    .. math::
            p = \left[1 x_{a}-x_{b} y_{a}-y_{b}\right]^{T}

    .. math::
            \rho_{a} = \sum_{b} \m_{b}W_{ab}^{MLS}
    References
    ----------
    .. [Dilts, 1999] Dilts, G. A. Moving-Least-Squares-Particle
        Hydrodynamics - I. Consistency and stability,
        Int. J. Numer. Meth. Engng, 1999.
    """

    def _get_helpers_(self):
        return [gj_solve, augmented_matrix]

    def initialize(self, d_idx, d_rho, d_rhotmp):
        d_rhotmp[d_idx] = d_rho[d_idx]

    def loop_all(self, d_idx, d_rho, d_x, d_y, s_x, s_y, d_h, s_h, s_m,
                 s_rhotmp, SPH_KERNEL, NBRS, N_NBRS):
        n, i, j, k, s_idx = declare('int', 5)
        n = 3
        amls = declare('matrix(9)')
        aug_mls = declare('matrix(12)')
        x = d_x[d_idx]
        y = d_y[d_idx]
        xij = declare('matrix(3)')
        for i in range(n):
            for j in range(n):
                amls[n * i + j] = 0.0
        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = 0.
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])
            hij = (d_h[d_idx] + s_h[s_idx]) * 0.5
            wij = SPH_KERNEL.kernel(xij, rij, hij)
            for i in range(n):
                if i == 0:
                    fac1 = 1.0
                else:
                    fac1 = xij[i - 1]
                for j in range(n):
                    if j == 0:
                        fac2 = 1.0
                    else:
                        fac2 = xij[j - 1]
                    amls[n * i + j] += fac1 * fac2 * \
                        s_m[s_idx] * wij / s_rhotmp[s_idx]
        res = declare('matrix(3)')
        res[0] = 1.0
        res[1] = 0.0
        res[2] = 0.0
        augmented_matrix(amls, res, 3, 1, 3, aug_mls)
        gj_solve(aug_mls, n, 1, res)
        b0 = res[0]
        b1 = res[1]
        b2 = res[2]
        d_rho[d_idx] = 0.0
        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = 0.
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])
            hij = (d_h[d_idx] + s_h[s_idx]) * 0.5
            wij = SPH_KERNEL.kernel(xij, rij, hij)
            wmls = (b0 + b1 * xij[0] + b2 * xij[1]) * wij
            d_rho[d_idx] += s_m[s_idx] * wmls


class MLSFirstOrder3D(Equation):

    def _get_helpers_(self):
        return [gj_solve, augmented_matrix]

    def initialize(self, d_idx, d_rho, d_rhotmp):
        d_rhotmp[d_idx] = d_rho[d_idx]

    def loop_all(self, d_idx, d_rho, d_x, d_y, d_z, s_x, s_y, s_z, d_h, s_h,
                 s_m, s_rhotmp, SPH_KERNEL, NBRS, N_NBRS):
        n, i, j, k, s_idx = declare('int', 5)
        n = 4
        amls = declare('matrix(16)')
        aug_mls = declare('matrix(20)')
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        xij = declare('matrix(4)')
        for i in range(n):
            for j in range(n):
                amls[n * i + j] = 0.0
        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            hij = (d_h[d_idx] + s_h[s_idx]) * 0.5
            wij = SPH_KERNEL.kernel(xij, rij, hij)
            for i in range(n):
                if i == 0:
                    fac1 = 1.0
                else:
                    fac1 = xij[i - 1]
                for j in range(n):
                    if j == 0:
                        fac2 = 1.0
                    else:
                        fac2 = xij[j - 1]
                    amls[n * i + j] += fac1 * fac2 * \
                        s_m[s_idx] * wij / s_rhotmp[s_idx]
        res = declare('matrix(4)')
        res[0] = 1.0
        res[1] = 0.0
        res[2] = 0.0
        res[3] = 0.0
        augmented_matrix(amls, res, n, 1, aug_mls)
        gj_solve(aug_mls, n, 1, res)
        b0 = res[0]
        b1 = res[1]
        b2 = res[2]
        b3 = res[3]
        d_rho[d_idx] = 0.0
        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            hij = (d_h[d_idx] + s_h[s_idx]) * 0.5
            wij = SPH_KERNEL.kernel(xij, rij, hij)
            wmls = (b0 + b1 * xij[0] + b2 * xij[1] + b3 * xij[2]) * wij
            d_rho[d_idx] += s_m[s_idx] * wmls
