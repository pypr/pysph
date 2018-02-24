from pysph.sph.equation import Equation
import numpy as np


def declare(s):
    if s == 'int':
        return 0
    if s[:6] == 'matrix':
        tup = s[7:-1]
        return np.zeros(eval(tup))


def gj_Solve(A=[1., 0.], b=[1., 0.], n=3, result=[0., 1.]):
    r""" A gauss-jordan method to solve an augmented matrix for the
        unknown variables, x, in Ax = b.

    References
    ----------
    https://ricardianambivalence.com/2012/10/20/pure-python-gauss-jordan
    -solve-ax-b-invert-a/
    """

    i = declare('int')
    j = declare('int')
    m = declare('matrix((4, 5))')
    for i in range(n):
        for j in range(n):
            m[i][j] = A[n * i + j]
        m[i][n] = b[i]

    eqns = declare('int')
    colrange = declare('int')
    augCol = declare('int')
    eqns = n
    colrange = n
    augCol = n + 1

    col = declare('int')
    row = declare('int')
    bigrow = declare('int')
    for col in range(colrange):
        bigrow = col
        for row in range(col + 1, colrange):
            if abs(m[row][col]) > abs(m[bigrow][col]):
                bigrow = row
                temp = m[row][col]
                m[row][col] = m[bigrow][col]
                m[bigrow][col] = temp

    rr = declare('int')
    rrcol = declare('int')
    for rrcol in range(0, colrange):
        for rr in range(rrcol + 1, eqns):
            cc = -(float(m[rr][rrcol]) / float(m[rrcol][rrcol]))
            for j in range(augCol):
                m[rr][j] = m[rr][j] + cc * m[rrcol][j]

    rb = declare('int')
    rbr = declare('int')
    backCol = declare('int')
    backColr = declare('int')
    kup = declare('int')
    kupr = declare('int')
    kleft = declare('int')
    kleftr = declare('int')
    tol = 1.0e-05
    for rbr in range(eqns):
        rb = eqns - rbr - 1
        if (m[rb][rb] == 0):
            if abs(m[rb][augCol - 1]) >= tol:
                return 0.0
        else:
            for backColr in range(rb, augCol):
                backCol = rb + augCol - backColr - 1
                m[rb][backCol] = float(m[rb][backCol]) / float(m[rb][rb])
            if not (rb == 0):
                for kupr in range(rb):
                    kup = rb - kupr - 1
                    for kleftr in range(rb, augCol):
                        kleft = rb + augCol - kleftr - 1
                        kk = -float(m[kup][rb]) / float(m[rb][rb])
                        m[kup][kleft] = m[kup][kleft] + \
                            kk * float(m[rb][kleft])
    for i in range(n):
        result[i] = m[i][n]


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
                 s_z, s_h, KERNEL, NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('long')
        xij = declare('matrix((3,))')
        tmp_w = 0.0
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            wij = KERNEL.kernel(xij, rij, s_h[s_idx])
            tmp_w += wij * s_m[s_idx] / s_rhotmp[s_idx]
        d_rho[d_idx] = 0.0
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            wij = KERNEL.kernel(xij, rij, s_h[s_idx])
            d_rho[d_idx] += wij * s_m[s_idx] / tmp_w


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
        return [gj_Solve]

    def initialize(self, d_idx, d_rho, d_rhotmp):
        d_rhotmp[d_idx] = d_rho[d_idx]

    def loop_all(self, d_idx, d_rho, d_x, d_y, s_x, s_y, s_h, s_m, s_rhotmp,
                 KERNEL, NBRS, N_NBRS):
        n = declare('int')
        n = 3
        i = declare('int')
        j = declare('int')
        k = declare('long')
        s_idx = declare('int')
        amls = declare('matrix((9,))')
        x = d_x[d_idx]
        y = d_y[d_idx]
        xij = declare('matrix((3,))')
        for i in range(n):
            for j in range(n):
                amls[n * i + j] = 0.0
        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = 0.
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1])
            wij = KERNEL.kernel(xij, rij, s_h[s_idx])
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
        res = declare('matrix((3,))')
        gj_Solve(amls, [1., 0., 0.], n, res)
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
            wij = KERNEL.kernel(xij, rij, s_h[s_idx])
            wmls = (b0 + b1 * xij[0] + b2 * xij[1]) * wij
            d_rho[d_idx] += s_m[s_idx] * wmls


class MLSFirstOrder3D(Equation):

    def _get_helpers_(self):
        return [gj_Solve]

    def initialize(self, d_idx, d_rho, d_rhotmp):
        d_rhotmp[d_idx] = d_rho[d_idx]

    def loop_all(self, d_idx, d_rho, d_x, d_y, d_z, s_x, s_y, s_z, s_h, s_m,
                 s_rhotmp, KERNEL, NBRS, N_NBRS):
        n = declare('int')
        n = 4
        i = declare('int')
        j = declare('int')
        k = declare('long')
        s_idx = declare('int')
        amls = declare('matrix((16,))')
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        xij = declare('matrix((4,))')
        for i in range(n):
            for j in range(n):
                amls[n * i + j] = 0.0
        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            wij = KERNEL.kernel(xij, rij, s_h[s_idx])
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
        res = declare('matrix((4,))')
        gj_Solve(amls, [1., 0., 0., 0.], n, res)
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
            wij = KERNEL.kernel(xij, rij, s_h[s_idx])
            wmls = (b0 + b1 * xij[0] + b2 * xij[1] + b3 * xij[2]) * wij
            d_rho[d_idx] += s_m[s_idx] * wmls
