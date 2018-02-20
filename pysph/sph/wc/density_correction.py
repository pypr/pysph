from pysph.sph.equation import Equation
import numpy as np


def declare(s):
    if s == 'int':
        return 0
    if s[:6] == 'matrix':
        tup = s[7:-1]
        return np.zeros(eval(tup))


def gj_Solve(A=[1., 0.], b=[1., 0.], n=3, result=[0., 1.]):
    """ A gauss-jordan method to solve an augmented matrix for the
        unknown variables, x, in Ax = b.
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


class ShepardFilterPreStep(Equation):
    r"""**Shepard Filter density reinitialization**
    This is a zeroth order density reinitialization

    .. math::
            \tilde{W_{ab}} = \frac{W_{ab}}{\sum_{b} W_{ab}\frac{m_{b}}
            {\rho_{b}}}
    References
    ----------
    .. [Panizzo, 2004] Panizzo, Physical and Numerical Modelling of
        Subaerial Landslide Generated Waves, PhD thesis.
    """

    def initialize(self, d_idx, d_tmp_w):
        d_tmp_w[d_idx] = 0.0

    def loop(self, d_idx, d_tmp_w, s_m, s_rho, s_idx, WIJ):
        d_tmp_w[d_idx] += WIJ * s_m[s_idx] / s_rho[s_idx]


class ShepardFilter(Equation):
    r"""
    .. math::
            \rho_{a} = \sum_{b} \m_{b}\tilde{W_{ab}}
    """

    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, s_idx, d_idx, s_m, d_tmp_w, d_rho, WIJ):
        d_rho[d_idx] += WIJ * s_m[s_idx] / d_tmp_w[d_idx]


class MLSFirstOrderPreStep2D(Equation):
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
    References
    ----------
    .. [Dilts, 1999] Dilts, G. A. Moving-Least-Squares-Particle
        Hydrodynamics - I. Consistency and stability,
        Int. J. Numer. Meth. Engng, 1999.
    """

    def _get_helpers_(self):
        return [gj_Solve]

    def initialize(self, d_idx, d_a):
        n = declare('int')
        n = 3
        i = declare('int')
        for i in range(n * n):
            d_a[n * n * d_idx + i] = 0.0

    def loop(self, s_idx, d_idx, d_a, s_m, s_rho, WIJ, XIJ):
        n = declare('int')
        n = 3
        i = declare('int')
        j = declare('int')
        for i in range(n):
            for j in range(n):
                fac1 = 1.0 if i == 0 else XIJ[i - 1]
                fac2 = 1.0 if j == 0 else XIJ[j - 1]
                d_a[n * n * d_idx + n * i + j] += fac1 * \
                    fac2 * WIJ * s_m[s_idx] / s_rho[s_idx]

    def post_loop(self, d_idx, s_idx, d_a, d_b):
        a = declare('matrix((9, ))')
        i = declare('int')
        j = declare('int')
        n = declare('int')
        n = 3
        for i in range(n):
            for j in range(n):
                a[n * i + j] = d_a[n * n * d_idx + n * i + j]
        res = declare('matrix((3,))')
        res[0] = 1.0
        for i in range(1, n):
            res[i] = 0.0
        gj_Solve(a, [1., 0., 0.], n, res)
        d_b[n * d_idx] = res[0]
        d_b[n * d_idx + 1] = res[1]
        d_b[n * d_idx + 2] = res[2]


class MLSFirstOrderPreStep3D(Equation):

    def _get_helpers_(self):
        return [gj_Solve]

    def initialize(self, d_idx, d_a):
        n = declare('int')
        n = 4
        i = declare('int')
        for i in range(n * n):
            d_a[n * n * d_idx + i] = 0.0

    def loop(self, s_idx, d_idx, d_a, s_m, s_rho, WIJ, XIJ):
        n = declare('int')
        n = 4
        i = declare('int')
        j = declare('int')
        for i in range(n):
            for j in range(n):
                fac1 = 1.0 if i == 0 else XIJ[i - 1]
                fac2 = 1.0 if j == 0 else XIJ[j - 1]
                d_a[n * n * d_idx + n * i + j] += fac1 * \
                    fac2 * WIJ * s_m[s_idx] / s_rho[s_idx]

    def post_loop(self, d_idx, s_idx, d_a, d_b):
        a = declare('matrix((16, ))')
        i = declare('int')
        j = declare('int')
        n = declare('int')
        n = 4
        for i in range(n):
            for j in range(n):
                a[n * i + j] = d_a[n * n * d_idx + n * i + j]
        res = declare('matrix((4,))')
        res[0] = 1.0
        for i in range(1, n):
            res[i] = 0.0
        gj_Solve(a, [1., 0., 0., 0.], n, res)
        d_b[n * d_idx] = res[0]
        d_b[n * d_idx + 1] = res[1]
        d_b[n * d_idx + 2] = res[2]
        d_b[n * d_idx + 3] = res[3]


class MLSFirstOrder(Equation):
    r"""
    .. math::
            \rho_{a} = \sum_{b} \m_{b}W_{ab}^{MLS}
    """

    def __init__(self, dest, sources, dim):
        if dim == 2:
            self.n = 3
        elif dim == 3:
            self.n = 4

        super(MLSFirstOrder, self).__init__(dest, sources)

    def initialize(self, d_rho, d_idx):
        d_rho[d_idx] = 0.0

    def loop(self, d_b, d_rho, d_idx, s_idx, s_m, XIJ, WIJ):
        n = declare('int')
        i = declare('int')
        n = self.n
        wmls = declare('double')
        wmls = 0.0
        for i in range(n):
            fac = 1.0 if i == 0 else XIJ[i - 1]
            wmls += d_b[n * d_idx + i] * fac * WIJ
        d_rho[d_idx] += s_m[s_idx] * wmls
