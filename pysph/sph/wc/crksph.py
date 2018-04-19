from pysph.sph.equation import Equation


class CRKSPHPreStep(Equation):

    def __init__(self, dest, sources, dim=2):
        self.dim = dim
        super(CRKSPHPreStep, self).__init__(dest, sources)

    def loop_all(self, d_idx, d_x, d_y, d_z, d_h, s_x, s_y, s_z, s_h, s_m,
                 s_rho, KERNEL, NBRS, N_NBRS, d_ai, d_gradai, d_bi, d_gradbi):
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        h = d_h[d_idx]
        i, j, k, s_idx, d = declare('int', 5)
        alp, bet, gam, phi, psi = declare('int', 5)
        xij = declare('matrix(3)')
        dwij = declare('matrix(3)')
        d = self.dim
        tol = 1.0e-10

        m0 = 0.0
        m1 = declare('matrix(3)')
        m2 = declare('matrix((3, 3))')
        grad_m0 = declare('matrix(3)')
        grad_m1 = declare('matrix((3, 3))')
        grad_m2 = declare('matrix((3, 3, 3))')
        ai = 0.0
        bi = declare('matrix(3)')
        grad_ai = declare('matrix(3)')
        grad_bi = declare('matrix((3, 3))')

        for i in range(3):
            m1[i] = 0.0
            grad_m0[i] = 0.0
            bi[i] = 0.0
            grad_ai[i] = 0.0
            for j in range(3):
                m2[i][j] = 0.0
                grad_m1[i][j] = 0.0
                grad_bi[i][j] = 0.0
                for k in range(3):
                    grad_m2[i][j][k] = 0.0

        for n in range(N_NBRS):
            s_idx = NBRS[n]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            hij = (h + s_h[s_idx]) * 0.5
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            wij = KERNEL.kernel(xij, rij, hij)
            KERNEL.gradient(xij, rij, hij, dwij)
            V = s_m[s_idx] / s_rho[s_idx]

            m0 += V * wij
            for alp in range(d):
                m1[alp] += V * wij * xij[alp]
                for bet in range(d):
                    m2[alp][bet] += V * wij * xij[alp] * xij[bet]
            for gam in range(d):
                grad_m0[gam] += V * dwij[gam]
                for alp in range(d):
                    fac = 1.0 if alp == gam else 0.0
                    temp = (xij[alp] * dwij[gam] + fac * wij)
                    grad_m1[gam][alp] += V * temp
                    for bet in range(d):
                        fac2 = 1.0 if bet == gam else 0.0
                        temp = xij[alp] * fac2 + xij[bet] * fac
                        temp2 = (xij[alp] * xij[bet] * dwij[gam] + temp * wij)
                        grad_m2[gam][alp][bet] += V * temp2

        ai = m0
        for alp in range(d):
            for bet in range(d):
                check = m2[alp][bet]
                if abs(check) >= tol:
                    ai -= m1[alp] * m1[bet] / check
                    bi[alp] -= m1[bet] / check
        if abs(ai) >= tol:
            ai = 1.0 / ai
            condition = True
        else:
            ai = 1.0
            condition = False
        for gam in range(d):
            grad_ai[gam] -= grad_m0[gam]
            for alp in range(d):
                for bet in range(d):
                    check = m2[alp][bet]
                    if abs(check) >= tol:
                        temp1 = grad_m1[gam][alp] * m1[bet]
                        temp2 = grad_m1[gam][bet] * m1[alp]
                        grad_ai[gam] += (temp1 + temp2) / check
                        temp = grad_m1[gam][bet] / check
                        grad_bi[gam][alp] -= temp
                    for phi in range(d):
                        for psi in range(d):
                            check = m2[alp][phi] * m2[psi][bet]
                            if abs(check) >= tol:
                                temp = grad_m2[gam][phi][psi]
                                temp = temp / check
                                grad_ai[gam] -= m1[bet] * m1[alp] * temp
                                grad_bi[gam][alp] += m1[bet] * temp
            grad_ai[gam] = grad_ai[gam] * ai * ai

        if N_NBRS > 0 or condition:
            d_ai[d_idx] = ai
            for i in range(d):
                d_gradai[d * d_idx + i] = grad_ai[i]
                d_bi[d * d_idx + i] = bi[i]
                for j in range(d):
                    d_gradbi[d * d_idx + d * i + j] = grad_bi[i][j]
        else:
            d_ai[d_idx] = 1.0
            for i in range(d):
                d_gradai[d * d_idx + i] = 0.0
                d_bi[d * d_idx + i] = 0.0
                for j in range(d):
                    d_gradbi[d * d_idx + d * i + j] = 0.0


class CRKSPH(Equation):

    def __init__(self, dest, sources, dim=2, tol=0.5):
        self.dim = dim
        self.tol = tol
        super(CRKSPH, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_ai, d_gradai, d_bi, d_gradbi, s_ai,
             s_gradai, s_bi, s_gradbi, WIJ, DWIJ, XIJ, HIJ):
        alp, gam, d = declare('int', 3)
        res = declare('matrix(3)')
        d = self.dim
        ai = d_ai[d_idx]
        aj = s_ai[s_idx]
        eps = 1.0e-04 * HIJ
        for gam in range(d):
            res[gam] = ai * DWIJ[gam] + d_gradai[d * d_idx + gam] * WIJ
            res[gam] += ai * d_bi[d * d_idx + gam] * WIJ
            res[gam] += aj * DWIJ[gam] - s_gradai[d * s_idx + gam] * WIJ
            res[gam] -= aj * s_bi[d * s_idx + gam] * WIJ
            for alp in range(d):
                res[gam] += ai * d_bi[d * d_idx + alp] * XIJ[alp] * DWIJ[gam]
                res[gam] -= aj * s_bi[d * s_idx + alp] * XIJ[alp] * DWIJ[gam]

                temp = d_gradai[d * d_idx + gam] * d_bi[d * d_idx + alp]
                res[gam] += temp * XIJ[alp] * WIJ
                temp = s_gradai[d * s_idx + gam] * s_bi[d * s_idx + alp]
                res[gam] += temp * XIJ[alp] * WIJ

                temp2 = ai * d_gradbi[d * d * d_idx + d * gam + alp]
                res[gam] += temp2 * XIJ[alp] * WIJ
                temp2 = aj * s_gradbi[d * d * s_idx + d * gam + alp]
                res[gam] += temp2 * XIJ[alp] * WIJ
            res[gam] *= 0.5

        change = 0.0
        for i in range(d):
            change += abs(DWIJ[i] - res[i]) / (abs(DWIJ[i]) + eps)
        if change <= self.tol:
            for i in range(d):
                DWIJ[i] = res[i]
