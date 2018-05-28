from pysph.sph.equation import Equation


class Continuity(Equation):

    def __init__(self, dest, sources, c0):
        self.c0 = c0

        super(Continuity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_u, d_v, d_w, s_u, s_v, s_w, d_cs, s_cs,
             d_rho, d_arho, s_rho, d_p, s_p, DWIJ, RIJ, XIJ):
        rl = d_rho[d_idx]
        rr = s_rho[s_idx]
        pl = d_p[d_idx]
        pr = s_p[s_idx]
        cl = d_cs[d_idx]
        cr = s_cs[s_idx]
        uxl = d_u[d_idx]
        uyl = d_v[d_idx]
        uzl = d_w[d_idx]
        uxr = s_u[s_idx]
        uyr = s_v[s_idx]
        uzr = s_w[s_idx]
        co = self.c0
        eij = declare('matrix(3)')
        vij = declare('matrix(3)')
        v_star = declare('matrix(3)')
        vij[0] = 0.5 * (uxl + uxr)
        vij[1] = 0.5 * (uyl + uyr)
        vij[2] = 0.5 * (uzl + uzr)

        for i in range(3):
            if RIJ >= 1.0e-12:
                eij[i] = -XIJ[i] / RIJ
            else:
                eij[i] = 0.0
        ul = uxl * eij[0] + uyl * eij[1] + uzl * eij[2]
        ur = uxr * eij[0] + uyr * eij[1] + uzr * eij[2]
        rhobar = 0.5 * (rl + rr)
        u_star = 0.5 * (ul + ur) + 0.5 * (pl - pr) / (rhobar * co)
        p_star = 0.5 * (pl + pr) + 0.5 * rhobar * co * (ul - ur)
        for i in range(3):
            v_star[i] = (u_star - 0.5 * (ul + ur)) * eij[i] + vij[i]
        vdotw = (uxl - v_star[0]) * DWIJ[0] + (uyl - v_star[1]) * DWIJ[1]
        vdotw += (uzl - v_star[2]) * DWIJ[2]
        d_arho[d_idx] += 2.0 * s_m[s_idx] * vdotw * rl / rr


class MomentumFluid(Equation):

    def __init__(self, dest, sources, c0, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.c0 = c0

        super(MomentumFluid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_idx, s_idx, s_m, d_u, d_v, d_w, s_u, s_v, s_w, d_cs, s_cs,
             d_rho, s_rho, d_p, s_p, d_au, d_av, d_aw, RIJ, XIJ, DWIJ, HIJ):
        rl = d_rho[d_idx]
        rr = s_rho[s_idx]
        pl = d_p[d_idx]
        pr = s_p[s_idx]
        cl = d_cs[d_idx]
        cr = s_cs[s_idx]
        uxl = d_u[d_idx]
        uyl = d_v[d_idx]
        uzl = d_w[d_idx]
        uxr = s_u[s_idx]
        uyr = s_v[s_idx]
        uzr = s_w[s_idx]
        m = s_m[s_idx]
        co = self.c0
        eij = declare('matrix(3)')
        vij = declare('matrix(3)')
        v_star = declare('matrix(3)')
        vij[0] = 0.5 * (uxl + uxr)
        vij[1] = 0.5 * (uyl + uyr)
        vij[2] = 0.5 * (uzl + uzr)

        for i in range(3):
            if RIJ >= 1.0e-12:
                eij[i] = -XIJ[i] / RIJ
            else:
                eij[i] = 0.0
        ul = uxl * eij[0] + uyl * eij[1] + uzl * eij[2]
        ur = uxr * eij[0] + uyr * eij[1] + uzr * eij[2]
        rhobar = 0.5 * (rl + rr)
        p_star = 0.5 * (pl + pr) + 0.5 * rhobar * co * (ul - ur)
        factor = -2.0 * m * p_star / (rl * rr)

        d_au[d_idx] += factor * DWIJ[0]
        d_av[d_idx] += factor * DWIJ[1]
        d_aw[d_idx] += factor * DWIJ[2]
