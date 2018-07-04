from pysph.sph.equation import Equation


class Continuity(Equation):

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

        if RIJ >= 1.0e-16:
            ul = -(uxl * XIJ[0] + uyl * XIJ[1] + uzl * XIJ[2]) / RIJ
            ur = -(uxr * XIJ[0] + uyr * XIJ[1] + uzr * XIJ[2]) / RIJ
        else:
            ul = 0.0
            ur = 0.0
        u_star = (ul * rl * cl + ur * rr * cr + pl - pr) / (rl * cl + rr * cr)
        dwdr = sqrt(DWIJ[0] * DWIJ[0] + DWIJ[1] * DWIJ[1] + DWIJ[2] * DWIJ[2])

        d_arho[d_idx] += 2.0 * s_m[s_idx] * dwdr * (ul - u_star) * rl / rr


class Momentum(Equation):

    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super(Momentum, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_idx, s_idx, s_m, d_u, d_v, d_w, s_u, s_v, s_w, d_cs, s_cs,
             d_rho, s_rho, d_p, s_p, d_au, d_av, d_aw, RIJ, XIJ, DWIJ):
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

        if RIJ >= 1.0e-16:
            ul = -(uxl * XIJ[0] + uyl * XIJ[1] + uzl * XIJ[2]) / RIJ
            ur = -(uxr * XIJ[0] + uyr * XIJ[1] + uzr * XIJ[2]) / RIJ
        else:
            ul = 0.0
            ur = 0.0
        p_star = pl * rr * cr + pr * cl * rl - rl * rr * cl * cr * (ur - ul)
        p_star /= (rl * cl + rr * cr)
        factor = -2.0 * m * p_star / (rl * rr)

        d_au[d_idx] += factor * DWIJ[0]
        d_av[d_idx] += factor * DWIJ[1]
        d_aw[d_idx] += factor * DWIJ[2]
