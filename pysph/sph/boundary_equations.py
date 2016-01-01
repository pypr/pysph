"""
SPH Boundary Equations
######################
"""

from pysph.sph.equation import Equation

def wendland_quintic(rij=1.0, h=1.0):
    q = rij/h
    q1 = 2.0 - q
    val = 0.0
    if q < 2.0:
        val = (1 + 2.5*q + 2*q*q)*q1*q1*q1*q1*q1

    return val


class MonaghanBoundaryForce(Equation):
    def __init__(self, dest, sources, deltap):
        self.deltap = deltap
        super(MonaghanBoundaryForce,self).__init__(dest,sources)

    def loop(self, d_idx, s_idx, s_m, s_rho, d_m, d_cs, s_cs, d_h,
             s_tx, s_ty, s_tz, s_nx, s_ny, s_nz,
             d_au, d_av, d_aw, XIJ):

        norm = declare('matrix((3,))')
        tang = declare('matrix((3,))')

        ma = d_m[d_idx]
        mb = s_m[s_idx]

        # particle sound speed
        cs = d_cs[d_idx]

        # boundary normals
        norm[0] = s_nx[s_idx]
        norm[1] = s_ny[s_idx]
        norm[2] = s_nz[s_idx]

        # boundary tangents
        tang[0] = s_tx[s_idx]
        tang[1] = s_ty[s_idx]
        tang[2] = s_tz[s_idx]

        # x and y projections
        x = XIJ[0]*tang[0] + XIJ[1]*tang[1] + XIJ[2]*tang[2]
        y = XIJ[0]*norm[0] + XIJ[1]*norm[1] + XIJ[2]*norm[2]

        # compute the force
        force = 0.0
        q = y/d_h[d_idx]

        xabs = fabs(x)

        if 0 <= xabs <= self.deltap:
            beta = 0.02 * cs * cs/y
            tforce = 1.0 - xabs/self.deltap

            if 0 < q <= 2.0/3.0:
                nforce =  2.0/3.0

            elif 2.0/3.0 < q <= 1.0:
                nforce = 2*q*(1.0 - 0.75*q)

            elif 1.0 < q <= 2.0:
                nforce = 0.5 * (2-q)*(2-q)

            else:
                nforce = 0.0

            force = (mb/(ma+mb)) * nforce * tforce * beta
        else:
            force = 0.0

        # boundary force accelerations
        d_au[d_idx] += force * norm[0]
        d_av[d_idx] += force * norm[1]
        d_aw[d_idx] += force * norm[2]

class MonaghanKajtarBoundaryForce(Equation):
    def __init__(self, dest, sources, K=None, beta=None, h=None):
        self.K = K
        self.beta = beta
        self.h = h

        if None in [K, beta, h]:
            raise ValueError("Invalid parameter values")

        super(MonaghanKajtarBoundaryForce,self).__init__(dest,sources)

    def _get_helpers_(self):
        return [wendland_quintic]

    def loop(self, d_idx, s_idx, d_m, s_m, d_au, d_av, d_aw, RIJ, R2IJ, XIJ):

        ma = d_m[d_idx]
        mb = s_m[s_idx]

        w = wendland_quintic(RIJ, self.h)
        force = self.K/self.beta * w/R2IJ * 2*mb/(ma + mb)

        d_au[d_idx] += force * XIJ[0]
        d_av[d_idx] += force * XIJ[1]
        d_aw[d_idx] += force * XIJ[2]
