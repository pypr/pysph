"""Boundary equations for Gas-dynamics"""

from pysph.sph.equation import Equation

class WallBoundary(Equation):
    def initialize(self, d_idx, d_p,  d_rho, d_e, d_m, d_cs, d_div, d_h,
            d_htmp, d_h0, d_u, d_v, d_w,  d_wij):
        d_p[d_idx] = 0.0
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0
        d_m[d_idx] = 0.0
        d_rho[d_idx] = 0.0
        d_e[d_idx] = 0.0
        d_cs[d_idx] = 0.0
        d_div[d_idx] = 0.0
        d_wij[d_idx] = 0.0
        d_h[d_idx]  = d_h0[d_idx]
        d_htmp[d_idx]  = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_e, d_m, d_cs, d_div, d_h, d_u,
            d_v, d_w, d_wij, d_htmp, s_p, s_rho, s_e, s_m, s_cs, s_h, s_div,
            s_u, s_v, s_w,  WI):
        d_wij[d_idx] += WI
        d_p[d_idx] += s_p[s_idx]*WI
        d_u[d_idx] -= s_u[s_idx]*WI
        d_v[d_idx] -= s_v[s_idx]*WI
        d_w[d_idx] -= s_w[s_idx]*WI
        d_m[d_idx] += s_m[s_idx]*WI
        d_rho[d_idx] +=  s_rho[s_idx]*WI
        d_e[d_idx] += s_e[s_idx]*WI
        d_cs[d_idx] += s_cs[s_idx]*WI
        d_div[d_idx] += s_div[s_idx]*WI
        d_htmp[d_idx]  += s_h[s_idx]*WI

    def post_loop(self, d_idx, d_p, d_rho, d_e, d_m, d_cs, d_div, d_h, d_u,
             d_v, d_w, d_wij, d_htmp):
        if (d_wij[d_idx]>1e-30):
            d_p[d_idx] = d_p[d_idx]/d_wij[d_idx]
            d_u[d_idx] = d_u[d_idx]/d_wij[d_idx]
            d_v[d_idx] = d_v[d_idx]/d_wij[d_idx]
            d_w[d_idx] = d_w[d_idx]/d_wij[d_idx]
            d_m[d_idx] = d_m[d_idx]/d_wij[d_idx]
            d_rho[d_idx] = d_rho[d_idx]/d_wij[d_idx]
            d_e[d_idx] =  d_e[d_idx]/d_wij[d_idx]
            d_cs[d_idx] =  d_cs[d_idx]/d_wij[d_idx]
            d_div[d_idx] =  d_div[d_idx]/d_wij[d_idx]
            d_h[d_idx] =  d_htmp[d_idx] /d_wij[d_idx]
