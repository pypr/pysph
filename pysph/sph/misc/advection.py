"""Functions for advection"""

from textwrap import dedent
from pysph.sph.equation import Equation
from numpy import cos, pi

class Advect(Equation):
    def loop(self, d_idx, d_ax, d_ay, d_u, d_v):
        d_ax[d_idx] = d_u[d_idx]
        d_ay[d_idx] = d_v[d_idx]

class MixingVelocityUpdate(Equation):
    def __init__(self, dest, source=None, T=1.0, symmetric=False):
        self.T = T
        super(MixingVelocityUpdate, self).__init__(dest, source, symmetric=symmetric)

    def loop(self, d_idx, d_u, d_v, d_u0, d_v0, t=0.1):
        d_u[d_idx] = cos(pi*t/self.T) * d_u0[d_idx]
        d_v[d_idx] = -cos(pi*t/self.T) * d_v0[d_idx]

    def cython_code(self):
        code = dedent("""
        from libc.math cimport sin, cos
        from libc.math cimport M_PI as pi
        """)
        return dict(helper=code)
