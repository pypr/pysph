"""Functions for advection"""

from textwrap import dedent
from equations import Equation, CodeBlock

class Advect(Equation):
    def setup(self):
        code = dedent("""
        d_ax[d_idx] = d_u[d_idx]
        d_ay[d_idx] = d_v[d_idx]
        """)

        self.loop = CodeBlock(code=code)

class MixingVelocityUpdate(Equation):
    def __init__(self, dest, sources, T=1.0):
        self.T = T
        super(MixingVelocityUpdate, self).__init__(dest, sources)

    def setup(self):
        code = dedent("""
        xi = d_x[d_idx]
        yi = d_y[d_idx]

        d_u[d_idx] = cos(pi*t/T) * d_u0[d_idx]
        d_v[d_idx] = -cos(pi*t/T) * d_v0[d_idx]

        """)

        self.loop = CodeBlock(code=code, T=self.T, xi=0.0, yi=0.0)
    
    def cython_code(self):
        code = dedent("""
        from libc.math cimport sin, cos
        from libc.math cimport M_PI as pi
        """)
        return dict(helper=code)
    
