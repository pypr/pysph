"""Functions for the transport velocity formulation of Adami et. al."""

from textwrap import dedent
from equations import Equation, CodeBlock

class DensitySummation(Equation):
    def setup(self):
        code = dedent('''

        d_V[d_idx] += WIJ
        d_rho[d_idx] += d_m[d_idx]*WIJ

        ''')
        self.loop = CodeBlock(code=code)

class VolumeSummation(Equation):
    def setup(self):
        code = dedent('''
        d_V[d_idx] += WIJ
        ''')
        self.loop = CodeBlock(code=code)

class ContinuityEquation(Equation):
    def setup(self):
        code = dedent('''
    Vj = 1./s_V[s_idx]
    vijdotdwij = VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] + VIJ[2] * DWIJ[2]
    d_arho[d_idx] += d_m[d_idx] * Vj * vijdotdwij
    ''')
        self.loop = CodeBlock(code=code, Vj=0., vijdotdwij=0.)
    
class StateEquation(Equation):
    def __init__(self, dest, sources, b=1.0):
        self.b=b
        super(StateEquation, self).__init__(dest, sources)

    def setup(self):
        code = dedent("""
        # update the pressure using the reference density
        d_p[d_idx] = d_p0[d_idx] * ( d_rho[d_idx]/d_rho0[d_idx] - b )
        """)

        self.loop = CodeBlock(code=code, b=self.b)

class SolidWallBC(Equation):
    def __init__(self, dest, sources, rho0=1.0, p0=100.0,
                 gx=0.0, gy=0.0, gz=0.0, ax=0.0, ay=0.0, az=0.0,
                 b=1.0):
        self.rho0 = rho0
        self.p0 = p0
        self.b=b
        self.gx = gx; self.ax = ax 
        self.gy = gy; self.ay = ay
        self.gz = gz; self.az = az
        super(SolidWallBC, self).__init__(dest, sources)

    def setup(self):
        code = dedent('''

        # smooth velocities at the ghost points
        d_u[d_idx] += s_u[s_idx]*WIJ
        d_v[d_idx] += s_v[s_idx]*WIJ

        # smooth pressure
        gdotxij = (gx-ax)*XIJ[0] + (gy-ay)*XIJ[1] + (gz-az)*XIJ[2]
        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx] * gdotxij * WIJ

        # denominator
        d_wij[d_idx] += WIJ

        ''')
        self.loop = CodeBlock(code=code, gx=self.gx, gy=self.gy, gz=self.gz,
                              ax=self.ax, ay=self.ay, az=self.az, gdotxij=0.0)

        code = dedent("""

        if d_wij[d_idx] > 1e-14:
            d_u[d_idx] /= d_wij[d_idx]
            d_v[d_idx] /= d_wij[d_idx]

            # solid wall condition
            d_u[d_idx] = 2*d_u0[d_idx] - d_u[d_idx]
            d_v[d_idx] = 2*d_v0[d_idx] - d_v[d_idx]

            # smooth interpolated pressure at the wall
            d_p[d_idx] /= d_wij[d_idx]

        # update the density from the pressure
        d_rho[d_idx] = s_rho0[0] * (d_p[d_idx]/s_p0[0] + b)

        """)
                      
        self.post_loop = CodeBlock(code=code, b=self.b)

class MomentumEquation(Equation):
    def __init__(self, dest, sources, pb=0.0, nu=0.01, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
                    
        self.nu = nu
        self.pb = pb
        super(MomentumEquation, self).__init__(dest, sources)

    def setup(self):
        code = dedent("""

        # averaged pressure
        rhoi = d_rho[d_idx]; rhoj = s_rho[s_idx]
        pi = d_p[d_idx]; pj = s_p[s_idx]

        pij = rhoj * pi + rhoi * pj
        pij /= (rhoj + rhoi)

        ui = d_u[d_idx]; uhati = d_uhat[d_idx]
        vi = d_v[d_idx]; vhati = d_vhat[d_idx]

        uj = s_u[s_idx]; uhatj = s_uhat[s_idx]
        vj = s_v[s_idx]; vhatj = s_vhat[s_idx]

        # averaged shear viscosity
        etai = nu * rhoi
        etaj = nu * rhoj

        etaij = 2 * (etai * etaj)/(etai + etaj)

        # scalar part of the kernel gradient
        Fij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        
        # particle volumes
        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # accelerations
        d_au[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * (-pij*DWIJ[0] + etaij*Fij/(R2IJ + 0.01 * HIJ * HIJ)*VIJ[0])
        d_av[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * (-pij*DWIJ[1] + etaij*Fij/(R2IJ + 0.01 * HIJ * HIJ)*VIJ[1])

        # contribution due to the background pressure
        d_auhat[d_idx] += -d_p0[d_idx]/d_m[d_idx] * (Vi2 + Vj2) * DWIJ[0]
        d_avhat[d_idx] += -d_p0[d_idx]/d_m[d_idx] * (Vi2 + Vj2) * DWIJ[1]

        """)

        self.loop = CodeBlock(code=code, pb=self.pb, nu=self.nu, rhoi=0.0, rhoj=0., 
                              pi=0., pj=0., pij=0., ui=0., uj=0., vi=0., vj=0., 
                              etai=0., etaj=0., etaij=0., uhati=0., uhatj=0., 
                              vhati=0., vhatj=0., Fij=0., Vi=0., Vj=0., Vi2=0., 
                              Vj2=0.)

class ArtificialStress(Equation):
    def setup(self):
        code = dedent("""

        # averaged pressure
        rhoi = d_rho[d_idx]; rhoj = s_rho[s_idx]

        ui = d_u[d_idx]; uhati = d_uhat[d_idx]
        vi = d_v[d_idx]; vhati = d_vhat[d_idx]

        uj = s_u[s_idx]; uhatj = s_uhat[s_idx]
        vj = s_v[s_idx]; vhatj = s_vhat[s_idx]

        Vi = 1./d_V[d_idx]; Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi; Vj2 = Vj * Vj

        # artificial stress tensor
        Axxi = rhoi*ui*(uhati - ui); Axyi = rhoi*ui*(vhati - vi)
        Ayxi = rhoi*vi*(uhati - ui); Ayyi = rhoi*vi*(vhati - vi)

        Axxj = rhoj*uj*(uhatj - uj); Axyj = rhoj*uj*(vhatj - vj)
        Ayxj = rhoj*vj*(uhatj - uj); Ayyj = rhoj*vj*(vhatj - vj)

        Ax = 0.5 * (Axxi + Axxj) * DWIJ[0] + 0.5 * (Axyi + Axyj) * DWIJ[1]
        Ay = 0.5 * (Ayxi + Ayxj) * DWIJ[0] + 0.5 * (Ayyi + Ayyj) * DWIJ[1]

        # accelerations
        d_au[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * Ax
        d_av[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * Ay

        """)

        self.loop = CodeBlock(code=code, rhoi=0.0, rhoj=0., pi=0.0, pj=0.0, pij=0.0,
                              ui=0., uj=0., vi=0., vj=0.,
                               uhati=0., uhatj=0., vhati=0., vhatj=0., 
                              Vi=0., Vj=0., Vi2=0., Vj2=0., Ax=0., Ay=0., 
                              Axxi=0., Axyi=0., Ayxi=0., Ayyi=0.,
                              Axxj=0., Axyj=0., Ayxj=0., Ayyj=0.)

