"""Functions for the transport velocity formulation of Adami et. al."""

from textwrap import dedent
from equations import Equation, CodeBlock

class TransportVelocitySummationDensity(Equation):
    def __init__(self, dest, sources, rho0=1.0, c0=10):
        self.rho0 = rho0
        self.c0 = c0
        super(TransportVelocitySummationDensity, self).__init__(dest, sources)

    def setup(self):
        code = dedent('''

        d_V[d_idx] += WIJ
        d_rho[d_idx] += d_m[d_idx]*WIJ

        ''')
        self.loop = CodeBlock(code=code)

        code = dedent("""
        # update the pressure using the equation of state
        d_p[d_idx] = d_p0[d_idx] * ( d_rho[d_idx]/d_rho0[d_idx] - 1 )

        """)

        self.post_loop = CodeBlock(code=code, rho0=self.rho0, c0=self.c0)

class TransportVelocitySolidWall(Equation):
    def __init__(self, dest, sources, rho0=1.0, p0=100.0,
                 gx=0.0, gy=0.0, gz=0.0, ax=0.0, ay=0.0, az=0.0):
        self.rho0 = rho0
        self.p0 = p0
        self.gx = gx; self.ax = ax 
        self.gy = gy; self.ay = ay
        self.gz = gz; self.az = az
        super(TransportVelocitySolidWall, self).__init__(dest, sources)

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
       
        # smooth velocity at the wall particle
        if d_wij[d_idx] < 1e-14:
            pass

        else:
            d_u[d_idx] /= d_wij[d_idx]
            d_v[d_idx] /= d_wij[d_idx]

            # solid wall condition
            d_u[d_idx] = 2*d_u0[d_idx] - d_u[d_idx]
            d_v[d_idx] = 2*d_v0[d_idx] - d_v[d_idx]
         
            # smooth interpolated pressure at the wall
            d_p[d_idx] /= d_wij[d_idx]

        # update the density from the pressure
        d_rho[d_idx] = s_rho0[0] * (d_p[d_idx]/s_p[0] + 1.0)

        """)
                      
        self.post_loop = CodeBlock(code=code,rho0=self.rho0,p0=self.p0)

class TransportVelocityMomentumEquation(Equation):
    def __init__(self, dest, sources, pb, nu=0.01, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
                    
        self.nu = nu
        self.pb = pb
        super(TransportVelocityMomentumEquation, self).__init__(dest, sources)

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

        # artificial stress tensor
        Ax = 0.5 * (rhoi*ui*(uhatj - uj)*DWIJ[0] + rhoi*ui*(vhatj - vj)*DWIJ[1] + rhoj*uj*(uhati - ui)*DWIJ[0] + rhoj*uj*(vhati - vi)*DWIJ[1])   
        Ay = 0.5 * (rhoi*vi*(uhatj - uj)*DWIJ[0] + rhoi*vi*(vhatj - vj)*DWIJ[1] + rhoj*vj*(uhati - ui)*DWIJ[0] + rhoj*vj*(vhati - vi)*DWIJ[1])   

        # accelerations
        d_au[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * (-pij*DWIJ[0] + Ax + etaij*Fij/(R2IJ + 0.01 * HIJ * HIJ)*VIJ[0])

        d_av[d_idx] += 1.0/d_m[d_idx] * (Vi2 + Vj2) * (-pij*DWIJ[1] + Ay + etaij*Fij/(R2IJ + 0.01 * HIJ * HIJ)*VIJ[1])

        # contribution due to the background pressure
        d_auhat[d_idx] += -pb/d_m[d_idx] * (Vi2 + Vj2) * DWIJ[0]
        d_avhat[d_idx] += -pb/d_m[d_idx] * (Vi2 + Vj2) * DWIJ[1]

        """)

        self.loop = CodeBlock(code=code, pb=self.pb, nu=self.nu, rhoi=0.0, rhoj=0., 
                              pi=0., pj=0., pij=0., ui=0., uj=0., vi=0., vj=0., 
                              etai=0., etaj=0., etaij=0., uhati=0., uhatj=0., 
                              vhati=0., vhatj=0., Fij=0., Vi=0., Vj=0., Vi2=0., 
                              Vj2=0., Ax=0., Ay=0.)
