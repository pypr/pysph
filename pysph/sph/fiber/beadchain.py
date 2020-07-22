"""Equations for fibers based on bead chain model.

Reference
---------

    .. [Meyer2020] N. Meyer et. al "Parameter Identification of Fiber Orientation
    Models Based on Direct Fiber Simulation with Smoothed Particle Hydrodynamics",
    Journal of Composites Science, 2020, 4, 77; doi:10.3390/jcs4020077

"""

from math import acos, pi, sin, sqrt

from pysph.sph.equation import Equation


class EBGVelocityReset(Equation):
    """Reset EBG velocities.

    This is only necessary, if subcycling is used.
    """

    def loop(self, d_idx, d_eu, d_ev, d_ew, d_ex, d_ey, d_ez,
             d_u, d_v, d_w, dt):
        d_eu[d_idx] = 0
        d_ev[d_idx] = 0
        d_ew[d_idx] = 0

        d_u[d_idx] += d_ex[d_idx]/dt
        d_v[d_idx] += d_ey[d_idx]/dt
        d_w[d_idx] += d_ez[d_idx]/dt

        d_ex[d_idx] = 0
        d_ey[d_idx] = 0
        d_ez[d_idx] = 0


class Tension(Equation):
    """Linear elastic fiber tension.

    Particle acceleration based on fiber tension is computed. The source
    must be chosen to be the same as the destination particles.
    See eq. (16) in [Meyer2020].
    """

    def __init__(self, dest, sources, ea):
        r"""
        Parameters
        ----------
        ea : float
            rod stiffness (elastic modulus x section area)
        """
        self.ea = ea
        super(Tension, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_lprev, d_lnext, s_fractag, d_fractag,
             d_au, d_av, d_aw, d_rxnext, d_rynext, d_rznext, d_rnext, d_rxprev,
             d_ryprev, d_rzprev, d_rprev):

        # previous particle
        if d_rprev[d_idx] > 1E-14:
            t = self.ea*(d_rprev[d_idx]/d_lprev[d_idx]-1)
            d_au[d_idx] += (t*d_rxprev[d_idx]/d_rprev[d_idx])/d_m[d_idx]
            d_av[d_idx] += (t*d_ryprev[d_idx]/d_rprev[d_idx])/d_m[d_idx]
            d_aw[d_idx] += (t*d_rzprev[d_idx]/d_rprev[d_idx])/d_m[d_idx]

        # next particle
        if d_rnext[d_idx] > 1E-14:
            t = self.ea*(d_rnext[d_idx]/d_lnext[d_idx]-1)
            d_au[d_idx] += (t*d_rxnext[d_idx]/d_rnext[d_idx])/d_m[d_idx]
            d_av[d_idx] += (t*d_rynext[d_idx]/d_rnext[d_idx])/d_m[d_idx]
            d_aw[d_idx] += (t*d_rznext[d_idx]/d_rnext[d_idx])/d_m[d_idx]


class ArtificialDamping(Equation):
    """Damp EBG particle motion.

    EBG Particles are damped based on EBG velocity. Use this in combination
    with EBGStep and EBGVelocityReset, if subcycles are applied.
    """

    def __init__(self, dest, sources, d):
        r"""
        Parameters
        ----------
        d : float
            damping coefficient
        """
        self.d = d
        super(ArtificialDamping, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_eu, d_ev, d_ew, d_au, d_av, d_aw):
        d_au[d_idx] -= 2*self.d*d_eu[d_idx]/d_m[d_idx]
        d_av[d_idx] -= 2*self.d*d_ev[d_idx]/d_m[d_idx]
        d_aw[d_idx] -= 2*self.d*d_ew[d_idx]/d_m[d_idx]


class Bending(Equation):
    r"""Linear elastic fiber bending

    Particle acceleration based on fiber bending is computed. The source
    particles must be chosen to be the same as the destination particles.
    See eq. (17) in [Meyer2020].
    """

    def __init__(self, dest, sources, ei):
        """
        Parameters
        ----------
        ei : float
            bending stiffness (elastic modulus x 2nd order moment)
        """
        self.ei = ei
        super(Bending, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_phi0, d_fractag, d_phifrac,
             d_rxnext, d_rynext, d_rznext, d_rnext,
             d_rxprev, d_ryprev, d_rzprev, d_rprev,
             d_au, d_av, d_aw):
        if d_rnext[d_idx] > 1E-14 and d_rprev[d_idx] > 1E-14:
            # vector to previous particle
            xab = d_rxprev[d_idx]
            yab = d_ryprev[d_idx]
            zab = d_rzprev[d_idx]
            rab = d_rprev[d_idx]
            # vector to next particle
            xbc = d_rxnext[d_idx]
            ybc = d_rynext[d_idx]
            zbc = d_rznext[d_idx]
            rbc = d_rnext[d_idx]

            # normed dot product between vectors
            # (limited to catch round off errors)
            dot_prod_norm = (xab*xbc+yab*ybc+zab*zbc)/(rab*rbc)
            dot_prod_norm = max(-1, dot_prod_norm)
            dot_prod_norm = min(1, dot_prod_norm)
            # angle between vectors
            phi = acos(dot_prod_norm)
            # direction of angle from cross product
            norm = rab*rbc*sin(phi)
            nx = (yab*zbc-zab*ybc)/norm
            ny = (zab*xbc-xab*zbc)/norm
            nz = (xab*ybc-yab*xbc)/norm

            # momentum
            Mx = 2*self.ei*(phi-d_phi0[d_idx])/(rab+rbc)*nx
            My = 2*self.ei*(phi-d_phi0[d_idx])/(rab+rbc)*ny
            Mz = 2*self.ei*(phi-d_phi0[d_idx])/(rab+rbc)*nz

            if abs(phi-d_phi0[d_idx]) > d_phifrac[d_idx]:
                d_fractag[d_idx] = 1

            # forces on neighbouring particles
            Fabx = (My*zab-Mz*yab)/(rab**2)
            Faby = (Mz*xab-Mx*zab)/(rab**2)
            Fabz = (Mx*yab-My*xab)/(rab**2)
            Fbcx = (My*zbc-Mz*ybc)/(rbc**2)
            Fbcy = (Mz*xbc-Mx*zbc)/(rbc**2)
            Fbcz = (Mx*ybc-My*xbc)/(rbc**2)

            d_au[d_idx] -= (Fabx-Fbcx)/d_m[d_idx]
            d_av[d_idx] -= (Faby-Fbcy)/d_m[d_idx]
            d_aw[d_idx] -= (Fabz-Fbcz)/d_m[d_idx]
            d_au[d_idx+1] -= Fbcx/d_m[d_idx+1]
            d_av[d_idx+1] -= Fbcy/d_m[d_idx+1]
            d_aw[d_idx+1] -= Fbcz/d_m[d_idx+1]
            d_au[d_idx-1] += Fabx/d_m[d_idx-1]
            d_av[d_idx-1] += Faby/d_m[d_idx-1]
            d_aw[d_idx-1] += Fabz/d_m[d_idx-1]


class Friction(Equation):
    """Fiber bending due to friction on fictive surfaces

    Since the fiber represented by a beadchain of particles has no thickness, a
    term has do compensate fritction due to shear on the particles surface.
    The source particles must be chosen to be the same as the destination
    particles.
    See Appendix A in [Meyer2020].
    """

    def __init__(self, dest, sources, J, dx, mu, d):
        r"""
        Parameters
        ----------
        J : float
            moment of inertia
        dx : float
            length of segment
        mu : float
            absolute viscosity
        d : float
            fiber diameter
        ar : float
            fiber aspect ratio
        """
        self.J = J
        self.dx = dx
        self.mu = mu
        self.d = d
        super(Friction, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_rho, d_rxnext, d_rynext, d_rznext,
             d_rnext, d_rxprev, d_ryprev, d_rzprev, d_rprev, d_fractag,
             d_au, d_av, d_aw, d_dudx, d_dudy, d_dudz, d_dvdx,
             d_dvdy, d_dvdz, d_dwdx, d_dwdy, d_dwdz):
        if (d_rnext[d_idx] > 1E-14 and d_rprev[d_idx] > 1E-14):

            dx = d_rxprev[d_idx]-d_rxnext[d_idx]
            dy = d_ryprev[d_idx]-d_rynext[d_idx]
            dz = d_rzprev[d_idx]-d_rznext[d_idx]
            r = sqrt(dx**2+dy**2+dz**2)
            s1 = dx/r
            s2 = dy/r
            s3 = dz/r

            # ensuring that [sx sy sz] is not parallel to [1 0 0]
            if abs(s2) > 1E-14 or abs(s3) > 1E-14:
                fac = (3.*self.dx*self.d**2/4.*self.mu*pi)/(s2**2+s3**2)

                Mx = fac*((s1*s2**2*s3+s1*s3**3)*d_dvdx[d_idx]
                          + (-s1**2*s2*s3+s2*s3)*d_dvdy[d_idx]
                          + (-s1**2*s3**2-s2**2)*d_dvdz[d_idx]
                          + (-s1*s2**3-s1*s2*s3**2)*d_dwdx[d_idx]
                          + (s1**2*s2**2+s3**2)*d_dwdy[d_idx]
                          + (s1**2*s2*s3-s2*s3)*d_dwdz[d_idx])
                My = fac*((-s1*s2**2*s3-s1*s3**3)*d_dudx[d_idx]
                          + (s1**2*s2*s3-s2*s3)*d_dudy[d_idx]
                          + (s1**2*s3**2+s2**2)*d_dudz[d_idx]
                          + (-s2**4-2*s2**2*s3**2-s3**4)*d_dwdx[d_idx]
                          + (s1*s2**3+s1*s2*s3**2)*d_dwdy[d_idx]
                          + (s1*s2**2*s3+s1*s3**3)*d_dwdz[d_idx])
                Mz = fac*((s1*s2**3+s1*s2*s3**2)*d_dudx[d_idx]
                          + (-s1**2*s2**2-s3**2)*d_dudy[d_idx]
                          + (-s1**2*s2*s3+s2*s3)*d_dudz[d_idx]
                          + (s2**4+2*s2**2*s3**2+s3**4)*d_dvdx[d_idx]
                          + (-s1*s2**3-s1*s2*s3**2)*d_dvdy[d_idx]
                          + (-s1*s2**2*s3-s1*s3**3)*d_dvdz[d_idx])
            else:
                fac = (3.*self.dx*self.d**2/4.*self.mu*pi)/(s1**2+s3**2)

                Mx = fac*((-s1*s2**2*s3+s1*s3)*d_dvdx[d_idx]
                          + (s1**2*s2*s3+s2*s3**3)*d_dvdy[d_idx]
                          + (-s2**2*s3**2-s1**2)*d_dvdz[d_idx]
                          + (-s1**3*s2-s1*s2*s3**2)*d_dwdx[d_idx]
                          + (s1**4+2*s1**2*s3**2+s3**4)*d_dwdy[d_idx]
                          + (-s1**2*s2*s3-s2*s3**3)*d_dwdz[d_idx])
                My = fac*((s1*s2**2*s3-s1*s3)*d_dudx[d_idx]
                          + (-s1**2*s2*s3-s2*s3**3)*d_dudy[d_idx]
                          + (s2**2*s3**2+s1**2)*d_dudz[d_idx]
                          + (-s1**2*s2**2-s3**2)*d_dwdx[d_idx]
                          + (s1**3*s2+s1*s2*s3**2)*d_dwdy[d_idx]
                          + (-s1*s2**2*s3+s1*s3)*d_dwdz[d_idx])
                Mz = fac*((s1**3*s2+s1*s2*s3**2)*d_dudx[d_idx]
                          + (-s1**4-2*s1**2*s3**2-s3**4)*d_dudy[d_idx]
                          + (s1**2*s2*s3+s2*s3**3)*d_dudz[d_idx]
                          + (s1**2*s2**2+s3**2)*d_dvdx[d_idx]
                          + (-s1**3*s2-s1*s2*s3**2)*d_dvdy[d_idx]
                          + (s1*s2**2*s3-s1*s3)*d_dvdz[d_idx])

            d_au[d_idx+1] += (My*d_rznext[d_idx]-Mz*d_rynext[d_idx])/(2*self.J)
            d_av[d_idx+1] += (Mz*d_rxnext[d_idx]-Mx*d_rznext[d_idx])/(2*self.J)
            d_aw[d_idx+1] += (Mx*d_rynext[d_idx]-My*d_rxnext[d_idx])/(2*self.J)

            d_au[d_idx-1] += (My*d_rzprev[d_idx]-Mz*d_ryprev[d_idx])/(2*self.J)
            d_av[d_idx-1] += (Mz*d_rxprev[d_idx]-Mx*d_rzprev[d_idx])/(2*self.J)
            d_aw[d_idx-1] += (Mx*d_ryprev[d_idx]-My*d_rxprev[d_idx])/(2*self.J)
