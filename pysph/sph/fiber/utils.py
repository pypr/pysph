"""Utitlity equations for fibers.

Reference
---------

    .. [Meyer2020] N. Meyer et. al "Parameter Identification of Fiber Orientation
    Models Based on Direct Fiber Simulation with Smoothed Particle Hydrodynamics",
    Journal of Composites Science, 2020, 4, 77; doi:10.3390/jcs4020077

"""

from math import acos, pi, sin, sqrt

from pysph.sph.equation import Equation


class ComputeDistance(Equation):
    """Compute Distances to neighbours.

    The loop saves vectors to previous and next particle only.
    """

    def loop(self, d_idx, s_idx, d_rxnext, d_rynext, d_rznext, d_rnext,
             d_rxprev, d_ryprev, d_rzprev, d_rprev, s_fractag, d_fidx,
             s_fidx, d_fractag, s_rxnext, s_rynext, s_rznext, s_rnext,
             s_rxprev, s_ryprev, s_rzprev, s_rprev, XIJ, RIJ):
        if d_fidx[d_idx] == s_fidx[s_idx]+1:
            if s_fractag[s_idx] == 0:
                d_rxprev[d_idx] = -XIJ[0]
                d_ryprev[d_idx] = -XIJ[1]
                d_rzprev[d_idx] = -XIJ[2]
                d_rprev[d_idx] = RIJ
            else:
                d_rxprev[d_idx] = 0.0
                d_ryprev[d_idx] = 0.0
                d_rzprev[d_idx] = 0.0
                d_rprev[d_idx] = 0.0
                s_rnext[s_idx] = 0.0
                s_rxnext[s_idx] = 0.0
                s_rynext[s_idx] = 0.0
                s_rznext[s_idx] = 0.0
        if d_fidx[d_idx] == s_fidx[s_idx]-1:
            if d_fractag[d_idx] == 0:
                d_rxnext[d_idx] = -XIJ[0]
                d_rynext[d_idx] = -XIJ[1]
                d_rznext[d_idx] = -XIJ[2]
                d_rnext[d_idx] = RIJ
            else:
                s_rxprev[s_idx] = 0.0
                s_ryprev[s_idx] = 0.0
                s_rzprev[s_idx] = 0.0
                s_rprev[s_idx] = 0.0
                d_rxnext[d_idx] = 0.0
                d_rynext[d_idx] = 0.0
                d_rznext[d_idx] = 0.0
                d_rnext[d_idx] = 0.0


class HoldPoints(Equation):
    """Holds flagged points.

    Points tagged with 'holdtag' == tag are excluded from accelaration. This
    little trick allows testing of fibers with fixed BCs.
    """

    def __init__(self, dest, sources, tag, x=True, y=True, z=True,
                 mirror_particle=0):
        r"""
        Parameters
        ----------
        tags : int
            tag of fixed particle defined as property 'holdtag'
        x : boolean
            True, if x-position should not be changed
        y : boolean
            True, if y-position should not be changed
        z : boolean
            True, if z-position should not be changed
        mirror_particle : int
            idx shift to a particle, which displacement should be mirrored to
            origin
        """
        self.tag = tag
        self.x = x
        self.y = y
        self.z = z
        self.mirror = mirror_particle
        super(HoldPoints, self).__init__(dest, sources)

    def loop(self, d_idx, d_holdtag, d_au, d_av, d_aw, d_auhat, d_avhat,
             d_awhat, d_u, d_v, d_w, d_x, d_y, d_z, d_Fx, d_Fy, d_Fz, d_m):
        if d_holdtag[d_idx] == self.tag:
            if self.x:
                d_Fx[d_idx] = d_m[d_idx] * d_au[d_idx]
                d_au[d_idx] = 0
                d_auhat[d_idx] = 0
                d_u[d_idx] = 0
            elif not self.mirror == 0:
                # Copy properties to mirror particle
                d_x[d_idx+self.mirror] = -d_x[d_idx]
                d_u[d_idx+self.mirror] = -d_u[d_idx]

            if self.y:
                d_Fy[d_idx] = d_m[d_idx] * d_av[d_idx]
                d_av[d_idx] = 0
                d_avhat[d_idx] = 0
                d_v[d_idx] = 0
            elif not self.mirror == 0:
                # Copy properties to mirror particle
                d_y[d_idx+self.mirror] = -d_y[d_idx]
                d_v[d_idx+self.mirror] = -d_v[d_idx]

            if self.z:
                d_Fz[d_idx] = d_m[d_idx] * d_aw[d_idx]
                d_aw[d_idx] = 0
                d_awhat[d_idx] = 0
                d_w[d_idx] = 0
            elif not self.mirror == 0:
                # Copy properties to mirror particle
                d_z[d_idx+self.mirror] = -d_z[d_idx]
                d_w[d_idx+self.mirror] = -d_w[d_idx]


class Vorticity(Equation):
    """Computes vorticity of velocity field

    According to Monaghan 1992 (2.12).
    """

    def initialize(self, d_idx, d_omegax, d_omegay, d_omegaz):
        d_omegax[d_idx] = 0.0
        d_omegay[d_idx] = 0.0
        d_omegaz[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_m, d_omegax, d_omegay, d_omegaz,
             DWIJ, VIJ):
        v = s_m[s_idx]/d_rho[d_idx]
        d_omegax[d_idx] += v*(VIJ[1]*DWIJ[2]-VIJ[2]*DWIJ[1])
        d_omegay[d_idx] += v*(VIJ[2]*DWIJ[0]-VIJ[0]*DWIJ[2])
        d_omegaz[d_idx] += v*(VIJ[0]*DWIJ[1]-VIJ[1]*DWIJ[0])


class VelocityGradient(Equation):
    """Computes 2nd order tensor representing the velocity gradient.

    See eq. (25) in [Meyer2020].
    """

    def initialize(self, d_idx, d_dudx, d_dudy, d_dudz, d_dvdx, d_dvdy, d_dvdz,
                   d_dwdx, d_dwdy, d_dwdz):
        d_dudx[d_idx] = 0.0
        d_dudy[d_idx] = 0.0
        d_dudz[d_idx] = 0.0

        d_dvdx[d_idx] = 0.0
        d_dvdy[d_idx] = 0.0
        d_dvdz[d_idx] = 0.0

        d_dwdx[d_idx] = 0.0
        d_dwdy[d_idx] = 0.0
        d_dwdz[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_dudx, d_dudy, d_dudz, d_dvdx,
             d_dvdy, d_dvdz, d_dwdx, d_dwdy, d_dwdz, DWIJ, VIJ):
        v = s_m[s_idx]/s_rho[s_idx]
        d_dudx[d_idx] -= v*VIJ[0]*DWIJ[0]
        d_dudy[d_idx] -= v*VIJ[0]*DWIJ[1]
        d_dudz[d_idx] -= v*VIJ[0]*DWIJ[2]

        d_dvdx[d_idx] -= v*VIJ[1]*DWIJ[0]
        d_dvdy[d_idx] -= v*VIJ[1]*DWIJ[1]
        d_dvdz[d_idx] -= v*VIJ[1]*DWIJ[2]

        d_dwdx[d_idx] -= v*VIJ[2]*DWIJ[0]
        d_dwdy[d_idx] -= v*VIJ[2]*DWIJ[1]
        d_dwdz[d_idx] -= v*VIJ[2]*DWIJ[2]


class Damping(Equation):
    """Damp particle motion.

    Particles are damped. Difference to ArtificialDamping: This damps real
    particle velocities and therefore affects not only the fiber iteration. In this
    contect it may be used to test fiber contact in a damped environment.
    """

    def __init__(self, dest, sources, d):
        r"""
        Parameters
        ----------
        d : float
            damping coefficient
        """
        self.d = d
        super(Damping, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_u, d_v, d_w, d_au, d_av, d_aw):
        d_au[d_idx] -= 2*self.d*d_u[d_idx]/d_m[d_idx]
        d_av[d_idx] -= 2*self.d*d_v[d_idx]/d_m[d_idx]
        d_aw[d_idx] -= 2*self.d*d_w[d_idx]/d_m[d_idx]


class SimpleContact(Equation):
    """This class computes simple fiber repulsion to stop penetration.

    It computes the force between two spheres as Hertz pressure.
    """

    def __init__(self, dest, sources, E, d, pois=0.3):
        r"""
        Parameters
        ----------
        E : float
            Young's modulus
        d : float
            fiber diameter
        pois : flost
            poisson number
        """
        self.E = E
        self.d = d
        self.pois = pois
        super(SimpleContact, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_fidx, s_fidx, d_m, d_au, d_av, d_aw,
             XIJ, RIJ):
        if not s_fidx[s_idx] == d_fidx[d_idx] and RIJ < self.d:
            E_star = 1/(2*((1-self.pois**2)/self.E))
            # effective radius for two spheres of same size
            R = self.d/4
            F = 4/3 * E_star * sqrt(R) * abs(self.d-RIJ)**1.5
            d_au[d_idx] += XIJ[0]/RIJ * F/d_m[d_idx]
            d_av[d_idx] += XIJ[1]/RIJ * F/d_m[d_idx]
            d_aw[d_idx] += XIJ[2]/RIJ * F/d_m[d_idx]


class Contact(Equation):
    """This class computes fiber repulsion to stop penetration.

    Itcomputes the force between two spheres based on Hertz pressure between two
    cylinders. This Equation requires a computation of ditances by the Bending
    equation.

    See eq. (27)-(34) in [Meyer2020].
    """
    def __init__(self, dest, sources, E, d, dim, pois=0.3, k=0.0, lim=0.1,
                 eta0=0.0, dt=0.0):
        r"""
        Parameters
        ----------
        E : float
            Young's modulus
        d : float
            fiber diameter
        dim : int
            dimensionaltiy of the problem
        pois : float
            poisson number
        k : float
            friction coefficient between fibers
        eta0 : float
            viscosity of suspension fluid
        """
        self.E = E
        self.d = d
        self.pois = pois
        self.k = k
        self.dim = dim
        self.lim = lim
        self.eta0 = eta0
        self.dt = dt
        self.Fx = 0.0
        self.Fy = 0.0
        self.Fz = 0.0
        self.nx = 0.0
        self.ny = 0.0
        self.nz = 0.0
        self.proj = 0.0
        self.dist = 0.0
        super(Contact, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_Fx, d_Fy, d_Fz):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0
        d_Fx[d_idx] = 0.0
        d_Fy[d_idx] = 0.0
        d_Fz[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_au, d_av, d_aw, d_rxnext, d_rynext,
             d_rznext, d_rnext, d_rxprev, d_ryprev, d_rzprev, d_rprev,
             s_rxnext, s_rynext, s_rznext, s_rnext, s_rxprev, s_ryprev,
             s_rzprev, s_rprev, d_Fx, d_Fy, d_Fz, d_fractag, d_tag, s_tag,
             XIJ, VIJ, RIJ, EPS):
        """Loop over all particles and compute interaction forces."""

        # exclude self and direct neighbors
        if (RIJ > 1E-14
            and RIJ < 1.5*self.d
            and abs(RIJ-d_rprev[d_idx]) > 1E-14
                and abs(RIJ-d_rnext[d_idx]) > 1E-14):

            # unit direction of destination fiber
            dx = d_rxprev[d_idx] - d_rxnext[d_idx]
            dy = d_ryprev[d_idx] - d_rynext[d_idx]
            dz = d_rzprev[d_idx] - d_rznext[d_idx]
            dr = sqrt(dx**2 + dy**2 + dz**2)
            dx = dx/dr
            dy = dy/dr
            dz = dz/dr

            # unit direction of source fiber
            sx = s_rxprev[s_idx] - s_rxnext[s_idx]
            sy = s_ryprev[s_idx] - s_rynext[s_idx]
            sz = s_rzprev[s_idx] - s_rznext[s_idx]
            sr = sqrt(sx**2 + sy**2 + sz**2)
            sx = sx/sr
            sy = sy/sr
            sz = sz/sr

            # conditions
            d_prev_tip = (d_rprev[d_idx] < EPS
                          and dx*XIJ[0]+dy*XIJ[1]+dz*XIJ[2] < EPS)
            s_prev_tip = (s_rprev[s_idx] < EPS
                          and sx*XIJ[0]+sy*XIJ[1]+sz*XIJ[2] > EPS)
            d_next_tip = (d_rnext[d_idx] < EPS
                          and dx*XIJ[0]+dy*XIJ[1]+dz*XIJ[2] > EPS)
            s_next_tip = (s_rnext[s_idx] < EPS
                          and sx*XIJ[0]+sy*XIJ[1]+sz*XIJ[2] < EPS)

            # determine case
            if d_prev_tip or d_next_tip or dr < EPS:
                if s_prev_tip or s_next_tip or sr < EPS:
                    alpha = self.compute_point_point_props(XIJ, RIJ)
                    w = 1.0
                else:
                    alpha = self.compute_center_point_props(XIJ, sx, sy, sz)
                    alpha = 0.0
                    w = self.weight(s_rnext[s_idx], s_rprev[s_idx], self.proj)
            elif s_prev_tip or s_next_tip or sr < EPS:
                if d_prev_tip or d_next_tip or dr < EPS:
                    alpha = self.compute_point_point_props(XIJ, RIJ)
                    w = 1.0
                else:
                    alpha = self.compute_center_point_props(XIJ, dx, dy, dz)
                    w = self.weight(d_rnext[d_idx], d_rprev[d_idx], self.proj)
            else:
                # center and center
                alpha = self.compute_center_center_props(
                    XIJ, dx, dy, dz, sx, sy, sz)
                w = self.weight(d_rnext[d_idx], d_rprev[d_idx], self.proj)

            d = self.d - self.dist
            if d >= 0.0:
                self.compute_contact_force(VIJ[0], VIJ[1], VIJ[2], w, d)
            elif alpha > 0.0:
                pass
                # self.compute_lubrication_force(VIJ[0], VIJ[1], VIJ[2], w, d, alpha)
            else:
                self.Fx = 0.0
                self.Fy = 0.0
                self.Fz = 0.0

            d_Fx[d_idx] += self.Fx
            d_Fy[d_idx] += self.Fy
            d_Fz[d_idx] += self.Fz

            d_au[d_idx] += self.Fx/d_m[d_idx]
            d_av[d_idx] += self.Fy/d_m[d_idx]
            d_aw[d_idx] += self.Fz/d_m[d_idx]

    def compute_center_point_props(self, XIJ, ax=0.0, ay=0.0, az=0.0):
        """Determine normal, distance and projected length - Center-Point.

        The projection looks like this:

                       x (source point )
                      /|
                  XIJ/ | distance
                    /  |
        dest point o---*----> unit vector a
                    proj
        """
        projection = -(ax*XIJ[0] + ay*XIJ[1] + az*XIJ[2])
        tx = XIJ[0] + projection*ax
        ty = XIJ[1] + projection*ay
        tz = XIJ[2] + projection*az
        tr = sqrt(tx**2 + ty**2 + tz**2)

        self.nx = tx/tr
        self.ny = ty/tr
        self.nz = tz/tr
        self.proj = projection
        self.dist = tr

        return 0.0

    def compute_point_point_props(self, XIJ, RIJ):
        """Compute the normal between end points.

        This case is trivial.
        """
        self.nx = XIJ[0]/RIJ
        self.ny = XIJ[1]/RIJ
        self.nz = XIJ[2]/RIJ
        self.dist = RIJ

        return 0.0

    def compute_center_center_props(self, XIJ, dx=0.0, dy=0.0, dz=0.0,
                                    sx=0.0, sy=0.0, sz=0.0):
        """Compute normal direction between two lines.

        The normal is given either by the cross product or by a projection, if
        both lines are (almost) parallel.
        """
        # normal direction at contact
        nx = dy*sz - dz*sy
        ny = dz*sx - dx*sz
        nz = dx*sy - dy*sx
        nr = sqrt(nx**2 + ny**2 + nz**2)

        # Parallel vectors should be treated with a projection
        if nr > 0.01:
            self.nx = nx/nr
            self.ny = ny/nr
            self.nz = nz/nr
            alpha = acos(dx*sx+dy*sy+dz*sz)
            self.project_center_center(XIJ, dx, dy, dz, sx, sy, sz)
        else:
            self.compute_center_point_props(XIJ, dx, dy, dz)
            alpha = pi/2.

        return alpha

    def project_center_center(self, XIJ, dx=0.0, dy=0.0, dz=0.0,
                              sx=0.0, sy=0.0, sz=0.0):
        """Find distance and projection of contact point on fiber line.

        Therefore, solve
        |dx  nx -sx|   |dest projection |     |XIJ[0]|
        |dy  ny -sy|   |distance        |  =  |XIJ[1]|
        |dz  nz -sz|   |src projection  |     |XIJ[2]|

        """
        nx = self.nx
        ny = self.ny
        nz = self.nz
        det = (dx*ny*(-sz) + nx*(-sy)*dz + (-sx)*dy*nz
               - (-sx)*ny*dz - dx*(-sy)*nz - nx*dy*(-sz))
        self.proj = ((sy*nz - ny*sz)*XIJ[0]
                     + (nx*sz - sx*nz)*XIJ[1]
                     + (sx*ny - nx*sy)*XIJ[2])/det
        self.dist = ((dy*sz - sy*dz)*XIJ[0]
                     + (sx*dz - dx*sz)*XIJ[1]
                     + (dx*sy - sx*dy)*XIJ[2])/det
        if self.dist < 0.0:
            self.dist = -self.dist
            self.nx = -nx
            self.ny = -ny
            self.nz = -nz

    def weight(self, next=0.0, prev=0.0, proj=0.0):
        """Weight force contribution among neighboring particles.

        Any projection outside the range between the previous and next particle
        is weighted with zero. Anything between is linearly distributed with
        special treating for the edge cases of fiber tips.
        """
        w = 0.0
        if next < 1E-14 and proj > -prev and proj <= 0:
            w = (prev+proj)/prev
        elif prev < 1E-14 and proj < next and proj >= 0:
            w = (next-proj)/next
        elif proj < prev and proj >= 0:
            w = (prev-proj)/prev
        elif proj > -next and proj <= 0:
            w = (next+proj)/next
        return w

    def compute_contact_force(self, vx=0.0, vy=0.0, vz=0.0, w=0.0, d=1.0):
        """Compute the contact force at interaction point."""

        v_dot_n = vx*self.nx + vy*self.ny + vz*self.nz

        # elastic factor from Hertz' pressure in contact
        E_star = 1/(2*((1-self.pois**2)/self.E))

        d = min(self.lim*self.d, d)

        if self.dim == 3:
            F = 4/3 * d**1.5 * sqrt(self.d/2) * E_star
        else:
            # workaround for 2D contact (2 reactangular surfaces)
            F = self.E*d

        vrx = vx - v_dot_n*self.nx
        vry = vy - v_dot_n*self.ny
        vrz = vz - v_dot_n*self.nz
        v_rel = sqrt(vrx**2 + vry**2 + vrz**2)
        v_rel = v_rel if v_rel > 1E-14 else 1

        self.Fx = w*(F*self.nx - self.k*F*vrx/v_rel)
        self.Fy = w*(F*self.ny - self.k*F*vry/v_rel)
        self.Fz = w*(F*self.nz - self.k*F*vrz/v_rel)

    def compute_lubrication_force(self, vx=0.0, vy=0.0, vz=0.0, w=0.0, d=1.0,
                                  alpha=pi/2):
        """Compute the lubrication force at interaction point."""
        # limit extreme forces
        d = min(d, -0.01*self.d)
        R = self.d/2

        v_dot_n = vx*self.nx + vy*self.ny + vz*self.nz

        if abs(alpha) > 0.1:
            A = R**2/abs(sin(alpha))
            F = 12.*A*pi*self.eta0*v_dot_n/d
        else:
            # Lindstroem limit to treat parallel cylinders (singularity!)
            A0 = 3.*pi*sqrt(2.)/8.
            A1 = 207*pi*sqrt(2.)/160.
            L = self.d
            F = -L*self.eta0*v_dot_n*(A0-A1*d/R)*(-d/R)**(-3./2.)

        self.Fx = w*F*self.nx
        self.Fy = w*F*self.ny
        self.Fz = w*F*self.nz
