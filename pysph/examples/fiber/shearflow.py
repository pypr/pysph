"""Example for fiber shearflow.

################################################################################
3D shearflow with a single fiber
################################################################################

Reference
---------
N. Meyer et. al "Parameter Identification of Fiber Orientation Models Based on Direct
Fiber Simulation with Smoothed Particle Hydrodynamics",
Journal of Composites Science, 2020, 4, 77; doi:10.3390/jcs4020077
"""

import os

import numpy as np

from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain_fiber,
                              get_particle_array_beadchain_fluid,
                              get_particle_array_beadchain_solid)
from pysph.solver.application import Application
from pysph.solver.utils import load, remove_irrelevant_files
from pysph.sph.scheme import BeadChainScheme
from scipy.integrate import odeint

# from pysph.solver.tools import FiberIntegrator


def get_zhang_aspect_ratio(aspect_ratio):
    """Jeffery's equivalent aspect ratio.

    Approximation for small aspect ratios from
    Zhang et al. 2011
    """
    return (0.000035*aspect_ratio**3 - 0.00467*aspect_ratio**2 +
            0.764*aspect_ratio + 0.404)


def jeffery_ode(phi, t, ar, G):
    """Jeffery's Equation for planar rotation of a rigid."""
    lbd = (ar**2. - 1.)/(ar**2. + 1.)
    return 0.5*G*(1. + lbd*np.cos(2.*phi))


class Channel(Application):
    """Application for the channel flow driven by top an bottom walls."""

    def create_scheme(self):
        """Use BeadChainScheme for this application."""
        return BeadChainScheme(['fluid'], ['channel'], ['fiber'], dim=3)

    def add_user_options(self, group):
        """Add options to aplication."""
        group.add_argument(
            "--dx", action="store", type=float, dest="dx",
            default=0.0001, help="Particle Spacing"
        )
        group.add_argument(
            "--lf", action="store", type=int, dest="lf",
            default=5, help="Fiber length in multiples of dx"
        )
        group.add_argument(
            "--mu", action="store", type=float, dest="mu",
            default=1.0, help="Absolute viscosity"
        )
        group.add_argument(
            "--S", action="store", type=float, dest="S",
            default=100, help="Dimensionless fiber stiffness"
        )
        group.add_argument(
            "--G", action="store", type=float, dest="G",
            default=1.0, help="Shear rate"
        )
        group.add_argument(
            "--Re", action="store", type=float, dest="Re",
            default=0.5, help="Desired Reynolds number."
        )
        group.add_argument(
            "--frac", action="store", type=float, dest="phifrac",
            default=5.0, help="Critical bending angle for fracture."
        )
        group.add_argument(
            "--rot", action="store", type=float, dest="rot",
            default=1.0, help="Number of half rotations."
        )

    def consume_user_options(self):
        """Initialize geometry, properties and time stepping."""
        # Initial spacing of particles
        self.dx = self.options.dx
        self.h0 = self.dx

        # The fiber length is the aspect ratio times fiber diameter
        self.Lf = self.options.lf*self.dx

        # Use fiber aspect ratio to determine the channel width.
        self.Ly = self.Lf + 2.*self.dx

        # Density from Reynolds number
        self.Vmax = self.options.G*self.Ly/2.
        self.rho0 = (self.options.mu*self.options.Re)/(self.Vmax*self.Lf)

        # The channel length is twice the width + dx to make it symmetric.
        self.Lx = 2.*self.Ly + self.dx

        # The position of the fiber's center is set to the center of the
        # channel.
        self.x_fiber = 0.5*self.Lx
        self.y_fiber = 0.5*self.Ly
        self.z_fiber = 0.5*self.Ly

        # The kinematic viscosity is computed from absolute viscosity and
        # scaled density.
        self.nu = self.options.mu/self.rho0

        # mass properties
        R = self.dx/np.sqrt(np.pi)    # Assuming cylindrical shape
        self.d = 2.*R
        self.ar = self.Lf/self.d      # Actual fiber aspect ratio
        print('Aspect ratio is %f' % self.ar)

        self.A = np.pi*R**2.
        self.Ip = np.pi*R**4./4.
        mass = 3.*self.rho0*self.dx*self.A
        self.J = 1./4.*mass*R**2. + 1./12.*mass*(3.*self.dx)**2.

        # stiffness from dimensionless stiffness
        self.E = 4.0/np.pi*(
            self.options.S*self.options.mu*self.options.G*self.ar)

        # SPH uses weakly compressible fluids. Therefore, the speed of sound c0
        # is computed as 10 times the maximum velocity. This should keep the
        # density change within 1%
        self.c0 = 10.*self.Vmax
        self.p0 = self.c0**2*self.rho0

        # Background pressure in Adami's transport velocity formulation
        self.pb = self.p0

        # Time
        ar = get_zhang_aspect_ratio(self.ar)
        self.t = self.options.rot*np.pi*(ar + 1./ar)/self.options.G
        print("Simulated time is %g s" % self.t)

    def configure_scheme(self):
        """Set up solver and scheme."""
        self.scheme.configure(
            rho0=self.rho0,
            c0=self.c0,
            nu=self.nu,
            p0=self.p0,
            pb=self.pb,
            h0=self.h0,
            dx=self.dx,
            A=self.A,
            Ip=self.Ip,
            J=self.J,
            E=self.E,
            d=self.d,
            direct=True)
        self.scheme.configure_solver(
            tf=self.t,
            # pfreq=1,
            N=self.options.rot*200
            )

    def create_particles(self):
        """Three particle arrays are created.

        A fluid, representing the polymer matrix, a fiber with additional
        properties and a channel of dummyparticles.
        """
        # The fluid might be scaled compared to the fiber. fdx is a shorthand
        # for the fluid spacing and dx2 is a shorthand for the half of it.
        fdx = self.dx
        dx2 = fdx/2.

        # Creating grid points for particles
        _x = np.arange(dx2, self.Lx, fdx)
        _y = np.arange(dx2, self.Ly, fdx)
        _z = np.arange(dx2, self.Ly, fdx)
        fx, fy, fz = self.get_meshgrid(_x, _y, _z)

        # Remove particles at fiber position
        indices = []
        for i in range(len(fx)):
            xx = self.x_fiber
            yy = self.y_fiber
            zz = self.z_fiber

            # vertical
            if (fx[i] < xx + self.dx/2. and fx[i] > xx - self.dx/2. and
                fy[i] < yy + self.Lf/2. and fy[i] > yy - self.Lf/2. and
                    fz[i] < zz + self.dx/2. and fz[i] > zz - self.dx/2.):
                indices.append(i)

        # create vertical fiber
        _fibx = np.array([xx])
        _fiby = np.arange(yy - self.Lf/2. + self.dx/2.,
                          yy + self.Lf/2. + self.dx/4.,
                          self.dx)

        _fibz = np.array([zz])
        fibx, fiby, fibz = self.get_meshgrid(_fibx, _fiby, _fibz)

        # Determine the size of dummy region
        ghost_extent = 3.*fdx

        # Create the channel particles at the top
        _y = np.arange(self.Ly + dx2, self.Ly + dx2 + ghost_extent, fdx)
        tx, ty, tz = self.get_meshgrid(_x, _y, _z)

        # Create the channel particles at the bottom
        _y = np.arange(-dx2, -dx2 - ghost_extent, -fdx)
        bx, by, bz = self.get_meshgrid(_x, _y, _z)

        # Concatenate the top and bottom arrays
        cx = np.concatenate((tx, bx))
        cy = np.concatenate((ty, by))
        cz = np.concatenate((tz, bz))

        # Computation of each particles initial volume.
        volume = fdx**3.

        # Mass is set to get the reference density of rho0.
        mass = volume*self.rho0

        # assign unique ID (within fiber) to each fiber particle.
        fidx = range(0, self.options.lf)

        # Initial inverse volume (necessary for transport velocity equations)
        V = 1./volume

        # Finally create all particle arrays. Note that fluid particles are
        # removed in the area, where the fiber is placed.
        channel = get_particle_array_beadchain_solid(
            name='channel', x=cx, y=cy, z=cz, m=mass, rho=self.rho0,
            h=self.h0, V=V)
        fluid = get_particle_array_beadchain_fluid(
            name='fluid', x=fx, y=fy, z=fz, m=mass, rho=self.rho0,
            h=self.h0, V=V)
        fluid.remove_particles(indices)
        fiber = get_particle_array_beadchain_fiber(
            name='fiber', x=fibx, y=fiby, z=fibz, m=mass,
            rho=self.rho0, h=self.h0, lprev=self.dx, lnext=self.dx,
            phi0=np.pi, phifrac=self.options.phifrac, fidx=fidx, V=V)

        # The number of fiber particles should match the aspect ratio. This
        # assertation fails, if something was wrong in the fiber generation.
        assert(fiber.get_number_of_particles() == self.options.lf)

        # Setting the initial velocities for a shear flow.
        fluid.u[:] = self.options.G*(fluid.y[:] - self.Ly/2.)
        fiber.u[:] = self.options.G*(fiber.y[:] - self.Ly/2.)
        channel.u[:] = self.options.G*np.sign(channel.y[:])*self.Ly/2.

        # Return the particle list.
        return [fluid, channel, fiber]

    def create_domain(self):
        """Create periodic boundary conditions in x-direction."""
        return DomainManager(xmin=0, xmax=self.Lx, zmin=0, zmax=self.Ly,
                             periodic_in_x=True, periodic_in_z=True)

    # def create_tools(self):
    #     """Add an integrator for the fiber."""
    #     return [FiberIntegrator(self.particles, self.scheme, self.domain)]

    def get_meshgrid(self, xx, yy, zz):
        """Generate meshgrids quickly."""
        x, y, z = np.meshgrid(xx, yy, zz)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        return [x, y, z]

    def _plots(self):
        """Create plots.

        It is employing a iteration over all time steps.
        """
        from matplotlib import pyplot as plt
        # empty list for time and orientation angle
        t = []
        angle = []
        N = 0

        # iteration over all output files
        output_files = remove_irrelevant_files(self.output_files)
        for i, fname in enumerate(output_files):
            data = load(fname)
            # extracting time and fiber data
            t.append(data['solver_data']['t'])
            fiber = data['arrays']['fiber']

            # computation of orientation angle
            dxx = fiber.x[0] - fiber.x[-1]
            dyy = fiber.y[0] - fiber.y[-1]
            a = np.arctan(dxx / (dyy + 0.01 * self.h0)) + N * np.pi
            if len(angle) > 0 and a - angle[-1] > 3:
                N -= 1
                a -= np.pi
            elif len(angle) > 0 and a - angle[-1] < -3:
                N += 1
                a += np.pi
            angle.append(a)

        # Integrate Jeffery's solution
        print("Solving Jeffery's ODE")
        t = np.array(t)
        phi0 = angle[0]
        ar_zhang = get_zhang_aspect_ratio(self.ar)
        angle_jeffery_zhang = odeint(jeffery_ode, phi0, t, atol=1E-15,
                                     args=(ar_zhang, self.options.G))

        # open new plot
        plt.figure()

        # plot computed angle and Jeffery's solution
        plt.plot(t*self.options.G, angle, '-k')
        plt.plot(t*self.options.G, angle_jeffery_zhang, '--k', color='grey')

        # labels
        plt.xlabel('Strains $tG$')
        plt.ylabel('Rotation angle $\phi$')
        plt.legend(['SPH Simulation', 'Jeffery (Zhang)'])
        plt.grid()
        x1, x2, y1, y2 = plt.axis()
        plt.axis((0, x2, 0, y2))
        ax = plt.gca()
        ax.set_yticks([0, 0.5*np.pi, np.pi, 1.5*np.pi])
        ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3/2\pi$'])
        plt.tight_layout()

        # save figure
        angfig = os.path.join(self.output_dir, 'angleplot.pdf')
        plt.savefig(angfig, dpi=300, bbox_inches='tight')
        try:
            tex_fig = os.path.join(self.output_dir, "angleplot.tex")
            from tikzplotlib import save as tikz_save
            tikz_save(tex_fig)
        except ImportError:
            print("Did not write tikz figure.")
        print("Angleplot written to %s." % angfig)

    def post_process(self, info_fname):
        """Build plots and files as results."""
        if len(self.output_files) == 0:
            return
        self._plots()


if __name__ == '__main__':
    app = Channel()
    app.run()
    app.post_process(app.info_filename)
