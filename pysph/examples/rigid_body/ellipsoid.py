"""Ellipsoid suspended in shear flow (2 hours).

An example to illustrate 3d pysph rigid_body framework
"""
from __future__ import print_function
import numpy as np
from scipy.integrate import odeint

from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_wcsph,
                              get_particle_array_rigid_body)
from pysph.solver.utils import load, remove_irrelevant_files
# PySPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, ContinuityEquation)
from pysph.sph.wc.basic import TaitEOSHGCorrection, MomentumEquation, TaitEOS
from pysph.sph.wc.viscosity import LaminarViscosity
from pysph.solver.application import Application
from pysph.sph.rigid_body import (NumberDensity, BodyForce, RigidBodyMoments,
                                  ViscosityRigidBody, PressureRigidBody,
                                  RigidBodyMotion, RK2StepRigidBody)


def jeffery_ode(phi, t, ar, G):
    """Jeffery's Equation for planar rotation of a rigid ellipsoid."""
    lbd = (ar**2-1.0)/(ar**2+1.0)
    return 0.5*G*(1.0+lbd*np.cos(2.0*phi))


class RigidFluidCoupling(Application):
    """Example of a rigid ellipsoid rotating in a shear flow."""

    def initialize(self):
        """Set up of general variables."""
        self.scale = 1000
        self.L = 0.0012
        self.dx = 0.000025
        self.hdx = 1.2
        self.rho = 1000*self.scale
        self.alpha = 0.0
        self.nu = 0.1/self.rho
        self.co = 0.010

    def create_particles(self):
        """Create particle arrays for fluis, ellipsiod and walls."""
        # General box
        _x = np.arange(-self.L/2+self.dx/2, self.L/2+self.dx/2, self.dx)
        _y = np.arange(-self.L/2+self.dx/2, self.L/2+self.dx/2, self.dx)
        _z = np.arange(-self.L/4+self.dx/2, self.L/4+self.dx/2, self.dx)
        x, y, z = np.meshgrid(_x, _y, _z)
        xf = x.ravel()
        yf = y.ravel()
        zf = z.ravel()

        # Determine the size of dummy region
        ghost_extend = 3*self.dx

        # Create the wall particles at the top
        _y = np.linspace(self.L/2+self.dx/2,
                         self.L/2-self.dx/2+ghost_extend, 3)
        x, y, z = np.meshgrid(_x, _y, _z)
        xt = x.ravel()
        yt = y.ravel()
        zt = z.ravel()

        # Create the wall particles at the bottom
        _y = np.linspace(-self.L/2+self.dx/2-ghost_extend,
                         -self.L/2-self.dx/2, 3)
        x, y, z = np.meshgrid(_x, _y, _z)
        xb = x.ravel()
        yb = y.ravel()
        zb = z.ravel()

        # Concatenate the top and bottom arrays
        xw = np.concatenate((xt, xb))
        yw = np.concatenate((yt, yb))
        zw = np.concatenate((zt, zb))

        # Create particle array for fluid
        m = self.rho * self.dx**3
        h = self.hdx * self.dx
        rad_s = self.dx/2
        V = self.dx**3
        cs = 0.0
        fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m,
                                         rho=self.rho, name="fluid")

        # Create particle array for walls
        walls = get_particle_array_wcsph(x=xw, y=yw, z=zw, h=h, m=m,
                                         rho=self.rho, rad_s=rad_s, V=V,
                                         name="walls")
        for name in ['fx', 'fy', 'fz']:
            walls.add_property(name)

        # Create particle array for ellipsoid
        cond = (((xf/(self.L/12))**2 +
                 (yf/(self.L/4))**2 +
                 (zf/(self.L/12))**2) <= 1.0)
        xe, ye, ze = xf[cond], yf[cond], zf[cond]

        ellipsoid = get_particle_array_rigid_body(x=xe, y=ye, z=ze, h=h, m=m,
                                                  rho=self.rho, rad_s=rad_s,
                                                  V=V, cs=cs, body_id=0,
                                                  name="ellipsoid")

        ellipsoid.total_mass[0] = np.sum(m)
        ellipsoid.add_property('cs')
        ellipsoid.add_property('arho')
        ellipsoid.set_lb_props(list(ellipsoid.properties.keys()))
        ellipsoid.set_output_arrays(
            ['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz',
             'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid'])

        fluid.remove_particles([i for i, c in enumerate(cond) if c])
        fluid.u[:] = fluid.y[:]
        ellipsoid.u[:] = ellipsoid.y[:]
        walls.u[:] = walls.y[:]

        print(
            fluid.get_number_of_particles(),
            walls.get_number_of_particles(),
            ellipsoid.get_number_of_particles(), )
        return [fluid, walls, ellipsoid]

    def create_domain(self):
        """Create the domain as periodic domain in x and z."""
        return DomainManager(xmin=-self.L/2, xmax=self.L/2, zmin=-self.L/4,
                             zmax=self.L/4, periodic_in_x=True,
                             periodic_in_z=True)

    def create_solver(self):
        """Create Solver with min. time step from CFL and viscous step."""
        kernel = CubicSpline(dim=3)
        integrator = EPECIntegrator(fluid=WCSPHStep(), walls=WCSPHStep(),
                                    ellipsoid=RK2StepRigidBody())

        h = self.hdx*self.dx
        dt_cfl = 0.4 * h/(1.1*self.co)
        dt_viscous = 0.125*h**2/self.nu
        dt = min(dt_viscous, dt_cfl)
        print("dt_cfl: %s" % dt_cfl)
        print("dt_viscous: %s" % dt_viscous)
        print("DT: %s" % dt)
        tf = 12
        solver = Solver(
            kernel=kernel,
            dim=3,
            integrator=integrator,
            dt=dt,
            tf=tf,
            adaptive_timestep=False, )

        return solver

    def create_equations(self):
        """Set up equations.

        Body force is necessary to reset fx,fy,fz, although
        not body force is applied.
        """
        equations = [
            Group(equations=[
                BodyForce(dest='ellipsoid', sources=None),
                NumberDensity(dest='ellipsoid', sources=['ellipsoid']),
                NumberDensity(dest='walls', sources=['walls'])
            ]),

            # Tait equation of state
            Group(equations=[
                TaitEOS(
                    dest='fluid', sources=None, rho0=self.rho, c0=self.co,
                    gamma=7.0),
                TaitEOSHGCorrection(
                    dest='ellipsoid', sources=None, rho0=self.rho, c0=self.co,
                    gamma=7.0),
                TaitEOSHGCorrection(
                    dest='walls', sources=None, rho0=self.rho, c0=self.co,
                    gamma=7.0),
            ], real=False),

            Group(equations=[
                ContinuityEquation(dest='fluid',
                                   sources=['fluid', 'walls', 'ellipsoid']),
                ContinuityEquation(dest='ellipsoid', sources=['fluid']),
                ContinuityEquation(dest='walls', sources=['fluid']),
                LaminarViscosity(dest='fluid', sources=['fluid', 'walls'],
                                 nu=self.nu),
                MomentumEquation(dest='fluid', sources=['fluid', 'walls'],
                                 alpha=self.alpha, beta=0.0, c0=self.co),
                ViscosityRigidBody(dest='fluid', sources=['ellipsoid'],
                                   nu=self.nu, rho0=self.rho),
                PressureRigidBody(dest='fluid', sources=['ellipsoid'],
                                  rho0=self.rho),
                XSPHCorrection(dest='fluid', sources=['fluid']),
            ]),
            Group(equations=[RigidBodyMoments(dest='ellipsoid',
                                              sources=None)]),
            Group(equations=[RigidBodyMotion(dest='ellipsoid',
                                             sources=None)]),
        ]
        return equations

    def post_process(self, info_fname):
        """Plot ellispoid angle and compare it to Jeffery's ODE."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            from matplotlib import rc
            rc('font', **{'family': 'serif',
                          'serif': ['Computer Modern'],
                          'size': 18})
            rc('text', usetex=True)
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        t = []
        phi = []
        output_files = remove_irrelevant_files(self.output_files)

        # Going through output files
        for i, fname in enumerate(output_files):
            data = load(fname)
            # Extract time
            t.append(data['solver_data']['t'])

            # extract relative positions of ellipsoid particles
            ellipsoid = data['arrays']['ellipsoid']
            x = ellipsoid.x-np.mean(ellipsoid.x)
            y = ellipsoid.y-np.mean(ellipsoid.y)

            # compute orienation as covariance matrix
            coords = np.vstack([x, y])
            cov = np.cov(coords)
            evals, evecs = np.linalg.eig(cov)
            sort_indices = np.argsort(evals)[::-1]
            dx, dy = evecs[:, sort_indices[0]]
            if abs(dx) < 1E-15:
                phi.append(0.0)
            else:
                phi.append(np.pi/2.0-np.arctan(dy/dx))

        # reference solution
        t = np.array(t)
        phi0 = 0.0
        angle_jeffery = odeint(jeffery_ode, phi0, t, atol=1E-15,
                               args=(3.0, 1.0))

        # open new plot
        plt.figure()

        # plot computed angle and Jeffery's solution
        plt.plot(t, phi, '-k')
        plt.plot(t, angle_jeffery, '--k')

        # labels
        plt.xlabel('Time $t$ in s')
        plt.ylabel('Rotation angle $\phi$')
        plt.legend(['SPH Simulation', 'Jeffery'])
        plt.grid()
        x1, x2, y1, y2 = plt.axis()
        plt.axis((0, x2, 0, y2))
        ax = plt.gca()
        ax.set_yticks([0, 0.5*np.pi, np.pi, 1.5*np.pi])
        ax.set_yticklabels(['0', '$\pi/2$', '$\pi$', '$3/2\pi$'])
        plt.tight_layout()

        plt.savefig("test.pdf", bbox_inches='tight')


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    app.post_process(app.info_filename)
