"""Collision of a fiber in a damped field (10 minutes).

################################################################################
Beam Collision
################################################################################

Reference
---------
N. Meyer et. al "Parameter Identification of Fiber Orientation Models Based on Direct
Fiber Simulation with Smoothed Particle Hydrodynamics",
Journal of Composites Science, 2020, 4, 77; doi:10.3390/jcs4020077
"""
from math import sqrt

import numpy as np

from pysph.base.kernels import QuinticSpline
# PySPH imports
from pysph.base.utils import get_particle_array_beadchain_fiber
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.equation import Group
from pysph.sph.fiber.beadchain import Bending, Tension
from pysph.sph.fiber.utils import ComputeDistance, Contact, Damping, HoldPoints
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.wc.transport_velocity import MomentumEquationPressureGradient


class Beam(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--D", action="store", type=float, dest="d",
            default=1, help="Factor for damping. 1 is aperiodic limit."
        )
        group.add_argument(
            "--E", action="store", type=float, dest="E",
            default=1E8, help="Young's modulus."
        )
        group.add_argument(
            "--N", action="store", type=int, dest="N",
            default=10, help="Number of particles."
        )
        group.add_argument(
            "--gx", action="store", type=float, dest="gx",
            default=0, help="Body force in x-direction."
        )
        group.add_argument(
            "--gy", action="store", type=float, dest="gy",
            default=100, help="Body force in y-direction."
        )
        group.add_argument(
            "--gz", action="store", type=float, dest="gz",
            default=0, help="Body force in z-direction."
        )
        group.add_argument(
            "--k", action="store", type=float, dest="k",
            default=0.0, help="Friction coefficient."
        )
        group.add_argument(
            "--eta", action="store", type=float, dest="eta",
            default=1.0, help="Absolute Viscosity."
        )

    def consume_user_options(self):
        # fiber length
        self.reducedL = 10.0

        # numerical setup
        self.N = self.options.N
        self.reducedL = self.reducedL/(1-1/(2*self.N))
        self.dx = self.reducedL/self.N     # particle spacing
        self.h = self.dx

        # fluid properties
        self.rho0 = 1.0
        self.p0 = 1.0

        # fiber properties
        self.A = 1.0
        self.Ip = self.A/12.0
        self.E = self.options.E
        # Analytical solution for angular eigenfrequencies:
        #       Pi/L np.sqrt(E/rho) (2n-1)/2
        # --> first analytical eigenfrequency:
        self.omega0_tension = np.pi/(2*self.reducedL)*np.sqrt(self.E/self.rho0)
        self.omega0_bending = 3.5156*np.sqrt(
            self.E*self.Ip/(self.rho0*self.A*self.reducedL**4))
        if self.options.gx > self.options.gy:
            self.omega0 = self.omega0_tension
        else:
            self.omega0 = self.omega0_bending
        m = self.rho0*self.A*self.dx
        self.D = self.options.d*m*self.omega0
        self.gx = self.options.gx
        self.gy = self.options.gy
        self.gz = self.options.gz
        print('Damping: %g, Omega0: %g' % (self.D, self.omega0))

        # setup time step
        if abs(self.gx) > 0.0 or abs(self.gy) > 0.0 or abs(self.gz) > 0.0:
            dt_force = 0.25 * np.sqrt(
                self.h/(sqrt(self.gx**2+self.gy**2+self.gz**2)))
        else:
            dt_force = 10000
        dt_tension = 0.5*self.h*np.sqrt(self.rho0/self.E)
        dt_bending = 0.5*self.h**2*np.sqrt(self.rho0*self.A/(self.E*2*self.Ip))

        self.tf = 20

        self.dt = min(dt_force, dt_tension, dt_bending)

    def create_scheme(self):
        return None

    def create_particles(self):
        _x = np.linspace(-self.dx, self.reducedL-self.dx, self.N+1)
        _y = np.array([0.0])
        _z = np.array([0.0])
        x, y, z = np.meshgrid(_x, _y, _z)
        fiber1_x = x.ravel()
        fiber1_y = y.ravel()
        fiber1_z = z.ravel()

        _x = np.array([0.75*self.reducedL])
        _y = np.array([-2*self.dx])
        _z = -np.linspace(-0.25*self.reducedL, 0.75*self.reducedL, self.N+1)
        # _x = np.linspace(0.0, self.reducedL, self.N+1)
        # _y = np.array([-1.5*self.dx])
        # _z = np.array([0.0])
        x, y, z = np.meshgrid(_x, _y, _z)
        fiber2_x = x.ravel()
        fiber2_y = y.ravel()
        fiber2_z = z.ravel()

        # volume is set as dx * A
        volume = self.A * self.dx

        # create arrays
        fiber1 = get_particle_array_beadchain_fiber(
            name='fiber1', x=fiber1_x, y=fiber1_y, z=fiber1_z,
            m=volume*self.rho0, rho=self.rho0, h=self.h, lprev=self.dx,
            lnext=self.dx, phi0=np.pi, phifrac=2.0, fidx=range(self.N+1),
            V=1./volume)
        fiber2 = get_particle_array_beadchain_fiber(
            name='fiber2', x=fiber2_x, y=fiber2_y, z=fiber2_z,
            m=volume*self.rho0, rho=self.rho0, h=self.h, lprev=self.dx,
            lnext=self.dx, phi0=np.pi, phifrac=2.0, fidx=range(self.N+1),
            V=1./volume)

        # tag particles to be hold
        fiber1.holdtag[:] = 0
        fiber1.holdtag[0] = 2
        fiber1.holdtag[1] = 1
        fiber1.holdtag[2] = 2

        # return the particle list
        return [fiber1, fiber2]

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    ComputeDistance(dest='fiber1', sources=['fiber1']),
                    ComputeDistance(dest='fiber2', sources=['fiber2']),
                ],
            ),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fiber1',
                        sources=['fiber1', 'fiber2'], pb=0.0,
                        gx=self.gx, gy=self.gy, gz=self.gz),
                    MomentumEquationPressureGradient(
                        dest='fiber2',
                        sources=['fiber1', 'fiber2'], pb=0.0,
                        gx=self.gx, gy=self.gy, gz=self.gz),
                    Tension(
                        dest='fiber1',
                        sources=None,
                        ea=self.E*self.A),
                    Tension(
                        dest='fiber2',
                        sources=None,
                        ea=self.E*self.A),
                    Bending(
                        dest='fiber1',
                        sources=None,
                        ei=self.E*self.Ip),
                    Bending(
                        dest='fiber2',
                        sources=None,
                        ei=self.E*self.Ip),
                    Contact(
                        dest='fiber1',
                        sources=['fiber1', 'fiber2'],
                        E=self.E, d=self.dx, dim=3,
                        k=self.options.k,
                        eta0=self.options.eta,
                        dt=self.dt),
                    Contact(
                        dest='fiber2',
                        sources=['fiber1', 'fiber2'],
                        E=self.E, d=self.dx, dim=3,
                        k=self.options.k,
                        eta0=self.options.eta,
                        dt=self.dt),
                    Damping(
                        dest='fiber1',
                        sources=None,
                        d=self.D),
                    Damping(
                        dest='fiber2',
                        sources=None,
                        d=self.D)
                ],
            ),
            Group(
                equations=[
                    HoldPoints(dest='fiber1', sources=None, tag=2, x=False),
                    HoldPoints(dest='fiber1', sources=None, tag=1, y=False),
                    HoldPoints(dest='fiber2', sources=None, tag=2, x=False),
                    HoldPoints(dest='fiber2', sources=None, tag=1, y=False),
                ],
            ),
        ]
        return equations

    def create_solver(self):
        """Set up the default integrator for fiber particles."""
        kernel = QuinticSpline(dim=3)
        integrator = EPECIntegrator(
            fiber1=TransportVelocityStep(),
            fiber2=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=3, integrator=integrator, dt=self.dt,
            tf=self.tf, N=200)
        return solver


if __name__ == '__main__':
    app = Beam()
    app.run()
