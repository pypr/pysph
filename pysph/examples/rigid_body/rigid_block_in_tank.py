"""A block of rigid solid falling on a tank of water. (6 minutes)

This also demonstrates that rigid-rigid collisions work as well.

"""
# NumPy
import numpy as np

# PySPH imports
from pysph.base.utils import get_particle_array_wcsph as gpa
from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.kernels import WendlandQuintic

from pysph.solver.solver import Solver
from pysph.solver.application import Application

from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

# the eqations
from pysph.sph.equation import Group

from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
from pysph.sph.wc.basic import TaitEOS, TaitEOSHGCorrection, MomentumEquation, \
    ContinuityEquationDeltaSPH


from pysph.sph.rigid_body import (BodyForce, NumberDensity, RigidBodyCollision,
    RigidBodyMoments, RigidBodyMotion,
    RK2StepRigidBody, ViscosityRigidBody, PressureRigidBody)

# domain and reference values
Lx = 2.0; Ly = 1.0; H = 0.5
gy = -1.0
Vmax = np.sqrt(abs(gy) * H)
c0 = 10 * Vmax; rho0 = 1000.0
rho_block = rho0
p0 = c0*c0*rho0
gamma = 1.0
alpha = 0.1
beta = 0.0
# Reynolds number and kinematic viscosity
Re = 100; nu = Vmax * Ly/Re

# Numerical setup
nx = 100; dx = Lx/nx
ghost_extent = 5.5 * dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0/( c0 + Vmax )
dt_viscous = 0.125 * h0**2/nu
dt_force = 0.25 * np.sqrt(h0/abs(gy))

tf = 6.0
dt = 0.5 * min(dt_cfl, dt_viscous, dt_force)

class RigidBlockInTank(Application):
    def create_particles(self):
        side = 0.1

        _x = np.arange(Lx*0.5-side , Lx*0.5 + side, dx )
        _y = np.arange( 0.8, 0.8+side*2, dx )
        x, y = np.meshgrid(_x, _y); x = x.ravel(); y = y.ravel()
        x += 0.25
        m = np.ones_like(x)*dx*dx*rho_block
        h = np.ones_like(x)*hdx*dx
        rho = np.ones_like(x)*rho_block
        block = get_particle_array_rigid_body(
            name='block', x=x, y=y, h=h, m=m, rho=rho
        )
        block.total_mass[0] = np.sum(m)
        block.vc[0] = 1.0

        # create all the particles
        _x = np.arange( -ghost_extent, Lx + ghost_extent, dx )
        _y = np.arange( -ghost_extent, Ly, dx )
        x, y = np.meshgrid(_x, _y); x = x.ravel(); y = y.ravel()

        # sort out the fluid and the solid
        indices = []
        for i in range(x.size):
            if ( (x[i] > 0.0) and (x[i] < Lx) ):
                if ( (y[i] > 0.0) and (y[i] < H) ):
                    indices.append(i)

        # create the arrays
        solid = gpa(name='solid', x=x, y=y)

        # remove the fluid particles from the solid
        fluid = solid.extract_particles(indices); fluid.set_name('fluid')
        solid.remove_particles(indices)

        # remove the lid to generate an open tank
        indices = []
        for i in range(solid.get_number_of_particles()):
            if solid.y[i] > H:
                if (0 < solid.x[i] < Lx):
                    indices.append(i)
        solid.remove_particles(indices)

        print("Hydrostatic tank :: nfluid = %d, nsolid=%d, dt = %g"%(
            fluid.get_number_of_particles(),
            solid.get_number_of_particles(), dt))

        ###### ADD PARTICLE PROPS SPH ######

        for prop in ('arho', 'cs', 'V', 'fx', 'fy', 'fz'):
            solid.add_property(prop )
            block.add_property(prop )

        ##### INITIALIZE PARTICLE PROPS #####
        fluid.rho[:] = rho0
        solid.rho[:] = rho0

        fluid.rho0[:] = rho0
        solid.rho0[:] = rho0

        # mass is set to get the reference density of rho0
        volume = dx * dx

        fluid.m[:] = volume * rho0
        solid.m[:] = volume * rho0

        # smoothing lengths
        fluid.h[:] = hdx * dx
        solid.h[:] = hdx * dx

        # return the particle list
        return [fluid, solid, block]

    def create_solver(self):
        kernel = WendlandQuintic(dim=2)

        integrator = EPECIntegrator(fluid=WCSPHStep(), block=RK2StepRigidBody())
        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        tf=tf, dt=dt, adaptive_timestep=False)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                    BodyForce(dest='block', sources=None, gy=gy),
                    NumberDensity(dest='block', sources=['block']),
                    NumberDensity(dest='solid', sources=['solid']),
                    ], ),

            # Equation of state is typically the Tait EOS with a suitable
            # exponent gamma
            Group(equations=[
                    TaitEOS(dest='fluid', sources=None, rho0=rho0, c0=c0, gamma=gamma),
                    TaitEOSHGCorrection(dest='solid', sources=None, rho0=rho0, c0=c0, gamma=gamma),
                    TaitEOSHGCorrection(dest='block', sources=None, rho0=rho0, c0=c0, gamma=gamma),
                    ], ),

            # Main acceleration block
            Group(equations=[

                    # Continuity equation with dissipative corrections for fluid on fluid
                    ContinuityEquationDeltaSPH(dest='fluid', sources=['fluid'], c0=c0, delta=0.1),
                    ContinuityEquation(dest='fluid', sources=['solid', 'block']),
                    ContinuityEquation(dest='solid', sources=['fluid']),
                    ContinuityEquation(dest='block', sources=['fluid']),

                    # Momentum equation
                    MomentumEquation(dest='fluid', sources=['fluid', 'solid', 'block'],
                                     alpha=alpha, beta=beta, gy=-9.81, c0=c0,
                                     tensile_correction=True),

                    PressureRigidBody(dest='fluid', sources=['block', 'solid'], rho0=rho0),
                    ViscosityRigidBody(dest='fluid', sources=['block', 'solid'], rho0=rho0, nu=nu),

                    # Position step with XSPH
                    XSPHCorrection(dest='fluid', sources=['fluid']),

                    RigidBodyCollision(
                        dest='block', sources=['solid'], k=1.0, d=2.0, eta=0.1, kt=0.1
                    ),

                    ]),
            Group(equations=[RigidBodyMoments(dest='block', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='block', sources=None)]),

        ]
        return equations


if __name__ == '__main__':
    app = RigidBlockInTank()
    app.run()
