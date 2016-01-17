"""SPHERIC benchmark case 6. (2 hours)

See https://wiki.manchester.ac.uk/spheric/index.php/Test6 for more details.
"""

# math
from math import exp

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application

from pysph.sph.integrator_step import TwoStageRigidBodyStep, TransportVelocityStep
from pysph.sph.integrator import Integrator

from pysph.tools import uniform_distribution

# SPH equations for this problem
from pysph.sph.equation import Group, Equation
from pysph.sph.wc.transport_velocity import SummationDensity,\
    StateEquation, MomentumEquationPressureGradient, MomentumEquationViscosity,\
    MomentumEquationArtificialStress, SolidWallPressureBC, SolidWallNoSlipBC,\
    SetWallVelocity

# domain and reference values
Lx = 10.0; Ly = 5.0
Umax = 1.0
c0 = 25.0 * Umax; rho0 = 1.0
p0 = c0*c0*rho0

# obstacle dimensions
obstacle_width = 1.0
obstacle_height = 1.0

# Reynolds number and kinematic viscosity
Re = 150; nu = Umax * obstacle_width/Re

# Numerical setup
nx = 50; dx = 0.20* Lx/nx
nghost_layers = 4
ghost_extent = nghost_layers * dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0/( c0 + Umax )
dt_viscous = 0.125 * h0**2/nu
dt_force = 1.0

tf = 8.0
dt = 0.8 * min(dt_cfl, dt_viscous, dt_force)

class SPHERICBenchmarkAcceleration(Equation):
    r"""Equation to set the acceleration for the moving square
    benchmark problem.

    We use scipy.optimize to fit the Gaussian:

    .. math::

        a \exp( -\frac{(t-b)^2}{2c^2} ) + d

    to the SPHERIC Motion.dat file. The values for the parameters are

    a = 2.8209512
    b = 0.525652151
    c = 0.14142151
    d = -2.55580905e-08

    Notes:

    This equation must be instantiated with no sources

    """
    def loop(self, d_idx, d_ax, t=0.0):
        a = 2.8209512
        b = 0.525652151
        c = 0.14142151
        d = -2.55580905e-08

        # compute the acceleration and set it for the destination
        d_ax[d_idx] = a*exp( -(t-b)*(t-b)/(2.0*c*c) ) + d


def _get_interior(x, y):
    indices = []
    for i in range(x.size):
        if ( (x[i] > 0.0) and (x[i] < Lx) ):
            if ( (y[i] > 0.0) and (y[i] < Ly) ):
                indices.append(i)

    return indices

def _get_obstacle(x, y):
    indices = []
    for i in range(x.size):
        if ( (1.0 <= x[i] <= 2.0) and (2.0 <= y[i] <= 3.0) ):
            indices.append(i)

    return indices


class MovingSquare(Application):
    def _setup_particle_properties(self, particles, volume):
        fluid, solid, obstacle = particles

        #### ADD PROPS FOR THE PARTICLES ###

        # volume from number density
        fluid.add_property('V')
        solid.add_property('V' )
        obstacle.add_property('V' )

        # extrapolated velocities for the fluid
        for name in ['uf', 'vf', 'wf']:
            solid.add_property(name)
            obstacle.add_property(name)

        # dummy velocities for the solid and obstacle
        # required for the no-slip BC
        for name in ['ug', 'vg', 'wg']:
            solid.add_property(name)
            obstacle.add_property(name)

        # advection velocities and accelerations for fluid
        for name in ('uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 'au', 'av', 'aw'):
            fluid.add_property(name)

        # kernel summation correction for solids
        solid.add_property('wij')
        obstacle.add_property('wij')

        # initial velocities and positions needed for the obstacle for
        # rigid-body integration
        obstacle.add_property('u0'); obstacle.u0[:] = 0.
        obstacle.add_property('v0'); obstacle.v0[:] = 0.
        obstacle.add_property('w0'); obstacle.w0[:] = 0.

        obstacle.add_property('x0')
        obstacle.add_property('y0')
        obstacle.add_property('z0')

        # imposed accelerations on the solid and obstacle
        solid.add_property('ax')
        solid.add_property('ay')
        solid.add_property('az')

        obstacle.add_property('ax')
        obstacle.add_property('ay')
        obstacle.add_property('az')

        # magnitude of velocity squared
        fluid.add_property('vmag2')

        #### SETUP PARTICLE PROPERTIES ###

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * rho0
        solid.m[:] = volume * rho0
        obstacle.m[:] = volume * rho0

        # volume is set as dx^2
        fluid.V[:] = 1./volume
        solid.V[:] = 1./volume
        obstacle.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = h0
        solid.h[:] = h0
        obstacle.h[:] = h0

        # set the output arrays
        fluid.set_output_arrays( ['x', 'y', 'u', 'v', 'vmag2', 'rho', 'p',
                                  'V', 'm', 'h'] )

        solid.set_output_arrays( ['x', 'y', 'rho', 'p'] )
        obstacle.set_output_arrays( ['x', 'y', 'u0', 'rho', 'p'] )

        particles = [fluid, solid, obstacle]
        return particles

    def add_user_options(self, group):
        group.add_argument(
            "--hcp", action="store_true", dest="hcp", default=False,
            help="Use hexagonal close packing of particles."
        )

    def create_particles(self):

        hcp = self.options.hcp
        #Initial distribution using Hexagonal close packing of particles
        # create all particles
        global dx
        if hcp:
            x, y, dx, dy, xmin, xmax, ymin, ymax = uniform_distribution.uniform_distribution_hcp2D(
                dx=dx, xmin=-ghost_extent, xmax=Lx+ghost_extent,
                ymin=-ghost_extent, ymax=Ly+ghost_extent)
        else:
            x, y, dx, dy, xmin, xmax, ymin, ymax = uniform_distribution.uniform_distribution_cubic2D(
                dx=dx, xmin=-ghost_extent, xmax=Lx+ghost_extent,
                ymin=-ghost_extent, ymax=Ly+ghost_extent)

        x = x.ravel(); y = y.ravel()

        # create the basic particle array
        solid = get_particle_array(name='solid', x=x, y=y)

        # now sort out the interior from all particles
        indices = _get_interior(solid.x, solid.y)
        fluid = solid.extract_particles( indices )
        fluid.set_name('fluid')

        solid.remove_particles( indices )

        # sort out the obstacle from the interior
        indices = _get_obstacle(fluid.x, fluid.y)
        obstacle = fluid.extract_particles( indices )
        obstacle.set_name('obstacle')

        fluid.remove_particles(indices)

        print("SPHERIC benchmark 6 :: Re = %d, nfluid = %d, nsolid=%d, nobstacle = %d, dt = %g"%(
            Re, fluid.get_number_of_particles(),
            solid.get_number_of_particles(),
            obstacle.get_number_of_particles(), dt))

        # setup requisite particle properties and initial conditions

        if hcp:
            wij_sum = uniform_distribution.get_number_density_hcp(dx, dy, kernel, h0)
            volume = 1./wij_sum
        else:
            volume = dx*dy

        particles = self._setup_particle_properties(
            [fluid, solid, obstacle], volume=volume
        )

        return particles

    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = Integrator(fluid=TransportVelocityStep(),
                                obstacle=TwoStageRigidBodyStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        tf=tf, dt=dt, adaptive_timestep=False,
                        output_at_times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        return solver

    def create_equations(self):
        equations = [

            # set the acceleration for the obstacle using the special function
            # mimicking the accelerations provided in the test.
            Group(
                equations=[
                    SPHERICBenchmarkAcceleration(dest='obstacle', sources=None),
                    ], real=False),

            # Summation density along with volume summation for the fluid
            # phase. This is done for all local and remote particles. At the
            # end of this group, the fluid phase has the correct density
            # taking into consideration the fluid and solid
            # particles.
            Group(
                equations=[
                    SummationDensity(dest='fluid', sources=['fluid','solid','obstacle']),
                    ], real=False),

            # Once the fluid density is computed, we can use the EOS to set
            # the fluid pressure. Additionally, the dummy velocity for the
            # channel is set, which is later used in the no-slip wall BC.
            Group(
                equations=[
                    StateEquation(dest='fluid', sources=None, p0=p0, rho0=rho0, b=1.0),
                    SetWallVelocity(dest='solid', sources=['fluid']),
                    SetWallVelocity(dest='obstacle', sources=['fluid']),
                    ], real=False),

            # Once the pressure for the fluid phase has been updated, we can
            # extrapolate the pressure to the ghost particles. After this
            # group, the fluid density, pressure and the boundary pressure has
            # been updated and can be used in the integration equations.
            Group(
                equations=[
                    SolidWallPressureBC(dest='obstacle', sources=['fluid'], b=1.0, rho0=rho0, p0=p0),
                    SolidWallPressureBC(dest='solid', sources=['fluid'], b=1.0, rho0=rho0, p0=p0),
                    ], real=False),

            # The main accelerations block. The acceleration arrays for the
            # fluid phase are upadted in this stage for all local particles.
            Group(
                equations=[
                    # Pressure gradient terms
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid', 'solid','obstacle'], pb=p0),

                    # fluid viscosity
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid'], nu=nu),

                    # No-slip boundary condition. This is effectively a
                    # viscous interaction of the fluid with the ghost
                    # particles.
                    SolidWallNoSlipBC(
                        dest='fluid', sources=['solid','obstacle'], nu=nu),

                    # Artificial stress for the fluid phase
                    MomentumEquationArtificialStress(dest='fluid', sources=['fluid']),

                    ], real=True),
        ]
        return equations


if __name__ == '__main__':
    app = MovingSquare()
    app.run()
