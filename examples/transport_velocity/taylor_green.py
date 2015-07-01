"""Taylor Green vortex flow"""

# PyZoltan imports
from pyzoltan.core.carray import LongArray

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array_tvf_fluid
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline, QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import TransportVelocityStep

# the eqations
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import SummationDensity,\
    StateEquation, MomentumEquationPressureGradient, MomentumEquationViscosity,\
    MomentumEquationArtificialStress, SolidWallPressureBC

# numpy
import numpy as np

# domain and constants
L = 1.0; U = 1.0
Re = 100.0; nu = U*L/Re
rho0 = 1.0; c0 = 10 * U
p0 = c0**2 * rho0
decay_rate = -8.0 * np.pi**2/Re
b = 1.0

# Numerical setup
nx = 50; dx = L/nx; volume = dx*dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0/( c0 + U )
dt_viscous = 0.125 * h0**2/nu
dt_force = 0.25 * 1.0

tf = 5.0
dt = 0.5 * min(dt_cfl, dt_viscous, dt_force)

def create_particles(**kwargs):
    # create the particles
    _x = np.arange( dx/2, L, dx )
    x, y = np.meshgrid(_x, _x); x = x.ravel(); y = y.ravel()
    h = np.ones_like(x) * dx

    # create the arrays
    fluid = get_particle_array_tvf_fluid(name='fluid', x=x, y=y, h=h)

    # add the requisite arrays
    fluid.add_property('color')

    print("Taylor green vortex problem :: nfluid = %d, dt = %g"%(
        fluid.get_number_of_particles(), dt))

    # setup the particle properties
    pi = np.pi; cos = np.cos; sin=np.sin

    # color
    fluid.color[:] = cos(2*pi*x) * cos(4*pi*y)

    # velocities
    fluid.u[:] = -U * cos(2*pi*x) * sin(2*pi*y)
    fluid.v[:] = +U * sin(2*pi*x) * cos(2*pi*y)

    # mass is set to get the reference density of each phase
    fluid.rho[:] = rho0
    fluid.m[:] = volume * fluid.rho

    # volume is set as dx^2
    fluid.V[:] = 1./volume

    # smoothing lengths
    fluid.h[:] = hdx * dx

    # return the particle list
    return [fluid,]

# domain for periodicity
domain = DomainManager(xmin=0, xmax=L, ymin=0, ymax=L,
                       periodic_in_x=True, periodic_in_y=True)

# Create the application.
app = Application(domain=domain)

# Create the kernel
#kernel = QuinticSpline(dim=2)
#kernel = WendlandQuintic(dim=2)
#kernel = Gaussian(dim=2)
kernel = CubicSpline(dim=2)

integrator = PECIntegrator(fluid=TransportVelocityStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator)

# Setup default parameters.
solver.set_time_step(dt)
solver.set_final_time(tf)

equations = [

    # Summation density along with volume summation for the fluid
    # phase. This is done for all local and remote particles. At the
    # end of this group, the fluid phase has the correct density
    # taking into consideration the fluid and solid
    # particles.
    Group(
        equations=[
            SummationDensity(dest='fluid', sources=['fluid']),
            ], real=False),

    # Once the fluid density is computed, we can use the EOS to set
    # the fluid pressure. Additionally, the shepard filtered velocity
    # for the fluid phase is determined.
    Group(
        equations=[
            StateEquation(dest='fluid', sources=None, p0=p0, rho0=rho0, b=1.0),
            ], real=False),

    # The main accelerations block. The acceleration arrays for the
    # fluid phase are upadted in this stage for all local particles.
    Group(
        equations=[
            # Pressure gradient terms
            MomentumEquationPressureGradient(
                dest='fluid', sources=['fluid'], pb=p0),

            # fluid viscosity
            MomentumEquationViscosity(
                dest='fluid', sources=['fluid'], nu=nu),

            # Artificial stress for the fluid phase
            MomentumEquationArtificialStress(dest='fluid', sources=['fluid']),

            ], real=True),

    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
