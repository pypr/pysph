"""Taylor Green vortex flow"""

# PyZoltan imports
from pyzoltan.core.carray import LongArray

# PySPH imports
from pysph.base.nnps import DomainLimits
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline, QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import TransportVelocityStep, Integrator

# the eqations
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import DensitySummation,\
    StateEquation, MomentumEquation, ArtificialStress

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

def create_particles(empty=False, **kwargs):
    if empty:
        fluid = get_particle_array(name='fluid')
    else:
        # create the particles
        _x = np.arange( dx/2, L, dx )
        x, y = np.meshgrid(_x, _x); x = x.ravel(); y = y.ravel()
        h = np.ones_like(x) * dx

        # create the arrays
        fluid = get_particle_array(name='fluid', x=x, y=y, h=h)

        # add the requisite arrays
        fluid.add_property( {'name': 'color'} )

        print "Taylor green vortex problem :: nfluid = %d, dt = %g"%(
            fluid.get_number_of_particles(), dt)

        # setup the particle properties
        pi = np.pi; cos = np.cos; sin=np.sin

        # color
        fluid.color[:] = cos(2*pi*x) * cos(4*pi*y)

        # velocities
        fluid.u[:] = -U * cos(2*pi*x) * sin(2*pi*y)
        fluid.v[:] = +U * sin(2*pi*x) * cos(2*pi*y)

        # add requisite properties to the arrays:
        # particle volume
        fluid.add_property( {'name': 'V'} )

        # advection velocities and accelerations
        fluid.add_property( {'name': 'uhat'} )
        fluid.add_property( {'name': 'vhat'} )

        fluid.add_property( {'name': 'auhat'} )
        fluid.add_property( {'name': 'avhat'} )

        fluid.add_property( {'name': 'au'} )
        fluid.add_property( {'name': 'av'} )

        fluid.add_property( {'name': 'vmag'} )

        # mass is set to get the reference density of each phase
        fluid.rho[:] = rho0
        fluid.m[:] = volume * fluid.rho

        # volume is set as dx^2
        fluid.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = hdx * dx

    # load balancing props
    fluid.set_lb_props( fluid.properties.keys() )

    # return the particle list
    return [fluid,]

# domain for periodicity
domain = DomainLimits(xmin=0, xmax=L, ymin=0, ymax=L,
                      periodic_in_x=True, periodic_in_y=True)

# Create the application.
app = Application(domain=domain)

# Create the kernel
kernel = QuinticSpline(dim=2)
#kernel = WendlandQuintic(dim=2)
#kernel = Gaussian(dim=2)

integrator = Integrator(fluid=TransportVelocityStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator)

# Setup default parameters.
solver.set_time_step(dt)
solver.set_final_time(tf)

equations = [

    # density summation
    Group(
        equations=[
            DensitySummation(dest='fluid', sources=['fluid']),
            ]),

    Group(
        equations=[
            StateEquation(dest='fluid', sources=None, rho0=rho0, p0=p0),

            MomentumEquation(dest='fluid', sources=['fluid'], nu=nu, pb=p0),

            ArtificialStress(dest='fluid', sources=['fluid']),

            ]),
    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
