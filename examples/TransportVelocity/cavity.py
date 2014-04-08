"""Lid driven cavity using the Transport Velocity formulation"""

# PyZoltan imports
from pyzoltan.core.carray import LongArray

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline, QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import TransportVelocityStep, Integrator

# the eqations
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import DensitySummation,\
    StateEquation, SolidWallBC, MomentumEquation, ArtificialStress

# numpy
import numpy as np

# domain and reference values
L = 1.0; Umax = 1.0
c0 = 10 * Umax; rho0 = 1.0
p0 = c0*c0*rho0

# Reynolds number and kinematic viscosity
Re = 100; nu = Umax * L/Re

# Numerical setup
nx = 50; dx = L/nx
ghost_extent = 5 * dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0/( c0 + Umax )
dt_viscous = 0.125 * h0**2/nu
dt_force = 1.0

tf = 5.0
dt = 0.75 * min(dt_cfl, dt_viscous, dt_force)

def create_particles(**kwargs):
    # create all the particles
    _x = np.arange( -ghost_extent - dx/2, L + ghost_extent + dx/2, dx )
    x, y = np.meshgrid(_x, _x); x = x.ravel(); y = y.ravel()

    # sort out the fluid and the solid
    indices = []
    for i in range(x.size):
        if ( (x[i] > 0.0) and (x[i] < L) ):
            if ( (y[i] > 0.0) and (y[i] < L) ):
                indices.append(i)

    to_extract = LongArray(len(indices)); to_extract.set_data(np.array(indices))
    
    # create the arrays
    solid = get_particle_array(name='solid', x=x, y=y)

    # remove the fluid particles from the solid
    fluid = solid.extract_particles(to_extract); fluid.set_name('fluid')
    solid.remove_particles(to_extract)

    print "Lid driven cavity :: Re = %d, nfluid = %d, nsolid=%d, dt = %g"%(
        Re, fluid.get_number_of_particles(),
        solid.get_number_of_particles(), dt)

    # add requisite properties to the arrays:
    # particle volume
    fluid.add_property( {'name': 'V'} )
    solid.add_property( {'name': 'V'} )

    # advection velocities and accelerations
    fluid.add_property( {'name': 'uhat'} )
    fluid.add_property( {'name': 'vhat'} )

    fluid.add_property( {'name': 'auhat'} )
    fluid.add_property( {'name': 'avhat'} )

    fluid.add_property( {'name': 'au'} )
    fluid.add_property( {'name': 'av'} )
    fluid.add_property( {'name': 'aw'} )

    # kernel summation correction for the solid
    solid.add_property( {'name': 'wij'} )

    # imopsed velocity on the solid
    solid.add_property( {'name': 'u0'} )
    solid.add_property( {'name': 'v0'} )

    # magnitude of velocity
    fluid.add_property({'name':'vmag'})

    # setup the particle properties
    volume = dx * dx
    
    # mass is set to get the reference density of rho0
    fluid.m[:] = volume * rho0
    solid.m[:] = volume * rho0
    
    # volume is set as dx^2
    fluid.V[:] = 1./volume
    solid.V[:] = 1./volume

    # smoothing lengths
    fluid.h[:] = hdx * dx
    solid.h[:] = hdx * dx

    # imposed horizontal velocity on the lid
    solid.u0[:] = 0.0
    solid.v0[:] = 0.0
    for i in range(solid.get_number_of_particles()):
        if solid.y[i] > L:
            solid.u0[i] = Umax

    # return the particle list
    fluid.set_lb_props( fluid.properties.keys() )
    solid.set_lb_props( solid.properties.keys() )

    return [fluid, solid]

# Create the application.
app = Application()

# Create the kernel
kernel = QuinticSpline(dim=2)

integrator = Integrator(fluid=TransportVelocityStep())
# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator)

# Setup default parameters.
solver.set_time_step(dt)
solver.set_final_time(tf)

equations = [

    # Summation density for the fluid phase
    Group(
        equations=[
            DensitySummation(dest='fluid', sources=['fluid','solid'],)

            ]),

    # boundary conditions for the solid wall
    Group(
        equations=[

            SolidWallBC(dest='solid', sources=['fluid',], b=1.0, rho0=rho0, p0=p0),

            ]),

    # acceleration equation
    Group(
        equations=[
            StateEquation(dest='fluid', sources=None, b=1.0, rho0=rho0, p0=p0),

            MomentumEquation(dest='fluid', sources=['fluid', 'solid'],
                             nu=nu, pb=p0),

            ArtificialStress(dest='fluid', sources=['fluid',]),

            ]),
    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
