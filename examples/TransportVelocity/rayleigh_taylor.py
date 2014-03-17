"""Rayleigh-Taylor instability problem"""

# PyZoltan imports
from pyzoltan.core.carray import LongArray

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import TransportVelocityStep, Integrator

# the eqations
from pysph.sph.equation import Group
from pysph.sph.wc.basic import BodyForce
from pysph.sph.wc.transport_velocity import DensitySummation,\
    StateEquation, SolidWallBC, MomentumEquation, ArtificialStress

# numpy
import numpy as np

# domain and reference values
gy = -1.0
Lx = 1.0; Ly = 2.0
Re = 420; Vmax = np.sqrt(2*abs(gy)*Ly)
nu = Vmax*Ly/Re

# density for the two phases
rho1 = 1.8; rho2 = 1.0

# speed of sound and reference pressure
c0 = 10 * Vmax
p1 = c0**2 * rho1
p2 = c0**2 * rho2

# Numerical setup
nx = 50; dx = Lx/nx
ghost_extent = 5 * dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0/( c0 + Vmax )
dt_viscous = 0.125 * h0**2/nu
dt_force = 0.25 * np.sqrt(h0/abs(gy))

tf = 25
dt = 0.5 * min(dt_cfl, dt_viscous, dt_force)

def create_particles(empty=False, **kwargs):
    if empty:
        fluid = get_particle_array(name='fluid')
        solid = get_particle_array(name='solid')
    else:
        # create all the particles
        _x = np.arange( -ghost_extent - dx/2, Lx + ghost_extent + dx/2, dx )
        _y = np.arange( -ghost_extent - dx/2, Ly + ghost_extent + dx/2, dx )
        x, y = np.meshgrid(_x, _y); x = x.ravel(); y = y.ravel()

        # sort out the fluid and the solid
        indices = []
        for i in range(x.size):
            if ( (x[i] > 0.0) and (x[i] < Lx) ):
                if ( (y[i] > 0.0)  and (y[i] < Ly) ):
                    indices.append(i)

        to_extract = LongArray(len(indices)); to_extract.set_data(np.array(indices))

        # create the arrays
        solid = get_particle_array(name='solid', x=x, y=y)

        # remove the fluid particles from the solid
        fluid = solid.extract_particles(to_extract); fluid.set_name('fluid')
        solid.remove_particles(to_extract)

        # sort out the two fluid phases
        indices = []
        for i in range(fluid.get_number_of_particles()):
            if fluid.y[i] > 1 - 0.15*np.sin(2*np.pi*fluid.x[i]):
                indices.append(i)

        to_extract = LongArray(len(indices)); to_extract.set_data(np.array(indices))

        fluid1 = fluid.extract_particles(to_extract); fluid1.set_name('fluid1')
        fluid2 = fluid
        fluid2.set_name('fluid2')
        fluid2.remove_particles(to_extract)

        fluid1.rho[:] = rho1
        fluid2.rho[:] = rho2

        print "Rayleigh Taylor Instability problem :: Re = %d, nfluid = %d, nsolid=%d, dt = %g"%(
            Re, fluid1.get_number_of_particles() + fluid2.get_number_of_particles(),
            solid.get_number_of_particles(), dt)

    # add requisite properties to the arrays:
    # particle volume
    fluid1.add_property( {'name': 'V'} )
    fluid2.add_property( {'name': 'V'} )
    solid.add_property( {'name': 'V'} )

    # advection velocities and accelerations
    fluid1.add_property( {'name': 'uhat'} )
    fluid1.add_property( {'name': 'vhat'} )

    fluid2.add_property( {'name': 'uhat'} )
    fluid2.add_property( {'name': 'vhat'} )

    fluid1.add_property( {'name': 'auhat'} )
    fluid1.add_property( {'name': 'avhat'} )

    fluid2.add_property( {'name': 'auhat'} )
    fluid2.add_property( {'name': 'avhat'} )

    fluid1.add_property( {'name': 'au'} )
    fluid1.add_property( {'name': 'av'} )
    fluid1.add_property( {'name': 'aw'} )

    fluid2.add_property( {'name': 'au'} )
    fluid2.add_property( {'name': 'av'} )
    fluid2.add_property( {'name': 'aw'} )

    # kernel summation correction for the solid
    solid.add_property( {'name': 'wij'} )

    # imopsed velocity on the solid
    solid.add_property( {'name': 'u0'} )
    solid.add_property( {'name': 'v0'} )

    # magnitude of velocity
    fluid1.add_property({'name':'vmag'})
    fluid2.add_property({'name':'vmag'})

    # setup the particle properties
    if not empty:
        volume = dx * dx

        # mass is set to get the reference density of each phase
        fluid1.m[:] = volume * rho1
        fluid2.m[:] = volume * rho2

        # volume is set as dx^2
        fluid1.V[:] = 1./volume
        fluid2.V[:] = 1./volume
        solid.V[:] = 1./volume

        # smoothing lengths
        fluid1.h[:] = hdx * dx
        fluid2.h[:] = hdx * dx
        solid.h[:] = hdx * dx

    # load balancing props
    fluid1.set_lb_props( fluid1.properties.keys() )
    fluid2.set_lb_props( fluid2.properties.keys() )
    solid.set_lb_props( solid.properties.keys() )

    # return the arrays
    return [fluid1, fluid2, solid]

# Create the application.
app = Application()

# Create the kernel
kernel = Gaussian(dim=2)

integrator = Integrator(fluid1=TransportVelocityStep(),
                        fluid2=TransportVelocityStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator)

# Setup default parameters.
solver.set_time_step(dt)
solver.set_final_time(tf)

equations = [

    # Summation density for each phase
    Group(
        equations=[
            DensitySummation(dest='fluid1', source='fluid1'),
            DensitySummation(dest='fluid1', source='fluid2'),
            DensitySummation(dest='fluid1', source='solid'),
            DensitySummation(dest='fluid2', source='fluid1'),
            DensitySummation(dest='fluid2', source='fluid2'),
            DensitySummation(dest='fluid2', source='solid'),
            ]),

    # boundary conditions for the solid wall from each phase
    Group(
        equations=[
            SolidWallBC(dest='solid', source='fluid1', gy=gy, rho0=rho1, p0=p1),
            SolidWallBC(dest='solid', source='fluid2', gy=gy, rho0=rho1, p0=p1),
            ]),

    # acceleration equations for each phase
    Group(
        equations=[
            StateEquation(dest='fluid1', source=None, b=1.0, rho0=rho1, p0=p1),
            StateEquation(dest='fluid2', source=None, b=1.0, rho0=rho2, p0=p2),

            BodyForce(dest='fluid1', source=None, fy=gy),
            BodyForce(dest='fluid2', source=None, fy=gy),

            MomentumEquation(dest='fluid1', source='fluid1', nu=nu),
            MomentumEquation(dest='fluid1', source='fluid2', nu=nu),
            MomentumEquation(dest='fluid1', source='solid', nu=nu),
            MomentumEquation(dest='fluid2', source='fluid1', nu=nu),
            MomentumEquation(dest='fluid2', source='fluid2', nu=nu),
            MomentumEquation(dest='fluid2', source='solid', nu=nu),

            ArtificialStress(dest='fluid1', source='fluid1'),
            ArtificialStress(dest='fluid1', source='fluid2'),
            ArtificialStress(dest='fluid2', source='fluid1'),
            ArtificialStress(dest='fluid2', source='fluid2')

            ]),
    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
