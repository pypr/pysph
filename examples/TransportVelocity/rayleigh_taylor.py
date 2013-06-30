"""Rayleigh-Taylor instability problem"""

# PyZoltan imports
from pyzoltan.core.carray import LongArray

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import TransportVelocityIntegrator

# the eqations
from pysph.sph.equation import Group
from pyface.sph.wc.basic import BodyForce
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
                fluid.rho[i] = rho1
            else:
                fluid.rho[i] = rho2

        print "Rayleigh Taylor Instability problem :: Re = %d, nfluid = %d, nsolid=%d, dt = %g"%(
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

    # reference densities and pressures
    fluid.add_property( {'name': 'rho0'} )
    fluid.rho0[:] = fluid.rho[:]

    fluid.add_property( {'name': 'p0'} )
    fluid.p0[:] = fluid.rho * c0**2

    # kernel summation correction for the solid
    solid.add_property( {'name': 'wij'} )

    # imopsed velocity on the solid
    solid.add_property( {'name': 'u0'} )
    solid.add_property( {'name': 'v0'} )

    # magnitude of velocity
    fluid.add_property({'name':'vmag'})

    # setup the particle properties
    if not empty:
        volume = dx * dx

        # mass is set to get the reference density of each phase
        fluid.m[:] = volume * fluid.rho

        # volume is set as dx^2
        fluid.V[:] = 1./volume
        solid.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = hdx * dx
        solid.h[:] = hdx * dx

    # return the arrays
    return [fluid, solid]

# Create the application.
app = Application()

# Create the kernel
kernel = Gaussian(dim=2)

# Create a solver.
solver = Solver(
    kernel=kernel, dim=2, integrator_type=TransportVelocityIntegrator)

# Setup default parameters.
solver.set_time_step(dt)
solver.set_final_time(tf)

equations = [

    # Summation density for each phase
    Group(
        equations=[
            DensitySummation(dest='fluid', sources=['fluid','solid']),

            ]),

    # boundary conditions for the solid wall from each phase
    Group(
        equations=[
            SolidWallBC(dest='solid', sources=['fluid'], gy=gy),
            ]),

    # acceleration equations for each phase
    Group(
        equations=[
            StateEquation(dest='fluid', sources=None, b=1.0),

            BodyForce(dest='fluid', sources=None, fy=gy),

            MomentumEquation(dest='fluid', sources=['fluid', 'solid'], nu=nu),

            ArtificialStress(dest='fluid', sources=['fluid',])

            ]),
    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
