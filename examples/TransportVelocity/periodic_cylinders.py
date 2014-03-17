"""Incompressible flow past a periodic array of cylinders"""

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
from pysph.sph.wc.basic import BodyForce
from pysph.sph.wc.transport_velocity import (ArtificialStress,
    DensitySummation, SolidWallBC, VolumeSummation, StateEquation,
    ContinuityEquation, MomentumEquation)

# numpy
import numpy as np

# domain and reference values
L = 0.12; Umax = 1.2e-4
a = 0.02; H = 4*a
fx = 2.5e-4
c0 = 0.1*np.sqrt(a*fx); rho0 = 1000.0
p0 = c0*c0*rho0

# Reynolds number and kinematic viscosity
nu = 0.1/rho0; Re = a*Umax/nu

# Numerical setup
nx = 144; dx = L/nx
ghost_extent = 5 * 1.5 * dx
hdx = 1.05

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0/( c0 + Umax )
dt_viscous = 0.125 * h0**2/nu
dt_force = 0.25 * np.sqrt(h0/abs(fx))

tf = 20.0
dt = 0.5 * min(dt_cfl, dt_viscous, dt_force)

def create_particles(empty=False, **kwargs):
    if empty:
        fluid = get_particle_array(name='fluid')
        solid = get_particle_array(name='solid')
    else:
        # create all the particles
        _x = np.arange( dx/2, L, dx )
        _y = np.arange( -ghost_extent, H+ghost_extent, dx )
        x, y = np.meshgrid(_x, _y); x = x.ravel(); y = y.ravel()

        # sort out the fluid and the solid
        indices = []
        cx = 0.5 * L; cy = 0.5 * H
        for i in range(x.size):
            xi = x[i]; yi = y[i]
            if ( np.sqrt( (xi-cx)**2 + (yi-cy)**2 ) > a ):
                if ( (yi > 0) and (yi < H) ):
                    indices.append(i)

        to_extract = LongArray(len(indices)); to_extract.set_data(np.array(indices))

        # create the arrays
        solid = get_particle_array(name='solid', x=x, y=y)

        # remove the fluid particles from the solid
        fluid = solid.extract_particles(to_extract); fluid.set_name('fluid')
        solid.remove_particles(to_extract)

        print "Periodic cylinders :: Re = %g, nfluid = %d, nsolid=%d, dt = %g"%(
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

    # reference densities and pressures
    fluid.add_property( {'name': 'rho0'} )
    fluid.add_property( {'name': 'p0'} )

    # density acceleration
    fluid.add_property( {'name':'arho'} )

    # magnitude of velocity
    fluid.add_property({'name':'vmag'})

    # setup the particle properties
    if not empty:
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * rho0
        solid.m[:] = volume * rho0
        solid.rho[:] = rho0

        # reference pressures and densities
        fluid.rho0[:] = rho0
        fluid.rho[:] = rho0
        fluid.p0[:] = p0

        # volume is set as dx^2
        fluid.V[:] = 1./volume
        solid.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = hdx * dx
        solid.h[:] = hdx * dx

    # return the particle list
    return [fluid, solid]

# domain for periodicity
domain = DomainLimits(
    xmin=0, xmax=L, periodic_in_x=True)

# Create the application.
app = Application(domain=domain)

# Create the kernel
#kernel = Gaussian(dim=2)
kernel = QuinticSpline(dim=2)

integrator = Integrator(fluid=TransportVelocityStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator, tdamp=0.0)

# Setup default parameters.
solver.set_time_step(dt)
solver.set_final_time(tf)

equations = [

    # State equation
    Group(
        equations=[
            DensitySummation(dest='fluid', source='fluid'),
            DensitySummation(dest='fluid', source='solid'),
            #VolumeSummation(dest='fluid',source='fluid',),
            #VolumeSummation(dest='fluid',source='solid',),
            ]),

    # solid wall bc
    Group(
        equations=[

            SolidWallBC(dest='solid', source='fluid', gx=fx, b=1.0),
            #SolidWallBC(dest='solid', source='fluid', gx=fx, b=0.0),

            ]),

    # accelerations
    Group(
        equations=[
            StateEquation(dest='fluid', source=None, b=1.0),
            BodyForce(dest='fluid', source=None, fx=fx),
            MomentumEquation(dest='fluid', source='fluid', nu=nu),
            MomentumEquation(dest='fluid', source='solid', nu=nu),
            ArtificialStress(dest='fluid', source='fluid')

            ]),
    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
