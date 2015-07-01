"""Incompressible flow past a periodic lattice of cylinders"""

# PyZoltan imports
from pyzoltan.core.carray import LongArray

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import TransportVelocityStep

# the eqations
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import (SummationDensity,
    SetWallVelocity, StateEquation,
    MomentumEquationPressureGradient, MomentumEquationViscosity,
    MomentumEquationArtificialStress,
    SolidWallPressureBC, SolidWallNoSlipBC)

# numpy
import numpy as np

# domain and reference values
L = 0.1; Umax = 5e-5
c0 = 10 * Umax; rho0 = 1000.0
p0 = c0*c0*rho0
a = 0.02; H = L
fx = 1.5e-7

# Reynolds number and kinematic viscosity
Re = 1.0; nu = a*Umax/Re

# Numerical setup
nx = 100; dx = L/nx
ghost_extent = 5 * 1.5 * dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0/( c0 + Umax )
dt_viscous = 0.125 * h0**2/nu
dt_force = 0.25 * np.sqrt(h0/abs(fx))

tf = 100.0
dt = 0.5 * min(dt_cfl, dt_viscous, dt_force)

def create_particles(**kwargs):
    # create all the particles
    _x = np.arange( dx/2, L, dx )
    _y = np.arange( dx/2, H, dx)
    x, y = np.meshgrid(_x, _y); x = x.ravel(); y = y.ravel()

    # sort out the fluid and the solid
    indices = []
    cx = 0.5 * L; cy = 0.5 * H
    for i in range(x.size):
        xi = x[i]; yi = y[i]
        if ( np.sqrt( (xi-cx)**2 + (yi-cy)**2 ) > a ):
                #if ( (yi > 0) and (yi < H) ):
            indices.append(i)

    # create the arrays
    solid = get_particle_array(name='solid', x=x, y=y)

    # remove the fluid particles from the solid
    fluid = solid.extract_particles(indices); fluid.set_name('fluid')
    solid.remove_particles(indices)

    print("Periodic cylinders :: Re = %g, nfluid = %d, nsolid=%d, dt = %g"%(
        Re, fluid.get_number_of_particles(),
        solid.get_number_of_particles(), dt))

    # add requisite properties to the arrays:
    # particle volume
    fluid.add_property('V')
    solid.add_property('V' )

    # advection velocities and accelerations
    for name in ('uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 'au', 'av', 'aw'):
        fluid.add_property(name)

    # kernel summation correction for the solid
    solid.add_property('wij')

    # imposed accelerations on the solid
    solid.add_property('ax')
    solid.add_property('ay')
    solid.add_property('az')

    # extrapolated velocities for the solid
    for name in ['uf', 'vf', 'wf']:
        solid.add_property(name)

    # dummy velocities for the solid wall
    # required for the no-slip BC
    for name in ['ug', 'vg', 'wg']:
        solid.add_property(name)

    # magnitude of velocity
    fluid.add_property('vmag2')

    # density acceleration
    fluid.add_property('arho')

    # setup the particle properties
    volume = dx * dx

    # mass is set to get the reference density of rho0
    fluid.m[:] = volume * rho0
    solid.m[:] = volume * rho0
    solid.rho[:] = rho0

    # reference pressures and densities
    fluid.rho[:] = rho0

    # volume is set as dx^2
    fluid.V[:] = 1./volume
    solid.V[:] = 1./volume

    # smoothing lengths
    fluid.h[:] = hdx * dx
    solid.h[:] = hdx * dx

    # return the particle list
    return [fluid, solid]

# domain for periodicity
domain = DomainManager(
    xmin=0, xmax=L, ymin=0, ymax=H, periodic_in_x=True,periodic_in_y=True)

# Create the application.
app = Application(domain=domain)

# Create the kernel
kernel = Gaussian(dim=2)

integrator = PECIntegrator(fluid=TransportVelocityStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                dt=dt, tf=tf)

equations = [

    # Summation density along with volume summation for the fluid
    # phase. This is done for all local and remote particles. At the
    # end of this group, the fluid phase has the correct density
    # taking into consideration the fluid and solid
    # particles.
    Group(
        equations=[
            SummationDensity(dest='fluid', sources=['fluid','solid']),
            ], real=False),


    # Once the fluid density is computed, we can use the EOS to set
    # the fluid pressure. Additionally, the dummy velocity for the
    # channel is set, which is later used in the no-slip wall BC.
    Group(
        equations=[
            StateEquation(dest='fluid', sources=None, p0=p0, rho0=rho0, b=1.0),
            SetWallVelocity(dest='solid', sources=['fluid']),
            ], real=False),

    # Once the pressure for the fluid phase has been updated, we can
    # extrapolate the pressure to the ghost particles. After this
    # group, the fluid density, pressure and the boundary pressure has
    # been updated and can be used in the integration equations.
    Group(
        equations=[
            SolidWallPressureBC(dest='solid', sources=['fluid'], gx=fx, b=1.0),
            ], real=False),

    # The main accelerations block. The acceleration arrays for the
    # fluid phase are upadted in this stage for all local particles.
    Group(
        equations=[
            # Pressure gradient terms
            MomentumEquationPressureGradient(
                dest='fluid', sources=['fluid', 'solid'], gx=fx, pb=p0),

            # fluid viscosity
            MomentumEquationViscosity(
                dest='fluid', sources=['fluid'], nu=nu),

            # No-slip boundary condition. This is effectively a
            # viscous interaction of the fluid with the ghost
            # particles.
            SolidWallNoSlipBC(
                dest='fluid', sources=['solid'], nu=nu),

            # Artificial stress for the fluid phase
            MomentumEquationArtificialStress(dest='fluid', sources=['fluid']),

            ], real=True),
    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
