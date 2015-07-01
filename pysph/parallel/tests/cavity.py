"""Lid driven cavity using the Transport Velocity formulation"""

# PyZoltan imports
from pyzoltan.core.carray import LongArray

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline, QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import TransportVelocityStep

# the eqations
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import SummationDensity,\
    StateEquation, MomentumEquationPressureGradient, MomentumEquationViscosity,\
    MomentumEquationArtificialStress, SolidWallPressureBC, SolidWallNoSlipBC,\
    SetWallVelocity

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

    # create the arrays
    solid = get_particle_array(name='solid', x=x, y=y)

    # remove the fluid particles from the solid
    fluid = solid.extract_particles(indices); fluid.set_name('fluid')
    solid.remove_particles(indices)

    print("Lid driven cavity :: Re = %d, nfluid = %d, nsolid=%d, dt = %g"%(
        Re, fluid.get_number_of_particles(),
        solid.get_number_of_particles(), dt)
    )

    # add requisite properties to the arrays:

    # particle volume for fluid and solid
    fluid.add_property('V')
    solid.add_property('V' )

    # extrapolated velocities for the solid
    for name in ['uf', 'vf', 'wf']:
        solid.add_property(name)

    # dummy velocities for the solid wall
    # required for the no-slip BC
    for name in ['ug', 'vg', 'wg']:
        solid.add_property(name)

    # advection velocities and accelerations
    for name in ('uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 'au', 'av', 'aw'):
        fluid.add_property(name)

    # kernel summation correction for the solid
    solid.add_property('wij')

    # imposed accelerations on the solid
    solid.add_property('ax')
    solid.add_property('ay')
    solid.add_property('az')

    # magnitude of velocity
    fluid.add_property('vmag2')

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
    solid.u[:] = 0.0
    solid.v[:] = 0.0
    for i in range(solid.get_number_of_particles()):
        if solid.y[i] > L:
            solid.u[i] = Umax

    # set the output arrays
    fluid.set_output_arrays( ['x', 'y', 'u', 'v', 'vmag2', 'rho', 'p',
                              'V', 'm', 'h', 'gid'] )

    solid.set_output_arrays( ['x', 'y', 'u', 'rho', 'p', 'gid'] )

    return [fluid, solid]

# Create the application.
app = Application()

# Create the kernel
#kernel = QuinticSpline(dim=2)
kernel = CubicSpline(dim=2)

integrator = PECIntegrator(fluid=TransportVelocityStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                tf=tf, dt=dt, adaptive_timestep=False)

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
            SolidWallPressureBC(dest='solid', sources=['fluid'], b=1.0, rho0=rho0, p0=p0),
            ], real=False),

    # The main accelerations block. The acceleration arrays for the
    # fluid phase are upadted in this stage for all local particles.
    Group(
        equations=[
            # Pressure gradient terms
            MomentumEquationPressureGradient(
                dest='fluid', sources=['fluid', 'solid'], pb=p0),

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
