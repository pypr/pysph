"""Rayleigh-Taylor instability problem"""

# PyZoltan imports
from pyzoltan.core.carray import LongArray

# PySPH imports
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
gy = -1.0
Lx = 1.0; Ly = 2.0
Re = 420; Vmax = np.sqrt(0.5*Ly*abs(gy))
nu = Vmax*Ly/Re

# density for the two phases
rho1 = 1.8; rho2 = 1.0

# speed of sound and reference pressure
Fr = 0.01
c0 = Vmax/Fr
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

    # create the arrays
    solid = get_particle_array(name='solid', x=x, y=y)

    # remove the fluid particles from the solid
    fluid = solid.extract_particles(indices); fluid.set_name('fluid')
    solid.remove_particles(indices)

    # sort out the two fluid phases
    indices = []
    for i in range(fluid.get_number_of_particles()):
        if fluid.y[i] > 1 - 0.15*np.sin(2*np.pi*fluid.x[i]):
            indices.append(i)

    fluid1 = fluid.extract_particles(indices); fluid1.set_name('fluid1')
    fluid2 = fluid
    fluid2.set_name('fluid2')
    fluid2.remove_particles(indices)

    fluid1.rho[:] = rho1
    fluid2.rho[:] = rho2

    print("Rayleigh Taylor Instability problem :: Re = %d, nfluid = %d, nsolid=%d, dt = %g"%(
        Re, fluid1.get_number_of_particles() + fluid2.get_number_of_particles(),
        solid.get_number_of_particles(), dt))

    # add requisite properties to the arrays:
    # particle volume
    fluid1.add_property('V')
    fluid2.add_property('V')
    solid.add_property('V')

    # advection velocities and accelerations
    for name in ('uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 'au', 'av', 'aw'):
        fluid1.add_property(name)
        fluid2.add_property(name)

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
    fluid1.add_property('vmag2')
    fluid2.add_property('vmag2')

    # setup the particle properties
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
    fluid1.set_lb_props( list(fluid1.properties.keys()) )
    fluid2.set_lb_props( list(fluid2.properties.keys()) )
    solid.set_lb_props( list(solid.properties.keys()) )

    # return the arrays
    return [fluid1, fluid2, solid]

# Create the application.
app = Application()

# Create the kernel
kernel = Gaussian(dim=2)

integrator = PECIntegrator(fluid1=TransportVelocityStep(),
                           fluid2=TransportVelocityStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator)

# Setup default parameters.
solver.set_time_step(dt)
solver.set_final_time(tf)

equations = [

    # Summation density for the fluid phase
    Group(
        equations=[
            SummationDensity(dest='fluid1', sources=['fluid1','fluid2', 'solid']),
            SummationDensity(dest='fluid2', sources=['fluid1','fluid2', 'solid']),
            ], real=False),

    # Once the fluid density is computed, we can use the EOS to set
    # the fluid pressure. Additionally, the dummy velocity for the
    # channel is set, which is later used in the no-slip wall BC.
    Group(
        equations=[
            StateEquation(dest='fluid1', sources=None, p0=p1, rho0=rho1, b=1.0),
            StateEquation(dest='fluid2', sources=None, p0=p1, rho0=rho2, b=1.0),
            SetWallVelocity(dest='solid', sources=['fluid1', 'fluid2']),
            SetWallVelocity(dest='solid', sources=['fluid1', 'fluid2']),
            ], real=False),

    # Once the pressure for the fluid phase has been updated, we can
    # extrapolate the pressure to the ghost particles. After this
    # group, the fluid density, pressure and the boundary pressure has
    # been updated and can be used in the integration equations.
    Group(
        equations=[
            SolidWallPressureBC(dest='solid', sources=['fluid1', 'fluid2'], gy=gy, b=1.0,
                                rho0=rho1, p0=p1),
            ], real=False),


    # The main accelerations block. The acceleration arrays for the
    # fluid phase are upadted in this stage for all local particles.
    Group(
        equations=[
            # Pressure gradient terms
            MomentumEquationPressureGradient(
                dest='fluid1', sources=['fluid1', 'fluid2', 'solid'], pb=p1, gy=gy),
            MomentumEquationPressureGradient(
                dest='fluid2', sources=['fluid1', 'fluid2', 'solid'], pb=p2, gy=gy),

            # fluid viscosity
            MomentumEquationViscosity(
                dest='fluid1', sources=['fluid1', 'fluid2'], nu=nu),
            MomentumEquationViscosity(
                dest='fluid2', sources=['fluid1', 'fluid2'], nu=nu),

            # No-slip boundary condition. This is effectively a
            # viscous interaction of the fluid with the ghost
            # particles.
            SolidWallNoSlipBC(
                dest='fluid1', sources=['solid'], nu=nu),

            SolidWallNoSlipBC(
                dest='fluid2', sources=['solid'], nu=nu),

            # Artificial stress for the fluid phase
            MomentumEquationArtificialStress(dest='fluid1', sources=['fluid1', 'fluid2']),
            MomentumEquationArtificialStress(dest='fluid2', sources=['fluid1', 'fluid2']),

            ], real=True),

    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
