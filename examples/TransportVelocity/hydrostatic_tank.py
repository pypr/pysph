"""Hydrostatic tank examples (Section 6.0) of Adami et. al. JCP 231, 7057-7075"""

# PyZoltan imports
from pyzoltan.core.carray import LongArray

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline,\
    QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import Integrator, TransportVelocityStep,\
    AdamiVerletStep

# the eqations
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import SummationDensity, \
    ContinuityEquation, VolumeSummation, VolumeFromMassDensity, \
    StateEquation, MomentumEquationPressureGradient, \
    MomentumEquationArtificialViscosity, MomentumEquationViscosity,\
    MomentumEquationArtificialStress, SolidWallPressureBC,\
    ShepardFilteredVelocity

# numpy
import numpy as np

# domain and reference values
Lx = 2.0; Ly = 1.0; H = 0.9
gy = -1.0
Vmax = np.sqrt(abs(gy) * H)
c0 = 10 * Vmax; rho0 = 1000.0
p0 = c0*c0*rho0

# Reynolds number and kinematic viscosity
Re = 100; nu = Vmax * Ly/Re

# Numerical setup
nx = 100; dx = Lx/nx
ghost_extent = 5.5 * dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0/( c0 + Vmax )
dt_viscous = 0.125 * h0**2/nu
dt_force = 0.25 * np.sqrt(h0/abs(gy))

tf = 2.0
dt = 0.75 * min(dt_cfl, dt_viscous, dt_force)

def create_particles(**kwargs):
    # create all the particles
    _x = np.arange( -ghost_extent, Lx + ghost_extent, dx )
    _y = np.arange( -ghost_extent, Ly, dx )
    x, y = np.meshgrid(_x, _y); x = x.ravel(); y = y.ravel()

    # sort out the fluid and the solid
    indices = []
    for i in range(x.size):
        if ( (x[i] > 0.0) and (x[i] < Lx) ):
            if ( (y[i] > 0.0) and (y[i] < H) ):
                indices.append(i)
                
    # create the arrays
    solid = get_particle_array(name='solid', x=x, y=y)

    # remove the fluid particles from the solid
    fluid = solid.extract_particles(indices); fluid.set_name('fluid')
    solid.remove_particles(indices)

    # remove the lid to generate an open tank
    indices = []
    for i in range(solid.get_number_of_particles()):
        if solid.y[i] > 0.9:
            if (0 < solid.x[i] < Lx):
                indices.append(i)
    solid.remove_particles(indices)

    print "Hydrostatic tank :: nfluid = %d, nsolid=%d, dt = %g"%(
        fluid.get_number_of_particles(),
        solid.get_number_of_particles(), dt)

    # add requisite properties to the arrays:
    # particle volume
    fluid.add_property('V')
    solid.add_property('V' )

    # advection velocities and accelerations
    for name in ('uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 'au', 'av', 'aw'):
        fluid.add_property(name)

    # kernel summation correction for the solid
    solid.add_property('wij')

    # density acceleration for fluid
    fluid.add_property('arho')

    # imposed velocity on the solid
    solid.add_property('u0'); solid.u0[:] = 0.
    solid.add_property('v0'); solid.v0[:] = 0.
    solid.add_property('w0'); solid.w0[:] = 0.

    # imposed accelerations on the solid
    solid.add_property('ax')
    solid.add_property('ay')
    solid.add_property('az')

    # Shepard filtered velocities for the fluid
    for name in ['uf', 'vf', 'wf']:
        fluid.add_property(name)

    # magnitude of velocity
    fluid.add_property('vmag')

    # setup the particle properties
    volume = dx * dx

    # densty is set to rho0
    fluid.rho[:] = rho0
    solid.rho[:] = rho0
    
    # mass is set to get the reference density of rho0
    fluid.m[:] = volume * rho0
    solid.m[:] = volume * rho0

    # volume is set as dx^2
    fluid.V[:] = 1./volume
    solid.V[:] = 1./volume

    # smoothing lengths
    fluid.h[:] = hdx * dx
    solid.h[:] = hdx * dx

    # return the particle list
    return [fluid, solid]

# Create the application.
app = Application()

# Create the kernel
#kernel = Gaussian(dim=2)
kernel = QuinticSpline(dim=2)

integrator = Integrator(fluid=AdamiVerletStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                tf=tf, dt=dt)

equations = [


    # Summation density along with volume summation for the fluid
    # phase. This is done for all local and remote particles. At the
    # end of this group, the fluid phase has the correct density
    # taking into consideration the fluid and solid
    # particles. 
    Group(
        equations=[
            VolumeFromMassDensity(dest='fluid', sources=None),
            StateEquation(dest='fluid', sources=None, p0=p0, rho0=rho0, b=1.0),
            ShepardFilteredVelocity(dest='fluid', sources=['fluid']),
            ], real=False),

    # Once the pressure for the fluid phase has been updated, we can
    # extrapolate the pressure to the ghost particles. After this
    # group, the fluid density, pressure and the boundary pressure has
    # been updated and can be used in the integration equations.
    Group(
        equations=[
            SolidWallPressureBC(dest='solid', sources=['fluid'], b=1.0, gy=gy, 
                                rho0=rho0, p0=p0),
            ], real=False),

    # The main accelerations block. The acceleration arrays for the
    # fluid phase are upadted in this stage for all local particles.
    Group(
        equations=[
            # continuityequation
            ContinuityEquation(dest='fluid', sources=['fluid','solid']),

            # Pressure gradient terms
            MomentumEquationPressureGradient(
                dest='fluid', sources=['fluid', 'solid'], pb=0.0, gy=gy),
            
            # fluid viscosity
            #MomentumEquationArtificialViscosity(
            #    dest='fluid', sources=['fluid'], alpha=0.24),
            MomentumEquationViscosity(
                dest='fluid', sources=['fluid'], nu=nu),
            
            ], real=True),
    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
