"""Hydrostatic tank example (Section 6.0) of Adami et. al. JCP 231, 7057-7075

Different boundary formulations can be used to check for this behaviour:

 - Adami et al. "A generalized wall boundary condition for smoothed
   particle hydrodynamics", 2012, JCP, 231, pp 7057--7075 (REF1)

 - Monaghan and Kajtar, "SPH particle boundary forces for arbitrary
   boundaries", 2009, 180, pp 1811--1820 (REF2)

 - Gesteria et al. "State-of-the-art of classical SPH for free-surface
   flows", 2010, JHR, pp 6--27 (REF3)

Of these, the first and third are ghost particle methods while the
second is the classical Monaghan style, repulsive particle approach.

We have a block of rigid solid falling on this tank.


"""
# NumPy
import numpy as np

# PySPH imports
from pysph.base.utils import get_particle_array_wcsph as gpa
from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline, QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import WCSPHStep

# the eqations
from pysph.sph.equation import Group

# Equations for REF1
from pysph.sph.wc.transport_velocity import VolumeFromMassDensity,\
    ContinuityEquation,\
    MomentumEquationPressureGradient, \
    MomentumEquationArtificialViscosity,\
    SolidWallPressureBC, SummationDensity

# Equations for the standard WCSPH formulation and dynamic boundary
# conditions defined in REF3
from pysph.sph.wc.basic import TaitEOS
from pysph.sph.basic_equations import XSPHCorrection

from pysph.sph.rigid_body import (BodyForce, NumberDensity,
    RigidBodyMoments, RigidBodyMotion,
    RK2StepRigidBody, ViscosityRigidBody, PressureRigidBody,
    SummationDensityRigidBody)

# domain and reference values
Lx = 2.0; Ly = 1.0; H = 0.7
gy = -1.0
Vmax = np.sqrt(abs(gy) * H)
c0 = 10 * Vmax; rho0 = 1000.0
p0 = c0*c0*rho0
gamma = 1.0

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

tf = 3.0
dt = 0.5 * min(dt_cfl, dt_viscous, dt_force)

def create_particles(**kwargs):
    side = 0.1

    _x = np.arange(Lx*0.5-side , Lx*0.5 + side, dx )
    _y = np.arange( 0.8, 0.8+side*2, dx )
    x, y = np.meshgrid(_x, _y); x = x.ravel(); y = y.ravel()
    rho_block = rho0*0.5
    m = np.ones_like(x)*dx*dx*rho_block
    h = np.ones_like(x)*hdx*dx
    rho = np.ones_like(x)*rho_block
    block = get_particle_array_rigid_body(
        name='block', x=x, y=y, h=h, m=m,
    )

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
    solid = gpa(name='solid', x=x, y=y)

    # remove the fluid particles from the solid
    fluid = solid.extract_particles(indices); fluid.set_name('fluid')
    solid.remove_particles(indices)

    # remove the lid to generate an open tank
    indices = []
    for i in range(solid.get_number_of_particles()):
        if solid.y[i] > H:
            if (0 < solid.x[i] < Lx):
                indices.append(i)
    solid.remove_particles(indices)

    print "Hydrostatic tank :: nfluid = %d, nsolid=%d, dt = %g"%(
        fluid.get_number_of_particles(),
        solid.get_number_of_particles(), dt)

    ###### ADD PARTICLE PROPS FOR MULTI-PHASE SPH ######

    # particle volume
    fluid.add_property('V')
    solid.add_property('V' )

    # kernel sum term for boundary particles
    solid.add_property('wij')

    # advection velocities and accelerations
    for name in ('auhat', 'avhat', 'awhat'):
        fluid.add_property(name)

    ##### INITIALIZE PARTICLE PROPS #####
    fluid.rho[:] = rho0
    solid.rho[:] = rho0

    fluid.rho0[:] = rho0
    solid.rho0[:] = rho0

    # mass is set to get the reference density of rho0
    volume = dx * dx

    # volume is set as dx^2
    fluid.V[:] = 1./volume
    solid.V[:] = 1./volume

    fluid.m[:] = volume * rho0
    solid.m[:] = volume * rho0

    # smoothing lengths
    fluid.h[:] = hdx * dx
    solid.h[:] = hdx * dx

    # return the particle list
    return [fluid, solid, block]

# Create the application.
app = Application()

# Create the kernel
#kernel = Gaussian(dim=2)
kernel = QuinticSpline(dim=2)

integrator = PECIntegrator(fluid=WCSPHStep(), block=RK2StepRigidBody())

# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                tf=tf, dt=dt, adaptive_timestep=False)


# Formulation for REF1
equations1 = [
    # For the multi-phase formulation, we require an estimate of the
    # particle volume. This can be either defined from the particle
    # number density or simply as the ratio of mass to density.
    Group(equations=[
            BodyForce(dest='block', sources=None, gy=gy),
            SummationDensity(dest='fluid', sources=['fluid', 'solid']),
            NumberDensity(dest='block', sources=['block']),
            ], ),

    Group(equations=[
        SummationDensityRigidBody(dest='fluid', sources=['block'], rho0=rho0)
        ]),

    # Equation of state is typically the Tait EOS with a suitable
    # exponent gamma
    Group(equations=[
            TaitEOS(dest='fluid', sources=None, rho0=rho0, c0=c0, gamma=gamma),
            ], ),

    # The boundary conditions are imposed by extrapolating the fluid
    # pressure, taking into considering the bounday acceleration
    Group(equations=[
            SolidWallPressureBC(dest='solid', sources=['fluid'], b=1.0, gy=gy,
                                rho0=rho0, p0=p0),
            ], ),

    # Main acceleration block
    Group(equations=[

            # Pressure gradient with acceleration damping.
            MomentumEquationPressureGradient(
                dest='fluid', sources=['fluid', 'solid'], pb=0.0, gy=gy,
                tdamp=1.0),

            # artificial viscosity for stability
            MomentumEquationArtificialViscosity(
                dest='fluid', sources=['fluid', 'solid'], alpha=0.25, c0=c0),

            PressureRigidBody(dest='fluid', sources=['block'], rho0=rho0),
            ViscosityRigidBody(dest='fluid', sources=['block'], rho0=rho0, nu=nu),

            # Position step with XSPH
            XSPHCorrection(dest='fluid', sources=['fluid'], eps=0.0)

            ]),
    Group(equations=[RigidBodyMoments(dest='block', sources=None)]),
    Group(equations=[RigidBodyMotion(dest='block', sources=None)]),

    ]


# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations1,
          particle_factory=create_particles)

app.run()
