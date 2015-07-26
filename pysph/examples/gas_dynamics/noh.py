"""Example for the Noh's cylindrical implosion test."""

# NumPy and standard library imports
import numpy

# PySPH base and carray imports
from pysph.base.utils import get_particle_array_gasd as gpa
from pysph.base.kernels import CubicSpline, Gaussian

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import GasDFluidStep

# PySPH sph imports
from pysph.sph.equation import Group
from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength, UpdateSmoothingLengthFromVolume,\
    SummationDensity, IdealGasEOS, MPMAccelerations, Monaghan92Accelerations

# problem constants
dim = 2
gamma = 5.0/3.0
gamma1 = gamma - 1.0

# scheme constants
alpha1 = 1.0
alpha2 = 0.1
beta = 2.0
kernel_factor = 1.2

# numerical constants
dt = 1e-4
tf = 1.0

# domain and particle spacings
xmin = ymin = -0.5
xmax = ymax = 0.5

nx = ny = 50
dx = (xmax-xmin)/nx
dxb2 = 0.5 * dx

# initial values
h0 = kernel_factor*dx
rho0 = 1.0
e0 = 1e-6
m0 = dx*dx * rho0
vr = -1.0

def create_particles(**kwargs):

    x, y = numpy.mgrid[
        xmin:xmax:dx, ymin:ymax:dx]

    # positions
    x = x.ravel(); y = y.ravel()

    rho = numpy.ones_like(x) * rho0
    m = numpy.ones_like(x) * m0
    e = numpy.ones_like(x) * e0
    h = numpy.ones_like(x) * h0
    p = gamma1*rho*e

    u = numpy.ones_like(x)
    v = numpy.ones_like(x)

    sin, cos, arctan = numpy.sin, numpy.cos, numpy.arctan2
    for i in range(x.size):
        theta = arctan(y[i],x[i])
        u[i] = vr*cos(theta)
        v[i] = vr*sin(theta)

    fluid = gpa(name='fluid', x=x,y=y,m=m,rho=rho, h=h,u=u,v=v,p=p,e=e)

    print("Noh's problem with %d particles"%(fluid.get_number_of_particles()))

    return [fluid,]

# Create the application.
app = Application()

# Set the SPH kernel
kernel = Gaussian(dim=2)

# Create the Integrator.
integrator = PECIntegrator(fluid=GasDFluidStep())

# Create the soliver
solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                dt=dt, tf=tf, adaptive_timestep=False)

# Define the SPH equations
equations = [

    # Scale smoothing length. Since the particle smoothing lengths are
    # updated, we need to re-compute the neighbors
    Group(
        equations=[
            ScaleSmoothingLength(dest='fluid', sources=None, factor=2.0),
            ], update_nnps=True
        ),

    # Given the new smoothing lengths and (possibly) new neighbors, we
    # compute the pilot density.
    Group(
        equations=[
            SummationDensity(dest='fluid', sources=['fluid',]),
            ], update_nnps=False
        ),

    # Once the pilot density has been computed, we can update the
    # smoothing length from the new estimate of particle volume. Once
    # again, the NNPS must be updated to reflect the updated smoothing
    # lengths
    Group(
        equations=[
            UpdateSmoothingLengthFromVolume(
                dest='fluid', sources=None, k=kernel_factor, dim=dim),
            ], update_nnps=True
        ),

    # Now that we have the correct smoothing length, we need to
    # evaluate the density which will be used in the
    # accelerations.
    Group(
        equations=[
            SummationDensity(dest='fluid', sources=['fluid',]),
            ], update_nnps=False
        ),

    # The equation of state is also done now to update the particle
    # pressure and sound speeds.
    Group(
        equations=[
            IdealGasEOS(dest='fluid', sources=None, gamma=gamma),
            ], update_nnps=False
        ),

    # Now that we have the density, pressure and sound speeds, we can
    # do the main acceleration block.
    Group(
        equations=[
            MPMAccelerations(dest='fluid', sources=['fluid',],
                              alpha1=alpha1, alpha2=alpha2, beta=beta)
            #Monaghan92Accelerations(dest='fluid', sources=['fluid',],
            #                        alpha=1.0, beta=2.0)
            ], update_nnps=False
        ),
    ]

# Setup the application and solver.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

# run the solver
app.run()
