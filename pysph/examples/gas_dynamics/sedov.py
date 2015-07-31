"""Sedov point explosion problem.

Particles are distributed on concentric circles about the origin with
increasing number of particles with increasing radius. A unit charge
is distributed about the center which gives the initial pressure
disturbance.

"""
# NumPy and standard library imports
import numpy
from numpy import sin, cos, pi

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
    SummationDensity, IdealGasEOS, MPMAccelerations

# PySPH tools
from pysph.tools import ndspmhd

# Numerical constants
dim = 2
gamma = 5.0/3.0
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-4
tf = 0.1

# scheme constants
alpha1 = 1.0
alpha2 = 0.1
beta = 2.0
kernel_factor = 1.2

def create_particles(**kwargs):
    data = numpy.load('ndspmhd-sedov-initial-conditions.npz')
    x = data['x']
    y = data['y']
    
    rho = data['rho']
    p = data['p']
    e = data['e']
    h = data['h']
    m = data['m']

    fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m)    

    # set the initial smoothing length proportional to the particle
    # volume
    fluid.h[:] = kernel_factor * (fluid.m/fluid.rho)**(1./dim)

    print("Sedov's point explosion with %d particles"%(fluid.get_number_of_particles()))

    return [fluid,]

# Create the application.
app = Application()

# Set the SPH kernel
kernel = Gaussian(dim=2)

# Create the Integrator.
integrator = PECIntegrator(fluid=GasDFluidStep())

# Create the soliver
solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                dt=dt, tf=tf, adaptive_timestep=False, pfreq=25)

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
    # do the main acceleratio block.
    Group(
        equations=[
            MPMAccelerations(dest='fluid', sources=['fluid',],
                              alpha1=alpha1, alpha2=alpha2, beta=beta)
            ], update_nnps=False
        ),
    ]

# Setup the application and solver.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

# run the solver
app.run()
