"""Two-dimensional Shocktube problem.

The density is assumed to be uniform and the shocktube problem is
defined by the pressure jump. The pressure jump of 10^5 (pl = 1000.0,
pr = 0.01) corresponds to the Woodward and Colella strong shock or
blastwave problem.

"""

# NumPy and standard library imports
import numpy
from numpy import sin, cos, pi

# PySPH base and carray imports
from pysph.base.nnps import DomainManager
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
from pysph.tools import uniform_distribution as ud

# Numerical constants
dim = 2
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 7.5e-6
tf = 0.0075

# domain size
xmin = -0.75; xmax = 0.75
dx = 0.01
ymin = 0; ymax = 30*dx

# scheme constants
alpha1 = 1.0
alpha2 = 0.1
beta = 2.0
kernel_factor = 1.2
h0 = kernel_factor*dx

# The SPH kernel
kernel = Gaussian(dim=dim)

def create_particles(**kwargs):
    global dx
    data = ud.uniform_distribution_cubic2D(dx, xmin, xmax, ymin, ymax)
    
    x = data[0]; y = data[1]
    dx = data[2]; dy = data[3]

    # volume estimate
    volume = dx*dy

    # indices on either side of the initial discontinuity
    right_indices = numpy.where( x > 0.0 )[0]

    # density is uniform
    rho = numpy.ones_like(x)
    
    # pl = 100.0, pr = 0.1
    p = numpy.ones_like(x) * 1000.0
    p[right_indices] = 0.01
    
    # const h and mass
    h = numpy.ones_like(x) * h0
    m = numpy.ones_like(x) * volume * rho

    # thermal energy from the ideal gas EOS
    e = p/(gamma1*rho)
    
    fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m, h0=h.copy())

    print("2D Shocktube with %d particles"%(fluid.get_number_of_particles()))

    return [fluid,]

# Create the application.
domain = DomainManager(
    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, 
    periodic_in_x=True, periodic_in_y=True)

app = Application(domain=domain)

# Create the Integrator.
integrator = PECIntegrator(fluid=GasDFluidStep())

# Create the soliver
solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                dt=dt, tf=tf, adaptive_timestep=False, pfreq=50)


# SPH equations using the Newton-Raphson iterations to determine the
# consistent smoothing length with the density.
equations_density_iterations = [

    ################ BEGIN ADAPTIVE DENSITY-H ###################
    # For the Newton-Raphson, iterative solution for the
    # density-smoothing length relation, we place the SummationDensity
    # Equation in a Group with the iterate argumnet set to True. After
    # this Group, the density and the consitent smoothing length is
    # available for the particles.
    Group(
        equations=[
            SummationDensity(dest='fluid', sources=['fluid',], 
                             k=kernel_factor, density_iterations=True, dim=dim, htol=1e-3),
            ], update_nnps=True, iterate=True, max_iterations=50,
        ),
    ################ END ADAPTIVE DENSITY-H ###################

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

# SPH equations using the GSPH form of adaptive smoothing lengths,
# using a pilot density
equations_pilot_density_adaptive_h = [

    ################ BEGIN ADAPTIVE DENSITY-H ###################
    #For the GSPH density and smoothing length update algorithm, we
    #first scale the smoothing lengths to get the pilot density
    #estimate. Since the particle smoothing lengths are updated, we
    #need to re-compute the neighbors.  
    
    Group( equations=[
           ScaleSmoothingLength(dest='fluid', sources=None, factor=2.0), ],
          update_nnps=True ),

    #Given the new smoothing lengths and (possibly) new neighbors, we
    #compute the pilot density.

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

    #Now that we have the correct smoothing length, we need to
    #evaluate the density which will be used in the
    #accelerations.

    Group(
       equations=[
           SummationDensity(dest='fluid', sources=['fluid',]),
           ], update_nnps=False
       ),
    ################ END ADAPTIVE DENSITY-H ###################

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
app.setup(solver=solver, equations=equations_density_iterations,
          particle_factory=create_particles)

# run the solver
app.run()
