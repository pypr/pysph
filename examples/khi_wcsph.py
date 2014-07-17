""" 2D Kelvin Helmoltz Instability example """
import numpy

# Particle generator
from pysph.base.utils import get_particle_array

# SPH kernels
from pysph.base.kernels import CubicSpline, WendlandQuintic, Gaussian

# SPH Equations and Group
from pysph.sph.equation import Group

from pysph.sph.basic_equations import IsothermalEOS, XSPHCorrection
from pysph.sph.wc.basic import PressureGradientUsingNumberDensity
from pysph.sph.wc.viscosity import ClearyArtificialViscosity
from pysph.sph.wc.transport_velocity import SummationDensity, MomentumEquationPressureGradient,\
    SolidWallPressureBC, SolidWallNoSlipBC, ShepardFilteredVelocity
from pysph.sph.surface_tension import ColorGradientUsingNumberDensity, \
    InterfaceCurvatureFromNumberDensity, ShadlooYildizSurfaceTensionForce

# PySPH solver and application
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# Integrators and Steppers
from pysph.sph.integrator_step import VerletSymplecticWCSPHStep
from pysph.sph.integrator import PECIntegrator

# Domain manager for periodic domains
from pysph.base.nnps import DomainManager

# problem parameters
dim = 2
domain_width = 1.0
domain_height = 1.0

# numerical constants
dt = 5e-5
tf = 1.0
alpha = 0.001
sigma = 0.03

# discretization parameters
nghost_layers = 5
dx = dy = 0.01
dxb2 = dyb2 = 0.5 * dx
volume = dx*dx
hdx = 1.3
h0 = hdx * dx
rho0 = 1000.0
c0 = 10.0
p0 = c0*c0*rho0
nu = 0.125 * alpha * h0 * c0

def create_particles(**kwargs):
    ghost_extent = (nghost_layers + 0.5)*dx

    x, y = numpy.mgrid[ dxb2:domain_width:dx, -ghost_extent:domain_height+ghost_extent:dy ]
    x = x.ravel(); y = y.ravel()

    m = numpy.ones_like(x) * volume * rho0
    rho = numpy.ones_like(x) * rho0
    h = numpy.ones_like(x) * h0
    cs = numpy.ones_like(x) * c0

    # additional properties required for the fluid.
    additional_props = ['V', 'color', 'cx', 'cy', 'cz', 'nx', 'ny', 'nz',
                        'ddelta', 'uf', 'vf', 'wf', 'auhat', 'avhat', 'awhat',
                        'ax', 'ay', 'az', 'wij', 'kappa']

    # get the fluid particle array
    fluid = get_particle_array(
        name='fluid', x=x, y=y, h=h, m=m, rho=rho, cs=cs, 
        additional_props=additional_props)

    # set the fluid velocity with respect to the sinusoidal
    # perturbation
    fluid.u[:] = -0.5
    mode = 1
    for i in range(len(fluid.x)):
        if fluid.y[i] > domain_height/2 + sigma*domain_height*numpy.sin(2*numpy.pi*fluid.x[i]/(mode*domain_width)):
            fluid.u[i] = 0.5
            fluid.color[i] = 1

    # extract the top and bottom boundary particles
    indices = numpy.where( fluid.y > domain_height )[0]
    wall = fluid.extract_particles( indices )
    fluid.remove_particles( indices )

    indices = numpy.where( fluid.y < 0 )[0]
    bottom = fluid.extract_particles( indices )
    fluid.remove_particles( indices )
    
    # concatenate the two boundaries
    print wall.num_real_particles, wall.get_number_of_particles()
    wall.append_parray( bottom )
    wall.set_name( 'wall' )

    print wall.num_real_particles, wall.get_number_of_particles()

    # set the number density for the wall particles initially.
    wall.V[:] = 1./volume

    # set additional output arrays for the fluid
    fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta'])
    
    print "2D KHI with %d fluid particles and %d wall particles"%(
            fluid.get_number_of_particles(), wall.get_number_of_particles())

    return [fluid, wall]
    
# domain for periodicity
domain = DomainManager(xmin=0, xmax=domain_width, ymin=0, ymax=domain_height,
                       periodic_in_x=True, periodic_in_y=False)

# Create the application.
app = Application(domain=domain)

# Create the kernel
kernel = Gaussian(dim=2)

# Create the Integrator.
integrator = PECIntegrator( fluid=VerletSymplecticWCSPHStep() )

# Create a solver.
solver = Solver(
    kernel=kernel, dim=dim, integrator=integrator,
    dt=dt, tf=tf, adaptive_timestep=False)

# create the equations
equations = [

    # We first compute the mass and number density of the fluid
    # phase. This is used in all force computations henceforth. The
    # number density (1/volume) is explicitly set for the solid phase
    # and this isn't modified for the simulation.
    Group(equations=[
            SummationDensity( dest='fluid', sources=['fluid', 'wall'] )
            ] ),
    
    # Given the updated number density for the fluid, we can update
    # the fluid pressure and compute the gradient of the color
    # function. At the end of this group, the interface normals and
    # dirac delta for the fluid phase is available. Additionally, we
    # can compute the Shepard Filtered velocity required for the
    # no-penetration boundary condition.
    Group(equations=[
            IsothermalEOS(dest='fluid', sources=None, rho0=rho0, c0=c0),
            ColorGradientUsingNumberDensity(dest='fluid', sources=['fluid', 'wall']),
            ShepardFilteredVelocity(dest='fluid', sources=['fluid']),
            ] ),

    # Once the pressure for the fluid phase has been updated, we can
    # extrapolate the pressure to the wall ghost particles. After this
    # group, the density and pressure of the boundary particles has
    # been updated and can be used in the integration
    # equations. Additionally, the interface curvature can be computed
    # using using the normals computed in the previous group.
    Group(
        equations=[
            SolidWallPressureBC(dest='wall', sources=['fluid'], p0=p0, rho0=rho0),
            InterfaceCurvatureFromNumberDensity(dest='fluid', sources=['fluid']),
            ], ),
    
    # The main acceleration block
    Group(
        equations=[

            # Gradient of pressure for the fluid phase using the
            # number density formulation. 
            PressureGradientUsingNumberDensity(dest='fluid', sources=['fluid']),

            # No penetration boundary condition using Adami et al's
            # generalized wall boundary condition. The extrapolated
            # pressure and density on the wall particles is used in
            # the gradient of pressure to simulate a repulsive force.
            MomentumEquationPressureGradient(dest='fluid', sources=['wall']),

            # Artificial viscosity for the fluid phase.
            ClearyArtificialViscosity(dest='fluid', sources=['fluid'], dim=dim, alpha=alpha),

            # No-slip boundary condition using Adami et al's
            # generalized wall boundary condition. This equation
            # basically computes the viscous contribution on the fluid
            # from the wall particles.
            SolidWallNoSlipBC(dest='fluid', sources=['wall'], nu=nu),
                                      
            # Surface tension force for the SY11 formulation
            ShadlooYildizSurfaceTensionForce(dest='fluid', sources=None, sigma=sigma),

            # XSPH velocity correction
            XSPHCorrection(dest='fluid', sources=['fluid'], eps=0.1),            
                                               
            ], )
    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
