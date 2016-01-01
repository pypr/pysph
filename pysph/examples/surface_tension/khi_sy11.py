"""2D Kelvin Helmoltz Instability example. (35 minutes)

This uses the Shadloo and Yildiz WCSPH formulation.

"""
import numpy

# Particle generator
from pysph.base.utils import get_particle_array
from pysph.base.kernels import WendlandQuintic

# SPH Equations and Group
from pysph.sph.equation import Group

# Equations for the fluid mechanics
from pysph.sph.basic_equations import XSPHCorrection, IsothermalEOS, BodyForce
from pysph.sph.wc.viscosity import ClearyArtificialViscosity
from pysph.sph.wc.basic import PressureGradientUsingNumberDensity
from pysph.sph.wc.transport_velocity import SummationDensity, SolidWallPressureBC

# Surface tension equations
from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength
from pysph.sph.surface_tension import ColorGradientUsingNumberDensity, \
    InterfaceCurvatureFromNumberDensity, ShadlooYildizSurfaceTensionForce,\
    SmoothedColor

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
alpha = 0.001
wavelength = 1.0
wavenumber = 2*numpy.pi/wavelength
Ri = 0.1
rho0 = rho1 = 1000.0
rho2 = rho1
U = 0.5
sigma = Ri * (rho1*rho2) * (2*U)**2/(wavenumber*(rho1 + rho2))
psi0 = 0.03*domain_height
gy = -9.81

# discretization parameters
nghost_layers = 5
dx = dy = 0.0125
dxb2 = dyb2 = 0.5 * dx
volume = dx*dx
hdx = 1.3
h0 = hdx * dx
rho0 = 1000.0
c0 = 10.0
p0 = c0*c0*rho0
nu = 0.125 * alpha * h0 * c0

# time steps and final time
tf = 3.0
dt = 1e-4

class KHISY11(Application):
    def create_particles(self):
        ghost_extent = (nghost_layers + 0.5)*dx

        x, y = numpy.mgrid[ dxb2:domain_width:dx, -ghost_extent:domain_height+ghost_extent:dy ]
        x = x.ravel(); y = y.ravel()

        m = numpy.ones_like(x) * volume * rho0
        rho = numpy.ones_like(x) * rho0
        p = numpy.ones_like(x) * p0
        h = numpy.ones_like(x) * h0
        cs = numpy.ones_like(x) * c0

        # additional properties required for the fluid.
        additional_props = [
            # volume inverse or number density
            'V',

            # color and gradients
            'color', 'scolor', 'cx', 'cy', 'cz', 'cx2', 'cy2', 'cz2',

            # discretized interface normals and dirac delta
            'nx', 'ny', 'nz', 'ddelta',

            # interface curvature
            'kappa',

            # filtered velocities
            'uf', 'vf', 'wf',

            # transport velocities
            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',

            # imposed accelerations on the solid wall
            'ax', 'ay', 'az', 'wij',

            # velocity of magnitude squared
            'vmag2',

            # variable to indicate reliable normals and normalizing
            # constant
            'N', 'wij_sum',

            ]

        # get the fluid particle array
        fluid = get_particle_array(
            name='fluid', x=x, y=y, h=h, m=m, rho=rho, cs=cs, p=p,
            additional_props=additional_props)

        # set the fluid velocity with respect to the sinusoidal
        # perturbation
        fluid.u[:] = -U
        mode = 1
        for i in range(len(fluid.x)):
            if fluid.y[i] > domain_height/2 + psi0*domain_height*numpy.sin(2*numpy.pi*fluid.x[i]/(mode*domain_width)):
                fluid.u[i] = U
                fluid.color[i] = 1

        # extract the top and bottom boundary particles
        indices = numpy.where( fluid.y > domain_height )[0]
        wall = fluid.extract_particles( indices )
        fluid.remove_particles( indices )

        indices = numpy.where( fluid.y < 0 )[0]
        bottom = fluid.extract_particles( indices )
        fluid.remove_particles( indices )

        # concatenate the two boundaries
        wall.append_parray( bottom )
        wall.set_name( 'wall' )

        # set the number density initially for all particles
        fluid.V[:] = 1./volume
        wall.V[:] = 1./volume

        # set additional output arrays for the fluid
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta',
                                 'p', 'rho', 'au', 'av'])

        print("2D KHI with %d fluid particles and %d wall particles"%(
                fluid.get_number_of_particles(), wall.get_number_of_particles()))

        return [fluid, wall]

    def create_domain(self):
        return DomainManager(xmin=0, xmax=domain_width, ymin=0, ymax=domain_height,
                               periodic_in_x=True, periodic_in_y=False)


    def create_solver(self):
        kernel = WendlandQuintic(dim=2)
        integrator = PECIntegrator( fluid=VerletSymplecticWCSPHStep() )
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False)
        return solver


    def create_equations(self):
        equations = [

            # We first compute the mass and number density of the fluid
            # phase. This is used in all force computations henceforth. The
            # number density (1/volume) is explicitly set for the solid phase
            # and this isn't modified for the simulation.
            Group(equations=[
                    SummationDensity( dest='fluid', sources=['fluid', 'wall'] )
                    ] ),

            # Given the updated number density for the fluid, we can update
            # the fluid pressure. Additionally, we compute the gradient of the
            # color function with respect to the original smoothing
            # length. This will compute the interface normals. Also compute
            # the smoothed color based on the color index for a particle.
            Group(equations=[
                    IsothermalEOS(dest='fluid', sources=None, rho0=rho0, c0=c0, p0=p0),
                    SmoothedColor( dest='fluid', sources=['fluid'] ),
                    ] ),

            #################################################################
            # Begin Surface tension formulation
            #################################################################
            # Scale the smoothing lengths to determine the interface
            # quantities. The NNPS need not be updated since the smoothing
            # length is decreased.
            Group(equations=[
                    ScaleSmoothingLength(dest='fluid', sources=None, factor=0.8)
                    ], update_nnps=False ),

            # Compute the gradient of the color function with respect to the
            # new smoothing length. At the end of this Group, we will have the
            # interface normals and the discretized dirac delta function for
            # the fluid-fluid interface.
            Group(equations=[
                    ColorGradientUsingNumberDensity(dest='fluid', sources=['fluid', 'wall']),
                    ],
                  ),

            # Compute the interface curvature using the modified smoothing
            # length and interface normals computed in the previous Group.
            Group(equations=[
                    InterfaceCurvatureFromNumberDensity(dest='fluid', sources=['fluid']),
                    ], ),

            # Now rescale the smoothing length to the original value for the
            # rest of the computations.
            Group(equations=[
                    ScaleSmoothingLength(dest='fluid', sources=None, factor=1.25)
                    ], update_nnps=False,
                  ),
            #################################################################
            # End Surface tension formulation
            #################################################################

            # Once the pressure for the fluid phase has been updated via the
            # state-equation, we can extrapolate the pressure to the wall
            # ghost particles. After this group, the density and pressure of
            # the boundary particles has been updated and can be used in the
            # integration equations.
            Group(
                equations=[
                    SolidWallPressureBC(
                        dest='wall', sources=['fluid'], p0=p0, rho0=rho0,
                        gy=gy, b=1.0),
                    ], ),

            # The main acceleration block
            Group(
                equations=[

                    # Body force due to gravity
                    BodyForce(dest='fluid', sources=None, fy=gy),

                    # Gradient of pressure for the fluid phase using the
                    # number density formulation. The no-penetration boundary
                    # condition is taken care of by using the boundary
                    # pressure and density.
                    PressureGradientUsingNumberDensity(
                        dest='fluid', sources=['fluid', 'wall']),

                    # Artificial viscosity for the fluid phase.
                    ClearyArtificialViscosity(
                        dest='fluid', sources=['fluid', 'wall'],
                        dim=dim, alpha=alpha),

                    # Surface tension force for the SY11 formulation
                    ShadlooYildizSurfaceTensionForce(
                        dest='fluid', sources=None, sigma=sigma),

                    # XSPH Correction
                    XSPHCorrection(dest='fluid', sources=['fluid'], eps=0.1),

                    ], )
        ]
        return equations

if __name__ == '__main__':
    app = KHISY11()
    app.run()
