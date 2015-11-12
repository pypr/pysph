"""Curvature computation for a circular droplet. (5 seconds)

For particles distributed in a box, an initial circular interface is
distinguished by coloring the particles within a circle. The resulting
equations can then be used to check for the numerical curvature and
discretized dirac-delta function based on this artificial interface.

"""

import numpy

# Particle generator
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline

# SPH Equations and Group
from pysph.sph.equation import Group

from pysph.sph.wc.viscosity import ClearyArtificialViscosity

from pysph.sph.wc.transport_velocity import SummationDensity, MomentumEquationPressureGradient,\
    SolidWallPressureBC, SolidWallNoSlipBC, \
    StateEquation, MomentumEquationArtificialStress, MomentumEquationViscosity

from pysph.sph.surface_tension import ColorGradientUsingNumberDensity, \
    InterfaceCurvatureFromNumberDensity, ShadlooYildizSurfaceTensionForce,\
    SmoothedColor, AdamiColorGradient, AdamiReproducingDivergence,\
    MorrisColorGradient

from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

# PySPH solver and application
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# Integrators and Steppers
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import PECIntegrator

# Domain manager for periodic domains
from pysph.base.nnps import DomainManager

# problem parameters
dim = 2
domain_width = 1.0
domain_height = 1.0

# numerical constants
wavelength = 1.0
wavenumber = 2*numpy.pi/wavelength
rho0 = rho1 = 1000.0
rho2 = 1*rho1
U = 0.5
sigma = 1.0

# discretization parameters
dx = dy = 0.0125
dxb2 = dyb2 = 0.5 * dx
hdx = 1.5
h0 = hdx * dx
rho0 = 1000.0
c0 = 20.0
p0 = c0*c0*rho0
nu = 0.01

# set factor1 to [0.5 ~ 1.0] to simulate a thick or thin
# interface. Larger values result in a thick interface. Set factor1 =
# 1 for the Morris Method I
factor1 = 1.0
factor2 = 1./factor1

# correction factor for Morris's Method I. Set with_morris_correction
# to True when using this correction.
epsilon = 0.01/h0

# time steps
dt_cfl = 0.25 * h0/( 1.1*c0 )
dt_viscous = 0.125 * h0**2/nu
dt_force = 1.0

dt = 0.9 * min(dt_cfl, dt_viscous, dt_force)
tf = 5*dt
staggered = True


class CircularDroplet(Application):
    def create_domain(self):
        return DomainManager(
            xmin=0, xmax=domain_width, ymin=0, ymax=domain_height,
            periodic_in_x=True, periodic_in_y=True)

    def create_particles(self):
        if staggered:
            x1, y1 = numpy.mgrid[ dxb2:domain_width:dx, dyb2:domain_height:dy ]
            x2, y2 = numpy.mgrid[ dx:domain_width:dx, dy:domain_height:dy ]

            x1 = x1.ravel(); y1 = y1.ravel()
            x2 = x2.ravel(); y2 = y2.ravel()

            x = numpy.concatenate( [x1, x2] )
            y = numpy.concatenate( [y1, y2] )
            volume = dx*dx/2
        else:
            x, y = numpy.mgrid[ dxb2:domain_width:dx, dyb2:domain_height:dy ]
            x = x.ravel(); y = y.ravel()
            volume = dx*dx

        m = numpy.ones_like(x) * volume * rho0
        rho = numpy.ones_like(x) * rho0
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

            # velocity of magnitude squared needed for TVF
            'vmag2',

            # variable to indicate reliable normals and normalizing
            # constant
            'N', 'wij_sum'

            ]

        # get the fluid particle array
        fluid = get_particle_array(
            name='fluid', x=x, y=y, h=h, m=m, rho=rho, cs=cs,
            additional_props=additional_props)

        # set the color of the inner circle
        for i in range(x.size):
            if ( ((fluid.x[i]-0.5)**2 + (fluid.y[i]-0.5)**2) <= 0.25**2 ):
                fluid.color[i] = 1.0

        # particle volume
        fluid.V[:] = 1./volume

        # set additional output arrays for the fluid
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta', 'p',
                                 'kappa', 'N', 'scolor'])

        print("2D Circular droplet deformation with %d fluid particles"%(
                fluid.get_number_of_particles()))

        return [fluid,]


    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = PECIntegrator( fluid=TransportVelocityStep() )
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False, pfreq=1)
        return solver

    def create_equations(self):
        equations = [

            # We first compute the mass and number density of the fluid
            # phase. This is used in all force computations henceforth. The
            # number density (1/volume) is explicitly set for the solid phase
            # and this isn't modified for the simulation.
            Group(equations=[
                    SummationDensity( dest='fluid', sources=['fluid'] ),
                    ] ),

            # Given the updated number density for the fluid, we can update
            # the fluid pressure. Also compute the smoothed color based on the
            # color index for a particle.
            Group(equations=[
                    StateEquation(dest='fluid', sources=None, rho0=rho0,
                                  p0=p0, b=1.0),
                    SmoothedColor( dest='fluid', sources=['fluid'], smooth=True ),
                    ] ),

            #################################################################
            # Begin Surface tension formulation
            #################################################################
            # Scale the smoothing lengths to determine the interface
            # quantities.
            Group(equations=[
                    ScaleSmoothingLength(dest='fluid', sources=None, factor=factor1)
                    ], update_nnps=False ),

            # Compute the gradient of the color function with respect to the
            # new smoothing length. At the end of this Group, we will have the
            # interface normals and the discretized dirac delta function for
            # the fluid-fluid interface.
            Group(equations=[
                    MorrisColorGradient(dest='fluid', sources=['fluid'], epsilon=0.01/h0),
                    #ColorGradientUsingNumberDensity(dest='fluid', sources=['fluid'],
                    #                                epsilon=epsilon),
                    #AdamiColorGradient(dest='fluid', sources=['fluid']),
                    ],
                  ),

            # Compute the interface curvature using the modified smoothing
            # length and interface normals computed in the previous Group.
            Group(equations=[
                    InterfaceCurvatureFromNumberDensity(dest='fluid', sources=['fluid'],
                                                        with_morris_correction=True),
                    #AdamiReproducingDivergence(dest='fluid', sources=['fluid'],
                    #                           dim=2),
                    ], ),

            # Now rescale the smoothing length to the original value for the
            # rest of the computations.
            Group(equations=[
                    ScaleSmoothingLength(dest='fluid', sources=None, factor=factor2)
                    ], update_nnps=False,
                  ),
            #################################################################
            # End Surface tension formulation
            #################################################################

            # The main acceleration block
            Group(
                equations=[

                    # Gradient of pressure for the fluid phase using the
                    # number density formulation.
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid'], pb=p0),

                    # Artificial viscosity for the fluid phase.
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid'], nu=nu),

                    # Surface tension force for the SY11 formulation
                    ShadlooYildizSurfaceTensionForce(dest='fluid', sources=None, sigma=sigma),

                    # Artificial stress for the fluid phase
                    MomentumEquationArtificialStress(dest='fluid', sources=['fluid']),

                    ], )
        ]
        return equations

if __name__ == '__main__':
    app = CircularDroplet()
    app.run()
