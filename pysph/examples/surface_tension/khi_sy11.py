from math import sqrt
import numpy as np
import os
import numpy
from pysph.sph.surface_tension import get_surface_tension_equations

from pysph.tools.geometry import get_2d_block, remove_overlap_particles
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, QuinticSpline

from pysph.solver.application import Application
from pysph.solver.solver import Solver

from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import PECIntegrator

from pysph.base.nnps import DomainManager
from pysph.solver.utils import iter_output

# problem parameters
dim = 2
domain_width = 1.0
domain_height = 1.0

# numerical constants
alpha = 0.0001
wavelength = 1.0
wavenumber = 2*numpy.pi/wavelength
Ri = 0.1
rho0 = rho1 = 1000.0
rho2 = rho1 = 2000.0
U = 0.5
sigma = Ri * (rho1*rho2) * (2*U)**2/(wavenumber*(rho1 + rho2))
psi0 = 0.03*domain_height
gy = -9.81

# discretization parameters
nghost_layers = 5
dx = dy = 0.0125
dxb2 = dyb2 = 0.5 * dx
volume = dx*dx
hdx = 1.0
h0 = hdx * dx
epsilon = 0.01/h0
rho0 = 1000.0
c0 = 10.0
p0 = c0*c0*rho0
nu = 0.125 * alpha * h0 * c0

# time steps and final time
tf = 3.0

dt1 = 0.25*np.sqrt(rho0*h0*h0*h0/(2.0*np.pi*sigma))

dt2 = 0.25*h0/(c0)

dt3 = 0.125*rho0*h0*h0/nu

dt = 0.9*min(dt1, dt2, dt3)

factor1 = 0.8
factor2 = 1/factor1


class SquareDroplet(Application):

    def create_particles(self):
        ghost_extent = (nghost_layers + 0.5)*dx

        x, y = numpy.mgrid[dxb2:domain_width:dx,
                           -ghost_extent:domain_height+ghost_extent:dy]
        x = x.ravel()
        y = y.ravel()

        m = numpy.ones_like(x) * volume * rho0
        rho = numpy.ones_like(x) * rho0
        p = numpy.ones_like(x) * p0
        h = numpy.ones_like(x) * h0
        cs = numpy.ones_like(x) * c0

        # additional properties required for the fluid.
        additional_props = [
            # volume inverse or number density
            'V', 'pi00', 'pi01', 'pi02', 'pi10', 'pi11', 'pi12', 'pi20',
            'pi21', 'pi22',

            # color and gradients
            'color', 'scolor', 'cx', 'cy', 'cz', 'cx2', 'cy2', 'cz2',

            # discretized interface normals and dirac delta
            'nx', 'ny', 'nz', 'ddelta',

            # interface curvature
            'kappa', 'nu', 'alpha',

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
            'N', 'wij_sum', 'wg', 'ug', 'vg'

        ]

        # get the fluid particle array
        fluid = get_particle_array(
            name='fluid', x=x, y=y, h=h, m=m, rho=rho, cs=cs, p=p,
            additional_props=additional_props)

        # set the fluid velocity with respect to the sinusoidal
        # perturbation
        fluid.u[:] = -U
        fluid.N[:] = 0.0
        fluid.nu[:] = nu
        fluid.alpha[:] = sigma
        mode = 1
        for i in range(len(fluid.x)):
            ang = 2*numpy.pi*fluid.x[i]/(mode*domain_width)
            if fluid.y[i] >= domain_height/2+psi0*domain_height*numpy.sin(ang):
                fluid.u[i] = U
                fluid.color[i] = 1.0
                fluid.rho[i] = 2000.0
        # extract the top and bottom boundary particles
        indices = numpy.where(fluid.y > domain_height)[0]
        wall = fluid.extract_particles(indices)
        fluid.remove_particles(indices)

        indices = numpy.where(fluid.y < 0)[0]
        bottom = fluid.extract_particles(indices)
        fluid.remove_particles(indices)

        # concatenate the two boundaries
        wall.append_parray(bottom)
        wall.set_name('wall')

        # set the number density initially for all particles
        fluid.V[:] = 1./volume
        wall.V[:] = 1./volume
        for i in range(len(wall.x)):
            if wall.y[i] > 0.5:
                wall.color[i] = 1.0
            else:
                wall.color[i] = 0.0
        # set additional output arrays for the fluid
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny',
                                 'ddelta', 'p', 'rho', 'au', 'av'])
        wall.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny',
                                'ddelta', 'p', 'rho', 'au', 'av'])

        print("2D KHI with %d fluid particles and %d wall particles" % (
            fluid.get_number_of_particles(),
            wall.get_number_of_particles()))

        return [fluid, wall]

    def create_domain(self):
        return DomainManager(xmin=0, xmax=domain_width, ymin=0,
                             ymax=domain_height,
                             periodic_in_x=True, periodic_in_y=False,
                             n_layers=5.0)

    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = PECIntegrator(fluid=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False)
        return solver

    def add_user_options(self, group):
        choices = ['morris', 'tvf', 'adami_stress', 'adami', 'shadloo']
        group.add_argument(
            "--scheme", action="store", dest='scheme', default='morris',
            choices=choices,
            help='Specify scheme to use among %s' % choices
        )

    def create_equations(self):
        return get_surface_tension_equations(['fluid'], ['wall'],
                                             self.options.scheme, rho0, p0, c0,
                                             0,  factor1, factor2, nu, sigma,
                                             2, epsilon, 1, real=False)


if __name__ == '__main__':
    app = SquareDroplet()
    app.run()
