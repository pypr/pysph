"""A very simple example to help benchmark PySPH.

The example creates a cube shaped block of water falling in free-space under
the influence of gravity while solving the incompressible, inviscid flow
equations.  Only 5 time steps are solved but with a million particles.  It is
easy to change the number of particles by simply changing the parameter `dx`.

For example to check the performance of PySPH using OpenMP one could try
the following::

    $ python cube.py --disable-output

    $ python cube.py --disable-output --openmp

"""

import numpy

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.nnps import LinkedListNNPS

from pysph.sph.basic_equations import ContinuityEquation, XSPHCorrection
from pysph.sph.wc.basic import TaitEOS, MomentumEquation

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import WCSPHStep

def make_cube(dx=0.1, hdx=1.5, rho0=1000.0):
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    zmin, zmax = 0.0, 1.0
    x, y, z = numpy.mgrid[xmin:xmax:dx, ymin:ymax:dx, zmin:zmax:dx]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    # set up particle properties
    h0 = hdx * dx

    volume = dx**3
    m0 = rho0 * volume

    fluid = get_particle_array_wcsph(name='fluid', x=x, y=y, z=z)
    fluid.m[:] = m0
    fluid.h[:] = h0

    fluid.rho[:] = rho0
    #nnps = LinkedListNNPS(dim=3, particles=[fluid])
    #nnps.spatially_order_particles(0)

    print("Number of particles:", x.size)
    fluid.set_lb_props( list(fluid.properties.keys()) )
    return [fluid]

# Timestep
dim = 3

dt = 1e-5
tf = 5e-5

# parameter to change the resolution.
dx = 0.01
nboundary_layers=3
hdx = 2.0
ro = 1000.0

h0 = dx * hdx
co = 10.0

gamma = 7.0
alpha = 0.5
beta = 0.0
B = co*co*ro/gamma

# Create the application.
app = Application()

# Create the kernel
kernel = CubicSpline(dim=dim)

# Create the integrator.
integrator = PECIntegrator(fluid=WCSPHStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=dim, integrator=integrator)

# Setup default parameters.
solver.set_time_step(dt)
solver.set_final_time(tf)

# create the equations
equations = [

        TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=gamma),

        ContinuityEquation(dest='fluid', sources=['fluid']),

        MomentumEquation(dest='fluid', sources=['fluid'],
                            alpha=alpha, beta=beta, gz=-9.81, c0=co),

        XSPHCorrection(dest='fluid', sources=['fluid']),

    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=make_cube, dx=dx)

app.run()
