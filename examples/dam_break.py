""" 2D Dam Break Over a dry bed. The case is described in "State of
the art classical SPH for free surface flows", Benedict D Rogers,
Robert A, Dalrymple and Alex J.C Crespo, Journal of Hydraulic
Research, Vol 48, Extra Issue (2010), pp 6-27


Setup:
------



x                   x !
x                   x !
x                   x !
x                   x !
x  o   o   o        x !
x    o   o          x !3m
x  o   o   o        x !
x    o   o          x !
x  o   o   o        x !
x                   x !
xxxxxxxxxxxxxxxxxxxxx |        o -- Fluid Particles
                               x -- Solid Particles
     -dx-                      dx = dy
_________4m___________

Y
|
|
|
|
|
|      /Z
|     /
|    /
|   /
|  /
| /
|/_________________X

Fluid particles are placed on a staggered grid. The nodes of the grid
are located at R = l*dx i + m * dy j with a two point bias (0,0) and
(dx/2, dy/2) refered to the corner defined by R. l and m are integers
and i and j are the unit vectors alon `X` and `Y` respectively.

For the Monaghan Type Repulsive boundary condition, a single row of
boundary particles is used with a boundary spacing delp = dx = dy.

For the Dynamic Boundary Conditions, a staggered grid arrangement is
used for the boundary particles.

Numerical Parameters:
---------------------

dx = dy = 0.012m
h = 0.0156 => h/dx = 1.3

Height of Water column = 2m
Length of Water column = 1m

Number of particles = 27639 + 1669 = 29308


ro = 1000.0
co = 10*sqrt(2*9.81*2) ~ 65.0
gamma = 7.0

Artificial Viscosity:
alpha = 0.5

XSPH Correction:
eps = 0.5

 """
import numpy
from db_geometry import DamBreak2DGeometry

from pysph.base.kernels import CubicSpline, WendlandQuintic
from pysph.base.kernels import get_correction
from pysph.sph.equation import Group
from pysph.sph.basic_equations import ContinuityEquation, XSPHCorrection
from pysph.sph.wc.basic import TaitEOS, MomentumEquation
from pysph.sph.wc.viscosity import LaminarViscosity

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import Integrator, WCSPHStep

dim = 2
fluid_column_height = 2.0
fluid_column_width  = 1.0
container_height = 3.0
container_width  = 4.0
nboundary_layers=2

dt = 1e-4
tf = 1.0

#h = 0.0156
h = 0.0390
hdx = 1.3
#h = 0.01
dx = dy = h/hdx
ro = 1000.0
co = 10.0 * numpy.sqrt(2*9.81*fluid_column_height)
gamma = 7.0
alpha = 0.5
beta = 0.0
B = co*co*ro/gamma
p0 = 1000.0

geom = DamBreak2DGeometry(
    container_width=container_width, container_height=container_height,
    fluid_column_height=fluid_column_height,
    fluid_column_width=fluid_column_width, dx=dx, dy=dy,
    nboundary_layers=nboundary_layers, ro=ro, co=co,
    with_obstacle=False)


# Create the application.
app = Application()

# Create the kernel
kernel = WendlandQuintic(dim=2)
kfactor = get_correction(kernel=kernel,h0=h)

integrator = Integrator(fluid=WCSPHStep(),
                        boundary=WCSPHStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=dim, integrator=integrator)
# Setup default parameters.
solver.set_time_step(dt)
solver.set_final_time(tf)

# create the equations
equations = [

    # Equation of state
    Group(equations=[

            TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=gamma, p0=0.0),
            TaitEOS(dest='boundary', sources=None, rho0=ro, c0=co, gamma=gamma, p0=0.0),

            ]),

    Group(equations=[

            # Continuity equation
            ContinuityEquation(dest='fluid', sources=['fluid', 'boundary']),
            ContinuityEquation(dest='boundary', sources=['fluid']),

            # Momentum equation
            MomentumEquation(dest='fluid', sources=['fluid', 'boundary'],
                     alpha=1.0, beta=1.0, gy=-9.81, c0=co, kfactor=kfactor),

            # Position step with XSPH
            XSPHCorrection(dest='fluid', sources=['fluid'])
            ]),
    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=geom.create_particles, hdx=hdx)

app.run()
