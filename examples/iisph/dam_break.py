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

import sys
import os
# Need this to import db_geometry.
sys.path.append(os.pardir)
import numpy
from db_geometry import DamBreak2DGeometry

from pysph.base.kernels import CubicSpline
from pysph.sph.basic_equations import XSPHCorrection
from pysph.sph.equation import Group

from pysph.sph.iisph import (AdvectionAcceleration, ComputeAII, ComputeAIIBoundary,
    ComputeDII,
    ComputeDIIBoundary, ComputeDIJPJ, ComputeRhoAdvection, ComputeRhoBoundary,
    IISPHStep, NumberDensity, PressureSolve, PressureSolveBoundary,
    PressureForce, PressureForceBoundary, SummationDensity, SummationDensityBoundary)

from pysph.sph.integrator import EulerIntegrator


from pysph.solver.application import Application
from pysph.solver.solver import Solver

dim = 2
fluid_column_height = 2.0
fluid_column_width  = 1.0
container_height = 3.0
container_width  = 4.0
nboundary_layers=2

dt = 1e-3
tf = 2.5

hdx = 1.0
dx = dy = 0.025
ro = 1000.0
beta = 0.0

geom = DamBreak2DGeometry(
    container_width=container_width, container_height=container_height,
    fluid_column_height=fluid_column_height,
    fluid_column_width=fluid_column_width, dx=dx, dy=dy,
    nboundary_layers=1, ro=ro, co=1.0,
    with_obstacle=False,
    beta=1.0, nfluid_offset=1, hdx=hdx, iisph=True)


def create_particles(**kw):
    fluid, boundary = geom.create_particles(**kw)
    boundary.x -= 0.05
    boundary.y -= 0.05
    return [fluid, boundary]

# Create the application.
app = Application()

# Create the kernel
kernel = CubicSpline(dim=2)


# Create the Integrator. Currently, PySPH supports multi-stage,
# predictor corrector and a TVD-RK3 integrators.

integrator = EulerIntegrator(fluid=IISPHStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                dt=dt, tf=tf,
                fixed_h=False)

# create the equations
equations = [

    #####################################################################
    # "Predict advection" step as per algorithm 1 in paper.
    Group(equations=[
            NumberDensity(dest='boundary', sources=['boundary']),
        ]
    ),
    Group(
        equations=[
            SummationDensity(dest='fluid', sources=['fluid']),
        ],
        real=False
    ),
    Group(
        equations=[
            SummationDensityBoundary(dest='fluid', sources=['boundary'], rho0=ro),
        ],
        real=False
    ),

    Group(
        equations=[
            AdvectionAcceleration(
                dest='fluid', sources=None, gx=0.0, gy=-9.81, gz=0.0
            ),
            ComputeDII(dest='fluid', sources=['fluid']),
            ComputeDIIBoundary(dest='fluid', sources=['boundary'], rho0=ro),
        ]
    ),

    Group(
        equations=[
            ComputeRhoAdvection(dest='fluid', sources=['fluid']),
            ComputeRhoBoundary(dest='fluid', sources=['boundary'], rho0=ro),
            ComputeAII(dest='fluid', sources=['fluid']),
            ComputeAIIBoundary(dest='fluid', sources=['boundary'], rho0=ro),
        ]
    ),

    #####################################################################
    # "Pressure solve" step as per algorithm 1 in paper.
    Group(
        equations=[
            Group(
                equations=[ComputeDIJPJ(dest='fluid', sources=['fluid'])]
            ),
            Group(
                equations=[
                    PressureSolve(
                        dest='fluid', sources=['fluid'], rho0=ro,
                        tolerance=1e-3, debug=False
                    ),
                    PressureSolveBoundary(
                        dest='fluid', sources=['boundary'], rho0=ro,
                    ),
                  ]
            ),
        ],
        iterate=True,
        max_iterations=20,
        min_iterations=2
    ),

    Group(
        equations=[
            PressureForce(dest='fluid', sources=['fluid']),
            PressureForceBoundary(dest='fluid', sources=['boundary'], rho0=ro),
            XSPHCorrection(dest='fluid', sources=['fluid']),
        ],
    ),
]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles, nboundary_layers=1, nfluid_offset=1)

app.run()
