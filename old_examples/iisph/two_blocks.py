""" This simulates two square blocks of water colliding with each other.
"""

import sys
import os
# Need this to import db_geometry.
sys.path.append(os.pardir)
import numpy
from db_geometry import create_2D_filled_region

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_iisph
from pysph.sph.equation import Group

from pysph.sph.iisph import (AdvectionAcceleration, ComputeAII, ComputeDII,
    ComputeDIJPJ, ComputeRhoAdvection, IISPHStep, PressureSolve,
    PressureForce, SummationDensity, ViscosityAcceleration)

from pysph.sph.integrator import EulerIntegrator


from pysph.solver.application import Application
from pysph.solver.solver import Solver

dx = 0.025
hdx = 1.0
rho0 = 1000

def create_particles():
    x1, y1 = create_2D_filled_region(-1, 0, 0, 1, dx)
    x2, y2 = create_2D_filled_region(0.5, 0, 1.5, 1, dx)

    x = numpy.concatenate((x1, x2))
    y = numpy.concatenate((y1, y2))
    u1 = numpy.ones_like(x1)
    u2 = -numpy.ones_like(x2)
    u = numpy.concatenate((u1, u2))

    rho = numpy.ones_like(u)*rho0
    h = numpy.ones_like(u)*hdx*dx
    m = numpy.ones_like(u)*dx*dx*rho0

    pa = get_particle_array_iisph(
        name='fluid', x=x, y=y, u=u, rho=rho, m=m, h=h
    )
    return [pa,]


dim = 2
dt = 1e-2
tf = 1.0

# Create the application.
app = Application()

# Create the kernel
#kernel = Gaussian(dim=dim)
kernel = CubicSpline(dim=dim)


# Create the Integrator. Currently, PySPH supports multi-stage,
# predictor corrector and a TVD-RK3 integrators.

integrator = EulerIntegrator(fluid=IISPHStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                dt=dt, tf=tf, adaptive_timestep=True,
                fixed_h=False)
solver.set_print_freq(10)

# create the equations
equations = [

    Group(
        equations=[
            SummationDensity(dest='fluid', sources=['fluid']),
        ],
        real=False
    ),

    Group(
        equations=[
            AdvectionAcceleration(
                dest='fluid', sources=None, gx=0.0, gy=0.0, gz=0.0
            ),
            #ViscosityAcceleration(dest='fluid', sources=['fluid'], nu=8.9e-4),
            ComputeDII(dest='fluid', sources=['fluid']),
        ]
    ),

    Group(
        equations=[
            ComputeRhoAdvection(dest='fluid', sources=['fluid']),
            ComputeAII(dest='fluid', sources=['fluid']),
        ]
    ),

    #####################################################################
    # "Pressure solve" step as per algorithm 1 in paper.
    Group(
        equations=[
            Group(
                equations=[
                    ComputeDIJPJ(dest='fluid', sources=['fluid']),
                ]
            ),
            Group(
                equations=[
                    PressureSolve(
                        dest='fluid', sources=['fluid'], rho0=rho0,
                        tolerance=1e-3, debug=False
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
        ],
    ),
]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
