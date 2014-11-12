""" This simulates two square blocks of water colliding with each other.

This example solves exactly the same problem as the two_blocks.py but shows
how they can be treated as different fluids.

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

    u1 = numpy.ones_like(x1)
    u2 = -numpy.ones_like(x2)

    rho = numpy.ones_like(x1)*rho0
    h = numpy.ones_like(u1)*hdx*dx
    m = numpy.ones_like(u1)*dx*dx*rho0

    fluid1 = get_particle_array_iisph(
        name='fluid1', x=x1, y=y1, u=u1, rho=rho, m=m, h=h
    )
    fluid2 = get_particle_array_iisph(
        name='fluid2', x=x2, y=y2, u=u2, rho=rho, m=m, h=h
    )
    return [fluid1,fluid2]


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

integrator = EulerIntegrator(fluid1=IISPHStep(), fluid2=IISPHStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                dt=dt, tf=tf, adaptive_timestep=True,
                fixed_h=False)
solver.set_print_freq(10)

# create the equations
# create the equations
equations = [

    Group(
        equations=[
            SummationDensity(dest='fluid1', sources=['fluid1', 'fluid2']),
            SummationDensity(dest='fluid2', sources=['fluid1', 'fluid2']),
        ],
        real=False
    ),

    Group(
        equations=[
            AdvectionAcceleration(
                dest='fluid1', sources=None, gx=0.0, gy=0.81, gz=0.0
            ),
            AdvectionAcceleration(
                dest='fluid2', sources=None, gx=0.0, gy=0.81, gz=0.0
            ),
            #ViscosityAcceleration(dest='fluid1', sources=['fluid1', 'fluid2'], nu=8.9e-4),
            ComputeDII(dest='fluid1', sources=['fluid1', 'fluid2']),
            ComputeDII(dest='fluid2', sources=['fluid2', 'fluid1']),
        ]
    ),

    Group(
        equations=[
            ComputeRhoAdvection(dest='fluid1', sources=['fluid1', 'fluid2']),
            ComputeRhoAdvection(dest='fluid2', sources=['fluid2', 'fluid1']),
            ComputeAII(dest='fluid1', sources=['fluid1', 'fluid2']),
            ComputeAII(dest='fluid2', sources=['fluid2', 'fluid1']),
        ]
    ),

    #####################################################################
    # "Pressure solve" step as per algorithm 1 in paper.
    Group(
        equations=[
            Group(
                equations=[
                    ComputeDIJPJ(dest='fluid1', sources=['fluid1', 'fluid2']),
                    ComputeDIJPJ(dest='fluid2', sources=['fluid1', 'fluid2'])
                ]
            ),
            Group(
                equations=[
                    PressureSolve(
                        dest='fluid1', sources=['fluid1', 'fluid2'], rho0=rho0,
                        tolerance=1e-3, debug=False
                    ),
                    PressureSolve(
                        dest='fluid2', sources=['fluid1', 'fluid2'], rho0=rho0,
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
            PressureForce(dest='fluid1', sources=['fluid1', 'fluid2']),
            PressureForce(dest='fluid2', sources=['fluid1', 'fluid2']),
        ],
    ),
]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
