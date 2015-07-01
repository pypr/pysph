"""Evolution of a circular patch of incompressible fluid. See
J. J. Monaghan "Simulating Free Surface Flows with SPH", JCP, 1994,
100, pp 399 - 406

This version uses the Implicit Incompressible SPH technique described by

M. Ihmsen, J. Cornelis, B. Solenthaler, C. Horvath, M. Teschner, "Implicit
Incompressible SPH," IEEE Transactions on Visualization and Computer Graphics,
vol. 20, no. 3, pp. 426-435, March 2014.
http://dx.doi.org/10.1109/TVCG.2013.105

An initially circular patch of fluid is subjected to a velocity
profile that causes it to deform into an ellipse. Incompressibility
causes the initially circular patch to deform into an ellipse such
that the area is conserved. An analytical solution for the locus of
the patch is available (exact_solution)

This is a standard test for the formulations for the incompressible
SPH equations.

"""
# NumPy and standard library imports
from numpy import ones_like, mgrid, sqrt

# PySPH base and carray imports
from pysph.base.utils import get_particle_array_iisph
from pysph.base.kernels import CubicSpline

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# PySPH sph imports
from pysph.sph.equation import Group
from pysph.sph.iisph import (AdvectionAcceleration, ComputeAII, ComputeDII,
    ComputeDIJPJ, ComputeRhoAdvection, IISPHStep, PressureSolve, PressureForce,
    SummationDensity)
from pysph.sph.integrator import EulerIntegrator


def exact_solution(tf=0.0075, dt=1e-4):
    """Exact solution for the locus of the circular patch."""
    import numpy

    A0 = 100
    a0 = 1.0

    t = 0.0

    theta = numpy.linspace(0,2*numpy.pi, 101)

    Anew = A0
    anew = a0

    while t <= tf:
        t += dt

        Aold = Anew
        aold = anew

        Anew = Aold +  dt*(Aold*Aold*(aold**4 - 1))/(aold**4 + 1)
        anew = aold +  dt*(-aold * Aold)

    dadt = Anew**2 * (anew**4 - 1)/(anew**4 + 1)
    po = 0.5*-anew**2 * (dadt - Anew**2)

    return anew*numpy.cos(theta), 1/anew*numpy.sin(theta), po

ro = 1.0
hdx = 1.3
def get_circular_patch(dx=0.025, **kwargs):
    """Create the circular patch of fluid."""
    name = 'fluid'
    x,y = mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
    x = x.ravel()
    y = y.ravel()

    m = ones_like(x)*dx*dx
    h = ones_like(x)*hdx*dx
    rho = ones_like(x) * ro

    p = ones_like(x)

    u = -100*x
    v = 100*y

    # remove particles outside the circle
    indices = []
    for i in range(len(x)):
        if sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
            indices.append(i)

    pa = get_particle_array_iisph(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v,
                                  name=name)
    pa.remove_particles(indices)

    print("Elliptical drop :: %d particles"%(pa.get_number_of_particles()))

    return [pa,]

# Create the application.
app = Application()

# Set the SPH kernel. The spline based kernels are much more efficient
#(but less accurate) than the Gaussian
kernel = CubicSpline(dim=2)

# Create the Integrator.
integrator = EulerIntegrator(fluid=IISPHStep())

# Construct the solver. Use the output_at_times list to specify instants of
# time at which the output solution is  required.
dt = 2e-4;
tf = 0.0075
solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                dt=dt, tf=tf, adaptive_timestep=False,
                cfl=0.05,
                output_at_times=[0.0033, 0.0052])

# select True if you want to dump out remote particle properties in
# parallel runs. This can be over-ridden with the --output-remote
# command line option
solver.set_output_only_real(True)
solver.set_print_freq(5)

# Define the SPH equations used to solve this problem
equations = [

    #####################################################################
    # "Predict advection" step as per algorithm 1 in paper.
    Group(
        equations=[
            SummationDensity(dest='fluid', sources=['fluid']),
        ],
        real=False
    ),

    Group(
        equations=[
            AdvectionAcceleration(dest='fluid', sources=None),
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
                equations=[ComputeDIJPJ(dest='fluid', sources=['fluid'])]
            ),
            Group(
                equations=[
                    PressureSolve(
                        dest='fluid', sources=['fluid'], rho0=ro,
                        tolerance=1e-2, debug=False
                    ),
                  ]
            ),
        ],
        iterate=True,
        max_iterations=20
    ),

    Group(
        equations=[
            PressureForce(dest='fluid', sources=['fluid']),
        ],
    ),

]

# Setup the application and solver.
app.setup(solver=solver, equations=equations,
          particle_factory=get_circular_patch)

# run the solver...
app.run()
