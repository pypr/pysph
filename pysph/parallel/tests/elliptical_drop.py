"""Evolution of a circular patch of incompressible fluid. See
J. J. Monaghan "Simulating Free Surface Flows with SPH", JCP, 1994,
100, pp 399 - 406

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
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import CubicSpline

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

# PySPH sph imports
from pysph.sph.equation import Group
from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
from pysph.sph.wc.basic import TaitEOS, MomentumEquation, UpdateSmoothingLengthFerrari, \
    ContinuityEquationDeltaSPH, MomentumEquationDeltaSPH

co = 1400.0; ro = 1.0
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

    p = ones_like(x) * 1./7.0 * co**2
    cs = ones_like(x) * co

    u = -100*x
    v = 100*y

    # remove particles outside the circle
    indices = []
    for i in range(len(x)):
        if sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
            indices.append(i)

    pa = get_particle_array_wcsph(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v,
                                  cs=cs, name=name)
    pa.remove_particles(indices)

    print("Elliptical drop :: %d particles"%(pa.get_number_of_particles()))

    # add requisite variables needed for this formulation
    for name in ('arho', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'rho0', 'u0',
                 'v0', 'w0', 'x0', 'y0', 'z0'):
        pa.add_property(name)

    # set the output property arrays
    pa.set_output_arrays( ['x', 'y', 'u', 'v', 'rho', 'h', 'p', 'pid', 'tag', 'gid'] )

    return [pa,]

# Create the application.
app = Application()

kernel = CubicSpline(dim=2)

integrator = EPECIntegrator(fluid=WCSPHStep())

dt = 5e-6; tf = 0.005
solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                dt=dt, tf=tf, adaptive_timestep=True,
                cfl=0.1, n_damp=40)

# select True if you want to dump out remote particle properties in
# parallel runs. This can be over-ridden with the --output-remote
# command line option
solver.set_output_only_real(True)
solver.set_print_freq(1000)

# Define the SPH equations used to solve this problem
equations = [

    # Equation of state: p = f(rho)
    Group(equations=[
            TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=7.0),
            ], real=False),

    # Block for the accelerations. Choose between either the Delta-SPH
    # formulation or the standard Monaghan 1994 formulation
    Group( equations=[

            # Density rate: drho/dt with dissipative penalization
            #ContinuityEquationDeltaSPH(dest='fluid',  sources=['fluid',], delta=0.1, c0=co),
            ContinuityEquation(dest='fluid',  sources=['fluid',]),

            # Acceleration: du,v/dt
            #MomentumEquationDeltaSPH(dest='fluid', sources=['fluid'], alpha=0.2, rho0=ro, c0=co),
            MomentumEquation(dest='fluid', sources=['fluid'], alpha=0.2, beta=0.0),

            # XSPH velocity correction
            XSPHCorrection(dest='fluid', sources=['fluid']),

            ],),

    # Update smoothing lengths at the end.
    Group( equations=[

            UpdateSmoothingLengthFerrari(dest='fluid', sources=None, dim=2, hdx=hdx),
            ], real=False),


    ]

# Setup the application and solver.
app.setup(solver=solver, equations=equations,
          particle_factory=get_circular_patch)

# run the solver...
app.run()
