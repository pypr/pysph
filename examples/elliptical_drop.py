"""Elliptical drop example"""

from numpy import ones_like, mgrid, sqrt, array, savez
from pysph.base.utils import get_particle_array_wcsph

from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import WCSPHStep, Integrator

from pyzoltan.core.carray import LongArray

from time import time
from pysph.sph.basic_equations import ContinuityEquation, XSPHCorrection
from pysph.sph.wc.basic import TaitEOS, MomentumEquation

def exact_solution(tf=0.0075, dt=1e-4):
    """ Exact solution for the the elliptical drop equations """
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

co = 1400.0; ro = 1.0
hdx = 1.3
def get_circular_patch(name="fluid", dx=0.025, empty=False, **kwargs):

    if empty:
        pa = get_particle_array_wcsph(name=name)
    else:
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

        indices = []
        for i in range(len(x)):
            if sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
                indices.append(i)

        pa = get_particle_array_wcsph(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v,
                                      cs=cs, name=name)

        la = LongArray(len(indices))
        la.set_data(array(indices))

        pa.remove_particles(la)

        print "Elliptical drop :: %d particles"%(pa.get_number_of_particles())

    # add requisite variables
    for name in ('arho', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'rho0', 'u0',
                 'v0', 'w0', 'x0', 'y0', 'z0'):
        pa.add_property( {'name': name} )

    return [pa,]

# Create the application.
app = Application()

# Create the kernel
kernel = CubicSpline(dim=2)

integrator = Integrator(fluid=WCSPHStep())

# Create a solver.
solver = Solver(kernel=kernel, dim=2, integrator=integrator)
# Setup default parameters.
solver.set_time_step(1e-5)
solver.set_final_time(0.0075)

equations = [TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=7.0),
             ContinuityEquation(dest='fluid',  sources=['fluid',]),
             MomentumEquation(dest='fluid', sources=['fluid'], alpha=1.0, beta=1.0),
             XSPHCorrection(dest='fluid', sources=['fluid']),
            ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=get_circular_patch,
          name='fluid')

t1 = time()
app.run()
elapsed = time() - t1
print "750 iterations in %gs, avg = %g s/iteration"%(elapsed, elapsed/750)
