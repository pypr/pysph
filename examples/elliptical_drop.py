"""Elliptical drop example"""

from numpy import ones_like, mgrid, sqrt, arange, array, savez
from pysph.base.utils import get_particle_array
from pysph.base.carray import LongArray

from pysph.base.kernels import CubicSpline
from pysph.base.locators import AllPairLocator
from pysph.base.nnps import NNPS

from time import time

from pysph.sph.equations import (TaitEOS, ContinuityEquation, MomentumEquation,
    XSPHCorrection)

from pysph.sph.integrator import WCSPHRK2Integrator
from pysph.sph.sph_eval import SPHEval

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
hdx = 2.0
def get_circular_patch(name="", type=0, dx=0.025,
                       cl_precision="single", **kwargs):
    
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
            
    pa = get_particle_array(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v,
                                 cs=cs,name=name, type=type,
                                 cl_precision=cl_precision)

    la = LongArray(len(indices))
    la.set_data(array(indices))

    pa.remove_particles(la)

    print 'Number of particles: ', len(pa.x)

    # add the acceleration variables
    pa.add_property( {'name': 'arho'} )
    pa.add_property( {'name': 'au'} )
    pa.add_property( {'name': 'av'} )
    pa.add_property( {'name': 'aw'} )

    pa.add_property( {'name': 'ax'} )
    pa.add_property( {'name': 'ay'} )
    pa.add_property( {'name': 'az'} )

    pa.add_property( {'name': 'rho0'} )
    pa.add_property( {'name': 'u0'} )
    pa.add_property( {'name': 'v0'} )
    pa.add_property( {'name': 'w0'} )

    pa.add_property( {'name': 'x0'} )
    pa.add_property( {'name': 'y0'} )
    pa.add_property( {'name': 'z0'} )

    return [pa,]

particles = get_circular_patch(name="fluid")
kernel = CubicSpline(dim=2)

equations = [TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=7.0),
             ContinuityEquation(dest='fluid',  sources=['fluid',]),
             MomentumEquation(dest='fluid', sources=['fluid'], alpha=1.0, beta=1.0),
             XSPHCorrection(dest='fluid', sources=['fluid']),
            ]

# Create the Locator and SPHEval object
locator = AllPairLocator()
evaluator = SPHEval(particles, equations, locator, kernel)

# Create the NNPS object
nnps = NNPS(dim=2, particles=particles, radius_scale=2.0)

# Set NNPS for SPHEval and the calc
evaluator.set_nnps(nnps)

with open('test.pyx', 'w') as f:
    print >> f, evaluator.ext_mod.code

# set the integrator
integrator = WCSPHRK2Integrator(evaluator=evaluator, particles=particles)

pa = particles[0]
ex, ey, ep = exact_solution()
t1 = time()
for i in range(750):
    integrator.integrate(1e-5)
    if i % 50 == 0:
        savez('solution%03d.npz'%i, x=pa.x, y=pa.y, ex=ex, ey=ey)

elapsed = time() - t1
savez('solution%03d.npz'%i, x=pa.x, y=pa.y, ex=ex, ey=ey)

print "750 iterations in %gs, avg = %g s/iteration"%(elapsed, elapsed/750)

# f = evaluator.calc.fluid
# print type(f), f, f.x
