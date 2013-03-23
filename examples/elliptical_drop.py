"""Elliptical drop example"""

from numpy import ones_like, mgrid, sqrt, arange, array
from pysph.base.particle_array import get_particle_array
from pysph.base.carray import LongArray

from pysph.base.kernels import CubicSpline
from pysph.base.locators import AllPairLocator
from pysph.sph.equations import (TaitEOS, ContinuityEquation, MomentumEquation,
    XSPHCorrection)
from pysph.sph.integrator import WCSPHRK2Integrator
from pysph.sph.sph_eval import SPHEval


hdx = 1.3
def get_circular_patch(name="", type=0, dx=0.025/hdx,
                       cl_precision="single", **kwargs):
    
    x,y = mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
    x = x.ravel()
    y = y.ravel()
 
    m = ones_like(x)*dx*dx
    h = ones_like(x)*hdx*dx
    rho = ones_like(x)

    p = 0.5*1.0*100*100*(1 - (x**2 + y**2))

    cs = ones_like(x) * 100.0

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

    pa.set(idx=arange(len(pa.x)))
 
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

equations = [TaitEOS(dest='fluid', sources=None),
             ContinuityEquation(dest='fluid',  sources=['fluid',]),
             MomentumEquation(dest='fluid', sources=['fluid']),
             XSPHCorrection(dest='fluid', sources=['fluid']),
            ]

locator = AllPairLocator()
evaluator = SPHEval(particles, equations, locator, kernel)

with open('test.pyx', 'w') as f:
    print >> f, evaluator.ext_mod.code

# set the integrator
integrator = WCSPHRK2Integrator(evaluator=evaluator, particles=particles)

#evaluator.compute()

# f = evaluator.calc.fluid
# print type(f), f, f.x
