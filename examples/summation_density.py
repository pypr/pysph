from numpy import linspace, ones_like
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import CubicSpline
from pysph.base.locators import AllPairLocator
from pysph.sph.basic_equations import SummationDensity, TaitEOS
from pysph.sph.wc.basic import TaitEOS
from pysph.sph.sph_eval import SPHEval
from pysph.sph.integrator import Integrator

def make_particles():
    x = linspace(0, 1.0, 11); dx = x[1] - x[0]
    h = ones_like(x) * 2*dx
    m = ones_like(x) * dx
    f = get_particle_array_wcsph(x=x, h=h, m=m, name = 'fluid')
    s = get_particle_array_wcsph(x=x, name = 'solid')
    return [f, s]

particles = make_particles()

kernel = CubicSpline(dim=1)

equations = [
             SummationDensity(dest='fluid', source='fluid'),
             SummationDensity(dest='fluid', source='solid'),
             TaitEOS(dest='fluid', source=None),
            ]

locator = AllPairLocator()
integrator = Integrator()
evaluator = SPHEval(particles, equations, locator, kernel, integrator)
evaluator.setup()
evaluator.compute()

print particles[0].rho
print particles[1].rho
