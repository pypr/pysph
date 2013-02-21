from numpy import linspace
from cgsph import CubicSpline, SummationDensity, AllPairLocator, SPHEval
from pysph.base.particle_array import get_particle_array

def make_particles():
    x = linspace(0, 5.0, 11)
    f = get_particle_array(x=x, name = 'fluid')
    s = get_particle_array(x=x, name = 'solid')
    return [f, s]

particles = make_particles()

kernel = CubicSpline()

equations = [SummationDensity(dest='fluid', sources=['fluid', 'solid']),
            ]            

locator = AllPairLocator()
evaluator = SPHEval(particles, equations, locator, kernel)
evaluator.setup_calc('test.pyx')
evaluator.compute()
    
print particles[0].rho
print particles[1].rho
