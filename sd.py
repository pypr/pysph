from cgsph import *
from pysph.base.particle_array import get_particle_array

def make_particles():
    f = get_particle_array(name = 'fluid')
    s = get_particle_array(name = 'solid')
    return [f, s]

particles = make_particles()

kernel = CubicSpline()

equations = [SummationDensity(dest='fluid', sources=['fluid', 'solid']),
            ]            

locator = AllPairLocator()
evaluator = SPHEval(particles, equations, locator, kernel)

with open('test.pyx', 'w') as f:
    evaluator.generate(f)


'''
evaluator.compute()

pa = particles.get_particle_array('fluid')
print pa.x, pa.rho
'''