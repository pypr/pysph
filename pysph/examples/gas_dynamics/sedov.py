"""Sedov point explosion problem. (7 minutes)

Particles are distributed on concentric circles about the origin with
increasing number of particles with increasing radius. A unit charge
is distributed about the center which gives the initial pressure
disturbance.

"""
# NumPy and standard library imports
import os.path
import numpy

# PySPH base and carray imports
from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application
from pysph.sph.scheme import GasDScheme

# Numerical constants
dim = 2
gamma = 5.0/3.0
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-4
tf = 0.1

# scheme constants
alpha1 = 10.0
alpha2 = 1.0
beta = 2.0
kernel_factor = 1.2

class SedovPointExplosion(Application):
    def create_particles(self):
        fpath = os.path.join(
            os.path.dirname(__file__), 'ndspmhd-sedov-initial-conditions.npz'
        )
        data = numpy.load(fpath)
        x = data['x']
        y = data['y']
        rho = data['rho']
        p = data['p']
        e = data['e']
        h = data['h']
        m = data['m']

        fluid = gpa(name='fluid', x=x, y=y, rho=rho, p=p, e=e, h=h, m=m)
        self.scheme.setup_properties([fluid])

        # set the initial smoothing length proportional to the particle
        # volume
        fluid.h[:] = kernel_factor * (fluid.m/fluid.rho)**(1./dim)

        print("Sedov's point explosion with %d particles"%(fluid.get_number_of_particles()))

        return [fluid,]

    def create_scheme(self):
        s = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
            kernel_factor=kernel_factor, alpha1=alpha1, alpha2=alpha2,
            beta=beta, adaptive_h_scheme="gsph",
            update_alpha1=True, update_alpha2=True
        )
        s.configure_solver(dt=dt, tf=tf, adaptive_timestep=False, pfreq=25)
        return s


if __name__ == '__main__':
    app = SedovPointExplosion()
    app.run()
