import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.particle_array import ParticleArray

from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep

from pysph.sph.integrator import EulerIntegrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.base.reduce_array import parallel_reduce_array, serial_reduce_array


class TotalMass(Equation):
    def reduce(self, dst, t, dt):
        m = serial_reduce_array(dst.m, op='sum')
        dst.total_mass[0] = parallel_reduce_array(m, op='sum')


class DummyStepper(IntegratorStep):
    def initialize(self):
        pass

    def stage1(self):
        pass


class MyApp(Application):
    def create_particles(self):
        x = np.linspace(0, 10, 10)
        m = np.ones_like(x)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        h = np.ones_like(x) * 0.2

        fluid = ParticleArray(name='fluid', x=x, y=y, z=z, m=m, h=h)
        fluid.add_constant('total_mass', 0.0)
        return [fluid]

    def create_solver(self):
        dim = 1
        # Create the kernel
        kernel = CubicSpline(dim=dim)

        # Create the integrator.
        integrator = EulerIntegrator(fluid=DummyStepper())

        solver = Solver(kernel=kernel, dim=dim, integrator=integrator)
        solver.set_time_step(0.1)
        solver.set_final_time(0.1)
        # There is no need to write any output as the test below
        # computes the total mass.
        solver.set_disable_output(True)
        return solver

    def create_equations(self):
        return [TotalMass(dest='fluid', sources=['fluid'])]


def main():
    # Create the application.
    app = MyApp()
    app.run()

    fluid = app.particles[0]
    err = fluid.total_mass[0] - 10.0
    assert abs(err) < 1e-16, "Error: %s" % err


if __name__ == '__main__':
    main()
