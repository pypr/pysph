from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application

from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import Integrator

from pysph.tools import uniform_distribution

# SPH equations for this problem
from pysph.sph.equation import Group, Equation
from geometry import *


class SPHERICBenchmarkAcceleration(Equation):
    r"""Equation to set the acceleration for a washing machine problem
    benchmark problem.

    Notes:
    This equation must be instantiated with no sources

    """

    def loop(self, d_idx, d_x, d_y, d_au, d_av, t=0.0):
        # compute the acceleration and set it for the destination
        import numpy as np
        omega = 1.0
        r = np.sqrt(d_x[d_idx] * d_x[d_idx] + d_y[d_idx] * d_y[d_idx])
        d_au[d_idx] = np.sin(omega * t / (2.0 * np.pi)) * r
        d_av[d_idx] = np.cos(omega * t / (2.0 * np.pi)) * r


def _get_interior(x, y):
    indices = []
    for i in range(x.size):
        if ((x[i] > 0.0) and (x[i] < Lx)):
            if ((y[i] > 0.0) and (y[i] < Ly)):
                indices.append(i)

    return indices


def _get_obstacle(x, y):
    indices = []
    for i in range(x.size):
        if ((1.0 <= x[i] <= 2.0) and (2.0 <= y[i] <= 3.0)):
            indices.append(i)

    return indices


class WashingMachine(Application):

	# def _setup_particle_properties(self, particles, volume):
	# 	wm, wall, fluid = particles

	# 	fluid.add_property('V')
	# 	wall.add_property('V' )
	# 	wm.add_property('V' )

	# 	for name in ['uf', 'vf', 'wf']:
	# 		wall.add_property(name)
	# 		wm.add_property(name)

	# 	for name in ['ug', 'vg', 'wg']:
	# 		wall.add_property(name)
	# 		wm.add_property(name)

	# 	for name in ('uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 'au', 'av', 'aw'):
	# 		fluid.add_property(name)

	# 	wall.add_property('wij')
	# 	wm.add_property('wij')

	# 	wall.add_property('u0'); wall.u0[:] = 0.
	# 	wall.add_property('v0'); wall.v0[:] = 0.
	# 	wall.add_property('w0'); wall.w0[:] = 0.

	# 	wall.add_property('x0')
	# 	wall.add_property('y0')
	# 	wall.add_property('z0')

	# 	wm.add_property('ax')
	# 	wm.add_property('ay')
	# 	wm.add_property('az')

	# 	wall.add_property('ax')
	# 	wall.add_property('ay')
	# 	wall.add_property('az')

	# 	fluid.add_property('vmag2')

	# 	fluid.V[:] = 1./volume
	# 	wall.V[:] = 1./volume
	# 	wm.V[:] = 1./volume

	# 	fluid.set_output_arrays( ['x', 'y', 'u', 'v', 'vmag2', 'rho', 'p', 
	# 							  'V', 'm', 'h'] )

	# 	wm.set_output_arrays( ['x', 'y', 'rho', 'p'] )
	# 	wall.set_output_arrays( ['x', 'y', 'u0', 'rho', 'p', 'u'] )

	# 	particles = [wm, wall, fluid]
	# 	return particles

	def create_particles(self):
		volume = 0.05 * 0.05 * 0.05
		wm, wall, fluid = washing_machine_model(dx_solid=0.05, dx_fluid=0.05)
		fluid.add_property('V'); fluid.V[:] = 1./volume
		wall.add_property('V'); wall.V[:] = 1./volume
		wm.add_property('V'); wm.V[:] = 1./volume
		for name in ['uf', 'vf', 'wf', 'wij']:
			wall.add_property(name)
			wm.add_property(name)

		for name in ['ug', 'vg', 'wg']:
			wall.add_property(name)
			wm.add_property(name)
		for name in ('uhat', 'vhat', 'what', 'auhat', 'avhat',
					 'awhat', 'au', 'av', 'aw'):
			fluid.add_property(name)
		fluid.add_property('vmag2')
		particles = [wm, wall, fluid]
		return particles

	def create_solver(self):
		kernel = CubicSpline(dim=3)
		integrator = Integrator(fluid=TransportVelocityStep())

		solver = Solver(kernel=kernel, dim=3, integrator=integrator,
						tf=10.0, dt=1.0e-04, adaptive_timestep=False,
						output_at_times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
		return solver

	def create_equations(self):
		from pysph.sph.wc.basic import TaitEOS
		from pysph.sph.basic_equations import XSPHCorrection
		from pysph.sph.wc.transport_velocity import (ContinuityEquation, 
			MomentumEquationPressureGradient, MomentumEquationViscosity, 
			MomentumEquationArtificialViscosity, SolidWallPressureBC, 
			SolidWallNoSlipBC, SetWallVelocity, VolumeSummation)

		all = ['wm', 'wall', 'fluid']
		fluids = ['fluid']

		eqns = [
            Group(
                equations=[
                    SPHERICBenchmarkAcceleration(dest='wall', sources=None),
                ], real=False
            ),

            Group(
                equations=[
					VolumeSummation(dest='fluid', sources=all),
					TaitEOS(dest='fluid', sources=None, rho0=100.0, 
							c0=7.0, gamma=7.0),
					VolumeSummation(dest='wm', sources=all),
					SetWallVelocity(dest='wm', sources=fluids),
					VolumeSummation(dest='wall', sources=all),
					SetWallVelocity(dest='wall', sources=fluids)
                ], real=False
            ),

            Group(
                equations=[
                    SolidWallPressureBC(
                        dest='wm', sources=fluids, b=1.0, rho0=100.0, p0=0.0),
                    SolidWallPressureBC(
                        dest='wall', sources=fluids, b=1.0, rho0=100.0, p0=0.0),
                ], real=False
            ),

            Group(
                equations=[
                    ContinuityEquation(dest='fluid', sources=all),
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=all, pb=0.0, gz=-9.81, tdamp=0.0),
                    XSPHCorrection(dest='fluid', sources=fluids),

                ], real=True
            ),
        ]
		return eqns


if __name__ == '__main__':
    app = WashingMachine()
    app.run()
