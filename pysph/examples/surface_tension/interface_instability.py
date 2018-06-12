import numpy as np
import os

from pysph.sph.wc.transport_velocity import SummationDensity, \
    MomentumEquationPressureGradient, StateEquation,\
    MomentumEquationArtificialStress, MomentumEquationViscosity

from pysph.sph.surface_tension import InterfaceCurvatureFromNumberDensity, \
    ShadlooYildizSurfaceTensionForce, CSFSurfaceTensionForce, \
    SmoothedColor, AdamiColorGradient, MorrisColorGradient, \
    SY11DiracDelta, SY11ColorGradient, MomentumEquationViscosityAdami, \
    AdamiReproducingDivergence, CSFSurfaceTensionForceAdami,\
    MomentumEquationPressureGradientAdami, ColorGradientAdami, \
    ConstructStressMatrix, SurfaceForceAdami, SummationDensitySourceMass, \
    MomentumEquationViscosityMorris, MomentumEquationPressureGradientMorris, \
    InterfaceCurvatureFromDensity

from pysph.sph.wc.basic import TaitEOS
from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

from pysph.tools.geometry import get_2d_block
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, QuinticSpline
from pysph.sph.equation import Group, Equation

from pysph.solver.application import Application
from pysph.solver.solver import Solver

from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import PECIntegrator

from pysph.base.nnps import DomainManager
from pysph.solver.utils import iter_output


dim = 2
Lx = 0.5
Ly = 1.0
factor1 = 0.8
factor2 = 1.0/factor1
nu = 0.0
sigma = 1.0
rho0 = 1.

c0 = 20.0
gamma = 1.4
R = 287.1

p0 = c0**2 * rho0

nx = 50
dx = Lx / nx
volume = dx * dx
hdx = 1.5

h0 = hdx * dx

tf = 0.5

epsilon = 0.01 / h0

KE = 10**(-6.6)*p0*p0*gamma/(c0 * c0 * rho0 * rho0 * nx * nx * (gamma - 1))

Vmax = np.sqrt(2 * KE / (rho0 * dx * dx))

dt1 = 0.25*np.sqrt(rho0*h0*h0*h0/(2.0*np.pi*sigma))

dt2 = 0.25*h0/(c0+Vmax)

dt = 0.9*min(dt1, dt2)


class MultiPhase(Application):
    def add_user_options(self, group):
        choices = ['morris', 'tvf', 'adami_stress', 'adami', 'shadloo']
        group.add_argument(
            "--scheme", action="store", dest='scheme', default='morris',
            choices=choices,
            help='Specify scheme to use among %s' % choices
        )

    def create_particles(self):
        fluid_x, fluid_y = get_2d_block(
            dx=dx, length=Lx-dx, height=Ly-dx, center=np.array([0., 0.5*Ly]))
        rho_fluid = np.ones_like(fluid_x) * rho0
        m_fluid = rho_fluid * volume
        h_fluid = np.ones_like(fluid_x) * h0
        cs_fluid = np.ones_like(fluid_x) * c0
        additional_props = ['V', 'color', 'scolor', 'cx', 'cy', 'cz',
                            'cx2', 'cy2', 'cz2', 'nx', 'ny', 'nz', 'ddelta',
                            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',
                            'ax', 'ay', 'az', 'wij', 'vmag2', 'N', 'wij_sum',
                            'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                            'kappa', 'arho', 'nu', 'pi00', 'pi01', 'pi10', 
                            'pi11']
        fluid = get_particle_array(
            name='fluid', x=fluid_x, y=fluid_y, h=h_fluid, m=m_fluid,
            rho=rho_fluid, cs=cs_fluid, additional_props=additional_props)
        for i in range(len(fluid.x)):
            if fluid.y[i] > 0.25 and fluid.y[i] < 0.75:
                fluid.color[i] = 1.0
            else:
                fluid.color[i] = 0.0
        fluid.V[:] = 1. / volume
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 
                                 'ddelta', 'kappa', 'N', 'scolor', 'p'])
        angles = np.random.random_sample((len(fluid.x),))*2*np.pi
        vel = np.sqrt(2 * KE / fluid.m)
        fluid.u = vel
        fluid.v = vel
        fluid.nu[:] = 0.0
        return [fluid]

    def create_domain(self):
        return DomainManager(
            xmin=-0.5 * Lx, xmax=0.5 * Lx, ymin=0.0, ymax=Ly,
            periodic_in_x=True, periodic_in_y=True, n_layers=6)

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = PECIntegrator(fluid=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False)
        return solver

    def create_equations(self):
        sy11_equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid'])
            ], real=False),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
                SY11ColorGradient(dest='fluid', sources=['fluid'])
            ], real=False),
            Group(equations=[
                ScaleSmoothingLength(dest='fluid', sources=None,
                                     factor=factor1)
            ], real=False, update_nnps=True),
            Group(equations=[
                SY11DiracDelta(dest='fluid', sources=['fluid'])
            ], real=False
            ),
            Group(equations=[
                InterfaceCurvatureFromNumberDensity(
                    dest='fluid', sources=['fluid'],
                    with_morris_correction=True),
            ], real=False),
            Group(equations=[
                ScaleSmoothingLength(dest='fluid', sources=None,
                                     factor=factor2)
            ], real=False, update_nnps=True,
            ),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid'], pb=0.0),
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid'], nu=nu),
                    ShadlooYildizSurfaceTensionForce(dest='fluid',
                                                     sources=None,
                                                     sigma=sigma),
                ], )
        ]

        adami_equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid'])
            ], real=False),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
            ], real=False),
            Group(equations=[
                AdamiColorGradient(dest='fluid', sources=['fluid']),
            ], real=False
            ),
            Group(equations=[
                AdamiReproducingDivergence(dest='fluid', sources=['fluid'],
                                           dim=2),
            ], real=False),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid'], pb=p0),
                    MomentumEquationViscosityAdami(
                        dest='fluid', sources=['fluid']),
                    CSFSurfaceTensionForceAdami(dest='fluid', sources=None,)
                ], )
        ]

        adami_stress_equations = [
            Group(equations=[
                SummationDensity(
                    dest='fluid', sources=[
                        'fluid']),
            ], real=False),
            Group(equations=[
                TaitEOS(dest='fluid', sources=None,
                        rho0=rho0, c0=c0, gamma=7, p0=p0),
            ], real=False),
            Group(equations=[
                ColorGradientAdami(dest='fluid', sources=['fluid']),
            ], real=False),
            Group(equations=[ConstructStressMatrix(
                dest='fluid', sources=None, sigma=sigma, d=2)], real=False),
            Group(
                equations=[
                    MomentumEquationPressureGradientAdami(
                        dest='fluid', sources=['fluid']),
                    MomentumEquationViscosityAdami(
                        dest='fluid', sources=['fluid']),
                    SurfaceForceAdami(
                        dest='fluid', sources=['fluid']),
                ]),
        ]

        tvf_equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid'])
            ], real=False),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
                SmoothedColor(dest='fluid', sources=['fluid']),
            ], real=False),
            Group(equations=[
                MorrisColorGradient(dest='fluid', sources=['fluid'],
                                    epsilon=epsilon),
            ], real=False
            ),
            Group(equations=[
                InterfaceCurvatureFromNumberDensity(
                    dest='fluid', sources=['fluid'],
                    with_morris_correction=True),
            ], real=False),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid'], pb=p0),
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid'], nu=nu),
                    CSFSurfaceTensionForce(dest='fluid', sources=None,
                                           sigma=sigma),
                    MomentumEquationArtificialStress(dest='fluid',
                                                     sources=['fluid']),
                ], )
        ]

        morris_equations = [
            Group(equations=[
                SummationDensitySourceMass(
                    dest='fluid', sources=[
                        'fluid']),
            ], real=False, update_nnps=False),
            Group(equations=[
                TaitEOS(dest='fluid', sources=None, rho0=rho0, c0=c0, 
                        gamma=1.0),
                SmoothedColor(
                    dest='fluid', sources=['fluid', ]),
                ScaleSmoothingLength(dest='fluid', sources=None,
                                     factor=2.0/3.0),
            ], real=False, update_nnps=False),
            Group(equations=[
                MorrisColorGradient(dest='fluid', sources=['fluid', ],
                                    epsilon=epsilon),
            ], real=False, update_nnps=False),
            Group(equations=[
                InterfaceCurvatureFromDensity(dest='fluid', sources=['fluid'],
                                              with_morris_correction=True),
                ScaleSmoothingLength(dest='fluid', sources=None, factor=1.5),
            ], real=False, update_nnps=False),
            Group(
                equations=[
                    MomentumEquationPressureGradientMorris(
                        dest='fluid', sources=['fluid']),
                    MomentumEquationViscosityMorris(
                        dest='fluid', sources=['fluid']),
                    CSFSurfaceTensionForce(
                        dest='fluid', sources=None, sigma=sigma),
                ], update_nnps=False)
        ]

        if self.options.scheme == 'tvf':
            return tvf_equations
        elif self.options.scheme == 'adami_stress':
            return adami_stress_equations
        elif self.options.scheme == 'adami':
            return adami_equations
        elif self.options.scheme == 'shadloo':
            return sy11_equations
        else:
            return morris_equations

    def post_process(self):
        import matplotlib.pyplot as plt
        from pysph.solver.utils import load
        files = self.output_files
        ke = []
        t = []
        for f in files:
            data = load(f)
            pa = data['arrays']['fluid']
            t.append(data['solver_data']['t'])
            m = pa.m
            u = pa.u
            v = pa.v
            length = len(m)
            ke.append(np.log10(sum(0.5 * m * (u**2 + v**2) / length)))
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, t=t, ke=ke)        
        plt.plot(t, ke, 'o')
        fig = os.path.join(self.output_dir, "KEvst.png")
        plt.savefig(fig)
        plt.close()


if __name__ == '__main__':
    app = MultiPhase()
    app.run()
    app.post_process()
