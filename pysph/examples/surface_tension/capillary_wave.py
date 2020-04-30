from math import sqrt
import numpy as np
import os

from pysph.sph.wc.transport_velocity import SummationDensity, \
    MomentumEquationPressureGradient, StateEquation,\
    MomentumEquationArtificialStress, MomentumEquationViscosity, \
    SolidWallNoSlipBC

from pysph.sph.surface_tension import InterfaceCurvatureFromNumberDensity, \
    ShadlooYildizSurfaceTensionForce, CSFSurfaceTensionForce, \
    SmoothedColor, AdamiColorGradient, MorrisColorGradient, \
    SY11DiracDelta, SY11ColorGradient, MomentumEquationViscosityAdami, \
    AdamiReproducingDivergence, CSFSurfaceTensionForceAdami,\
    MomentumEquationPressureGradientAdami, ColorGradientAdami, \
    ConstructStressMatrix, SurfaceForceAdami, SummationDensitySourceMass, \
    MomentumEquationViscosityMorris, MomentumEquationPressureGradientMorris, \
    InterfaceCurvatureFromDensity, SolidWallPressureBCnoDensity


from pysph.sph.wc.basic import TaitEOS
from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

from pysph.tools.geometry import get_2d_block, remove_overlap_particles, \
    get_2d_circle
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
Lx = 1.0
Ly = 1.0

nu1 = 0.05
nu2 = 0.0005
sigma = 1.0
factor1 = 0.5
factor2 = 1 / factor1
rho1 = 1.0

c0 = 20.0
gamma = 1.4
R = 287.1
rho2 = 0.001

p1 = c0**2 * rho1
p2 = c0*c0*rho2
nx = 60
dx = Lx / nx
volume = dx * dx
hdx = 1.0

h0 = hdx * dx

tf = 0.5

epsilon = 0.01 / h0
v0 = 10.0
r0 = 0.05

dt1 = 0.25*np.sqrt(rho2*h0*h0*h0/(2.0*np.pi*sigma))

dt2 = 0.25*h0/(c0+v0)

dt3 = 0.125*rho2*h0*h0/nu2

dt = 0.9*min(dt1, dt2, dt3)


d = 2


def r(x, y):
    return x*x + y*y


class MultiPhase(Application):

    def create_particles(self):
        fluid_x, fluid_y = get_2d_block(
            dx=dx, length=Lx, height=Ly, center=np.array([0., 0.]))
        rho_fluid = np.ones_like(fluid_x) * rho2
        m_fluid = rho_fluid * volume
        h_fluid = np.ones_like(fluid_x) * h0
        cs_fluid = np.ones_like(fluid_x) * c0
        circle_x, circle_y = get_2d_circle(dx=dx, r=0.2,
                                           center=np.array([0.0, 0.0]))
        rho_circle = np.ones_like(circle_x) * rho1
        m_circle = rho_circle * volume
        h_circle = np.ones_like(circle_x) * h0
        cs_circle = np.ones_like(circle_x) * c0
        wall_x, wall_y = get_2d_block(dx=dx, length=Lx+6*dx, height=Ly+6*dx,
                                      center=np.array([0., 0.]))
        rho_wall = np.ones_like(wall_x) * rho2
        m_wall = rho_wall * volume
        h_wall = np.ones_like(wall_x) * h0
        cs_wall = np.ones_like(wall_x) * c0
        additional_props = ['V', 'color', 'scolor', 'cx', 'cy', 'cz',
                            'cx2', 'cy2', 'cz2', 'nx', 'ny', 'nz', 'ddelta',
                            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',
                            'ax', 'ay', 'az', 'wij', 'vmag2', 'N', 'wij_sum',
                            'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                            'kappa', 'arho', 'nu', 'wg', 'ug', 'vg',
                            'pi00', 'pi01', 'pi02', 'pi10', 'pi11', 'pi12',
                            'pi20', 'pi21', 'pi22']
        gas = get_particle_array(
            name='gas', x=fluid_x, y=fluid_y, h=h_fluid, m=m_fluid,
            rho=rho_fluid, cs=cs_fluid, additional_props=additional_props)
        gas.nu[:] = nu2
        gas.color[:] = 0.0
        liquid = get_particle_array(
            name='liquid', x=circle_x, y=circle_y, h=h_circle, m=m_circle,
            rho=rho_circle, cs=cs_circle, additional_props=additional_props)
        liquid.nu[:] = nu1
        liquid.color[:] = 1.0
        wall = get_particle_array(
            name='wall', x=wall_x, y=wall_y, h=h_wall, m=m_wall,
            rho=rho_wall, cs=cs_wall, additional_props=additional_props)
        wall.color[:] = 0.0
        remove_overlap_particles(wall, liquid, dx_solid=dx, dim=2)
        remove_overlap_particles(wall, gas, dx_solid=dx, dim=2)
        remove_overlap_particles(gas, liquid, dx_solid=dx, dim=2)
        gas.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta',
                               'kappa', 'N', 'scolor', 'p'])
        liquid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny',
                                  'ddelta', 'kappa', 'N', 'scolor', 'p'])
        wall.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny',
                                'ddelta', 'kappa', 'N', 'scolor', 'p'])
        for i in range(len(gas.x)):
            R = sqrt(r(gas.x[i], gas.y[i]) + 0.0001*gas.h[i]*gas.h[i])
            f = np.exp(-R/r0)
            gas.u[i] = v0*gas.x[i]*(1.0-(gas.y[i]*gas.y[i])/(r0*R))*f/r0
            gas.v[i] = -v0*gas.y[i]*(1.0-(gas.x[i]*gas.x[i])/(r0*R))*f/r0
        for i in range(len(liquid.x)):
            R = sqrt(r(liquid.x[i], liquid.y[i]) +
                     0.0001*liquid.h[i]*liquid.h[i])
            liquid.u[i] = v0*liquid.x[i] * \
                (1.0 - (liquid.y[i]*liquid.y[i])/(r0*R))*f/r0
            liquid.v[i] = -v0*liquid.y[i] * \
                (1.0 - (liquid.x[i]*liquid.x[i])/(r0*R))*f/r0
        return [liquid, gas, wall]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = PECIntegrator(liquid=TransportVelocityStep(),
                                   gas=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False)
        return solver

    def create_equations(self):
        adami_stress_equations = [
            Group(equations=[
                SummationDensity(
                    dest='liquid', sources=[
                        'liquid', 'wall', 'gas']),
                SummationDensity(
                    dest='gas', sources=[
                        'liquid', 'wall', 'gas']),
                SummationDensity(
                    dest='wall', sources=['liquid', 'wall', 'gas'])
            ]),
            Group(equations=[
                TaitEOS(dest='liquid', sources=None, rho0=rho1, c0=c0, gamma=1,
                        p0=p1),
                TaitEOS(dest='gas', sources=None,
                        rho0=rho2, c0=c0, gamma=1, p0=p1),
                SolidWallPressureBCnoDensity(dest='wall', sources=['liquid',
                                                                   'gas']),
            ]),
            Group(equations=[
                ColorGradientAdami(dest='liquid', sources=['liquid', 'wall',
                                                           'gas']),
                ColorGradientAdami(dest='gas', sources=[
                              'liquid', 'wall', 'gas']),
            ]),
            Group(equations=[ConstructStressMatrix(dest='liquid', sources=None,
                                                   sigma=sigma, d=2),
                             ConstructStressMatrix(dest='gas', sources=None,
                                                   sigma=sigma, d=2)]),
            Group(
                equations=[
                    MomentumEquationPressureGradientAdami(
                        dest='liquid', sources=['liquid', 'wall', 'gas']),
                    MomentumEquationPressureGradientAdami(
                        dest='gas', sources=['liquid', 'wall', 'gas']),
                    MomentumEquationViscosityAdami(
                        dest='liquid', sources=['liquid', 'gas']),
                    MomentumEquationViscosityAdami(
                        dest='gas', sources=['liquid', 'gas']),
                    SurfaceForceAdami(
                        dest='liquid', sources=['liquid', 'wall', 'gas']),
                    SurfaceForceAdami(
                        dest='gas', sources=['liquid', 'wall', 'gas']),
                    SolidWallNoSlipBC(dest='liquid', sources=['wall'], nu=nu1),
                    SolidWallNoSlipBC(dest='gas', sources=['wall'], nu=nu2),
                ]),
                ]
        return adami_stress_equations

    def post_process(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("Post processing requires Matplotlib")
            return
        from pysph.solver.utils import load
        files = self.output_files
        t = []
        centerx = []
        centery = []
        for f in files:
            data = load(f)
            pa = data['arrays']['liquid']
            t.append(data['solver_data']['t'])
            x = pa.x
            y = pa.y
            length = len(x)
            cx = 0
            cy = 0
            count = 0
            for i in range(length):
                if x[i] > 0 and y[i] > 0:
                    cx += x[i]
                    cy += y[i]
                    count += 1
                else:
                    continue
            # As the masses are all the same in this case
            centerx.append(cx/count)
            centery.append(cy/count)
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, t=t, centerx=centerx, centery=centery)
        plt.plot(t, centerx, label='x position')
        plt.plot(t, centery, label='y position')
        plt.legend()
        fig1 = os.path.join(self.output_dir, 'centerofmassposvst')
        plt.savefig(fig1)
        plt.close()


if __name__ == '__main__':
    app = MultiPhase()
    app.run()
    app.post_process()
