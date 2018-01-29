"""Demonstrate the windtunnel simulation using inlet and outlet feature in 2D

               inlet       fluid       outlet
              ---------    --------    --------
             | * * * x |  |        |  |        |
     u       | * * * x |  |        |  |        |
    --->     | * * * x |  | airfoil|  |        |
             | * * * x |  |        |  |        |
              --------     --------    --------

In the figure above, the 'x' are the initial inlet particles.  The '*' are the
copies of these.  The particles are moving to the right and as they do, new
fluid particles are added and as the fluid particles flow into the outlet they
are converted to the outlet particle array and at last as the particles leave
the outlet they are removed from the simulation.  The `create_particles` and
`create_inlet_outlet` functions may also be passed to the `app.setup` method
if needed.

This example can be run in parallel.

"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_wcsph
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator
from pysph.sph.simple_inlet_outlet import SimpleInlet, SimpleOutlet
from pysph.sph.integrator_step import InletOutletStep, TransportVelocityStep
from pysph.sph.integrator_step import InletOutletStep, WCSPHStep

from pysph.sph.scheme import TVFScheme
from pysph.sph.scheme import WCSPHScheme

from pysph.tools.geometry import get_2d_wall, get_2d_block
from pysph.tools.geometry import get_5digit_naca_airfoil
from pysph.tools.geometry import get_4digit_naca_airfoil
from pysph.tools.geometry import remove_overlap_particles


def windtunnel_airfoil_model(dx_wall=0.01, dx_airfoil=0.01, dx_fluid=0.01,
                             r_solid=100.0, r_fluid=100.0, airfoil='2412',
                             hdx=1.1, chord=1.0, h_tunnel=1.0, l_tunnel=10.0):
    """
    Generates a geometry which can be used for wind tunnel like simulations.

    Parameters
    ----------
    dx_wall : a number which is the dx of the wind tunnel wall
    dx_airfoil : a number which is the dx of the airfoil used
    dx_fluid : a number which is the dx of the fluid used
    r_solid : a number which is the initial density of the solid particles
    r_fluid : a number which is the initial density of the fluid particles
    airfoil : 4 or 5 digit string which is the airfoil name
    hdx : a number which is the hdx for the particle arrays
    chord : a number which is the chord of the airfoil
    h_tunnel : a number which is the height of the wind tunnel
    l_tunnel : a number which is the length of the wind tunnel

    Returns
    -------
    wall : pysph wcsph particle array for the wind tunnel walls
    wing : pysph wcsph particle array for the airfoil
    fluid : pysph wcsph particle array for the fluid
    """

    wall_center_1 = np.array([0.0, h_tunnel / 2.])
    wall_center_2 = np.array([0.0, -h_tunnel / 2.])
    x_wall_1, y_wall_1 = get_2d_wall(dx_wall, wall_center_1, l_tunnel)
    x_wall_2, y_wall_2 = get_2d_wall(dx_wall, wall_center_2, l_tunnel)
    x_wall = np.concatenate([x_wall_1, x_wall_2])
    y_wall = np.concatenate([y_wall_1, y_wall_2])
    y_wall_1 = y_wall_1 + dx_wall
    y_wall_2 = y_wall_2 - dx_wall
    y_wall = np.concatenate([y_wall, y_wall_1, y_wall_2])
    y_wall_1 = y_wall_1 + dx_wall
    y_wall_2 = y_wall_2 - dx_wall
    y_wall = np.concatenate([y_wall, y_wall_1, y_wall_2])
    y_wall_1 = y_wall_1 + dx_wall
    y_wall_2 = y_wall_2 - dx_wall
    x_wall = np.concatenate([x_wall, x_wall, x_wall, x_wall])
    y_wall = np.concatenate([y_wall, y_wall_1, y_wall_2])
    h_wall = np.ones_like(x_wall) * dx_wall * hdx
    rho_wall = np.ones_like(x_wall) * r_solid
    mass_wall = rho_wall * dx_wall * dx_wall
    wall = get_particle_array_wcsph(name='wall', x=x_wall, y=y_wall, h=h_wall,
                                    rho=rho_wall, m=mass_wall)
    if len(airfoil) == 4:
        x_airfoil, y_airfoil = get_4digit_naca_airfoil(
            dx_airfoil, airfoil, chord)
    else:
        x_airfoil, y_airfoil = get_5digit_naca_airfoil(
            dx_airfoil, airfoil, chord)
    x_airfoil = x_airfoil - 0.5
    h_airfoil = np.ones_like(x_airfoil) * dx_airfoil * hdx
    rho_airfoil = np.ones_like(x_airfoil) * r_solid
    mass_airfoil = rho_airfoil * dx_airfoil * dx_airfoil
    wing = get_particle_array_wcsph(name='wing', x=x_airfoil, y=y_airfoil,
                                    h=h_airfoil, rho=rho_airfoil,
                                    m=mass_airfoil)
    x_fluid, y_fluid = get_2d_block(dx_fluid, 1.6, h_tunnel)
    h_fluid = np.ones_like(x_fluid) * dx_fluid * hdx
    rho_fluid = np.ones_like(x_fluid) * r_fluid
    mass_fluid = rho_fluid * dx_fluid * dx_fluid
    fluid = get_particle_array_wcsph(name='fluid', x=x_fluid, y=y_fluid,
                                     h=h_fluid, rho=rho_fluid, m=mass_fluid)
    remove_overlap_particles(fluid, wall, dx_wall, 2)
    remove_overlap_particles(fluid, wing, dx_airfoil, 2)
    return wall, wing, fluid


class WindTunnel(Application):

    def add_user_options(self, group):
        group.add_argument("--speed", action="store",
                           type=float,
                           dest="speed",
                           default=1.0,
                           help="Speed of inlet particles.")

    def create_particles(self):
        dx_airfoil = 0.002
        dx_wall = 0.002
        wall, wing, fluid = windtunnel_airfoil_model(
            dx_wall=dx_wall, dx_airfoil=dx_airfoil)
        outlet = get_particle_array_wcsph(name='outlet')

        dx = 0.01
        y = np.linspace(-0.49, 0.49, 99)
        x = np.zeros_like(y) - 0.81
        rho = np.ones_like(x) * 100.0
        m = rho * dx * dx
        h = np.ones_like(x) * dx * 1.1

        u = np.ones_like(x) * self.options.speed

        inlet = get_particle_array_wcsph(name='inlet', x=x, y=y, m=m, h=h,
                                         u=u, rho=rho)

        return [inlet, fluid, wing, outlet, wall]

    def create_inlet_outlet(self, particle_arrays):
        fluid_pa = particle_arrays['fluid']
        inlet_pa = particle_arrays['inlet']
        outlet_pa = particle_arrays['outlet']

        inlet = SimpleInlet(
            inlet_pa, fluid_pa, spacing=0.01, n=19, axis='x', xmin=-1.00,
            xmax=-0.81, ymin=-0.49, ymax=0.49
        )
        outlet = SimpleOutlet(
            outlet_pa, fluid_pa, xmin=3.0, xmax=4.0, ymin=-0.5, ymax=0.5
        )
        return [inlet, outlet]

    def create_scheme(self):
        s = WCSPHScheme(['fluid', 'inlet', 'outlet'], ['wing', 'wall'],
                        dim=2, rho0=100.0, c0=10.0, h0=0.011, hdx=1.1,
                        hg_correction=True)
        return s

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = PECIntegrator(
            fluid=WCSPHStep(), inlet=InletOutletStep(),
            outlet=InletOutletStep()
        )

        dt = 0.00005
        tf = 20.0

        solver = Solver(
            kernel=kernel, dim=2, integrator=integrator, dt=dt, tf=tf,
            adaptive_timestep=False, pfreq=20
        )
        return solver


if __name__ == '__main__':
    app = WindTunnel()
    app.run()
