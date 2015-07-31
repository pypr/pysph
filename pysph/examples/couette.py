"""Couette flow using the transport velocity formulation (30 seconds).
"""

import os
import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian
from pysph.solver.solver import Solver
from pysph.solver.utils import load
from pysph.solver.application import Application
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.tools.interpolator import Interpolator

# the eqations
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import (SummationDensity,
    SetWallVelocity, StateEquation,
    MomentumEquationPressureGradient, MomentumEquationViscosity,
    MomentumEquationArtificialStress,
    SolidWallPressureBC, SolidWallNoSlipBC)


# domain and reference values
Re = 0.0125
d = 0.5; Ly = 2*d; Lx = 0.4*Ly
rho0 = 1.0; nu = 0.01

# upper wall velocity based on the Reynolds number and channel width
Vmax = nu*Re/(2*d)
print(Vmax)
c0 = 10*Vmax; p0 = c0*c0*rho0

# Numerical setup
dx = 0.05
ghost_extent = 5 * dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0/( c0 + Vmax )
dt_viscous = 0.125 * h0**2/nu
dt_force = 1.0

tf = 100.0
dt = 0.5 * min(dt_cfl, dt_viscous, dt_force)

class CouetteFlow(Application):
    def create_domain(self):
        return DomainManager(xmin=0, xmax=Lx, periodic_in_x=True)

    def create_particles(self):
        _x = np.arange( dx/2, Lx, dx )

        # create the fluid particles
        _y = np.arange( dx/2, Ly, dx )

        x, y = np.meshgrid(_x, _y); fx = x.ravel(); fy = y.ravel()

        # create the channel particles at the top
        _y = np.arange(Ly+dx/2, Ly+dx/2+ghost_extent, dx)
        x, y = np.meshgrid(_x, _y); tx = x.ravel(); ty = y.ravel()

        # create the channel particles at the bottom
        _y = np.arange(-dx/2, -dx/2-ghost_extent, -dx)
        x, y = np.meshgrid(_x, _y); bx = x.ravel(); by = y.ravel()

        # concatenate the top and bottom arrays
        cx = np.concatenate( (tx, bx) )
        cy = np.concatenate( (ty, by) )

        # create the arrays
        channel = get_particle_array(name='channel', x=cx, y=cy)
        fluid = get_particle_array(name='fluid', x=fx, y=fy)

        print("Couette flow :: Re = %g, nfluid = %d, nchannel=%d, dt = %g"%(
            Re, fluid.get_number_of_particles(),
            channel.get_number_of_particles(), dt))

        # add requisite properties to the arrays:
        # particle volume
        fluid.add_property('V')
        channel.add_property('V')

        # advection velocities and accelerations
        for name in ('uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 'au', 'av', 'aw'):
            fluid.add_property(name)

        # kernel summation correction for the channel
        channel.add_property('wij')

        channel.add_property('ax')
        channel.add_property('ay')
        channel.add_property('az')

        # extrapolated velocities for the channel
        for name in ['uf', 'vf', 'wf']:
            channel.add_property(name)

        # dummy velocities for the channel
        # required for the no-slip BC
        for name in ['ug','vg','wg']:
            channel.add_property(name)

        # magnitude of velocity
        fluid.add_property('vmag2')

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * rho0
        channel.m[:] = volume * rho0

        # volume is set as dx^2
        fluid.V[:] = 1./volume
        channel.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = hdx * dx
        channel.h[:] = hdx * dx

        # channel velocity on upper portion
        indices = np.where(channel.y > d)[0]
        channel.u[indices] = Vmax

        # load balancing props
        fluid.set_lb_props( list(fluid.properties.keys()) )
        channel.set_lb_props( list(channel.properties.keys()) )

        # return the particle list
        return [fluid, channel]

    def create_solver(self):
        kernel = Gaussian(dim=2)
        integrator = PECIntegrator(fluid=TransportVelocityStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator)

        # Setup default parameters.
        solver.set_time_step(dt)
        solver.set_final_time(tf)
        return solver

    def create_equations(self):
        equations = [

            # Summation density along with volume summation for the fluid
            # phase. This is done for all local and remote particles. At the
            # end of this group, the fluid phase has the correct density
            # taking into consideration the fluid and solid
            # particles.
            Group(
                equations=[
                    SummationDensity(dest='fluid', sources=['fluid','channel']),
                    ], real=False),

            # Once the fluid density is computed, we can use the EOS to set
            # the fluid pressure. Additionally, the dummy velocity for the
            # channel is set, which is later used in the no-slip wall BC.
            Group(
                equations=[
                    StateEquation(dest='fluid', sources=None, p0=p0, rho0=rho0, b=1.0),
                    SetWallVelocity(dest='channel', sources=['fluid']),
                    ], real=False),

            # Once the pressure for the fluid phase has been updated, we can
            # extrapolate the pressure to the ghost particles. After this
            # group, the fluid density, pressure and the boundary pressure has
            # been updated and can be used in the integration equations.
            Group(
                equations=[
                    SolidWallPressureBC(dest='channel', sources=['fluid'],
                                        b=1.0, rho0=rho0, p0=p0),
                    ], real=False),

            # The main accelerations block. The acceleration arrays for the
            # fluid phase are upadted in this stage for all local particles.
            Group(
                equations=[
                    # Pressure gradient terms
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid', 'channel'], pb=p0),

                    # fluid viscosity
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid'], nu=nu),

                    # No-slip boundary condition. This is effectively a
                    # viscous interaction of the fluid with the ghost
                    # particles.
                    SolidWallNoSlipBC(
                        dest='fluid', sources=['channel'], nu=nu),

                    # Artificial stress for the fluid phase
                    MomentumEquationArtificialStress(dest='fluid', sources=['fluid']),

                    ], real=True),
        ]
        return equations


    def post_process(self, info_fname):
        info = self.read_info(info_fname)

        self._plot_u_vs_y()
        self._plot_ke_history()

    def _plot_ke_history(self):
        from pysph.tools.pprocess import get_ke_history
        from matplotlib import pyplot as plt
        t, ke = get_ke_history(self.output_files, 'fluid')
        plt.clf()
        plt.plot(t, ke)
        plt.xlabel('t'); plt.ylabel('Kinetic energy')
        fig = os.path.join(self.output_dir, "ke_history.png")
        plt.savefig(fig, dpi=300)

    def _plot_u_vs_y(self):
        files = self.output_files
        # interpolate the u-velocity profile along the centerline
        y = np.linspace(0, 1, 101)
        x = np.ones_like(y) * 0.2
        h = np.ones_like(y) * 1.5 * 0.01

        # take the last solution data
        fname = files[-1]
        data = load(fname)
        tf = data['solver_data']['t']

        interp = Interpolator(list(data['arrays'].values()), x=x, y=y)
        ui = interp.interpolate('u')

        # exact parabolic profile for the u-velocity
        ue = Vmax*y/Ly
        from matplotlib import pyplot as plt
        plt.clf()
        plt.plot(y, ue, label="exact")
        plt.plot(y, ui, label="computed")
        plt.xlabel('y'); plt.ylabel('u')
        plt.legend()
        plt.title('Velocity profile at %s'%tf)
        fig = os.path.join(self.output_dir, "comparison.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = CouetteFlow()
    app.run()
    app.post_process(app.info_filename)
