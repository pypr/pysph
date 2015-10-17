"""Lid driven cavity using the Transport Velocity formulation. (10 minutes)
"""

import os

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import TransportVelocityStep

# the eqations
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import SummationDensity,\
    StateEquation, MomentumEquationPressureGradient, MomentumEquationViscosity,\
    MomentumEquationArtificialStress, SolidWallPressureBC, SolidWallNoSlipBC,\
    SetWallVelocity

# numpy
import numpy as np

# domain and reference values
L = 1.0; Umax = 1.0
c0 = 10 * Umax
rho0 = 1.0
p0 = c0*c0*rho0

# Numerical setup
hdx = 1.0


class LidDrivenCavity(Application):
    def add_user_options(self, group):
        group.add_option(
            "--nx", action="store", type=int, dest="nx", default=50,
            help="Number of points along x direction."
        )
        group.add_option(
            "--re", action="store", type=float, dest="re", default=100,
            help="Reynolds number (defaults to 100)."
        )

    def consume_user_options(self):
        nx = self.options.nx
        self.dx = L/nx
        self.re = self.options.re
        h0 = hdx * self.dx
        self.nu = Umax*L/self.re
        dt_cfl = 0.25 * h0/( c0 + Umax )
        dt_viscous = 0.125 * h0**2/self.nu
        dt_force = 1.0

        self.tf = 10.0
        self.dt = 0.75 * min(dt_cfl, dt_viscous, dt_force)

    def create_particles(self):
        dx = self.dx
        ghost_extent = 5 * dx
        # create all the particles
        _x = np.arange( -ghost_extent - dx/2, L + ghost_extent + dx/2, dx )
        x, y = np.meshgrid(_x, _x); x = x.ravel(); y = y.ravel()

        # sort out the fluid and the solid
        indices = []
        for i in range(x.size):
            if ( (x[i] > 0.0) and (x[i] < L) ):
                if ( (y[i] > 0.0) and (y[i] < L) ):
                    indices.append(i)

        # create the arrays
        solid = get_particle_array(name='solid', x=x, y=y)

        # remove the fluid particles from the solid
        fluid = solid.extract_particles(indices); fluid.set_name('fluid')
        solid.remove_particles(indices)

        print("Lid driven cavity :: Re = %d, nfluid = %d, nsolid=%d, dt = %g"%(
            self.re, fluid.get_number_of_particles(),
            solid.get_number_of_particles(), self.dt))

        # add requisite properties to the arrays:

        # particle volume for fluid and solid
        fluid.add_property('V')
        solid.add_property('V' )

        # extrapolated velocities for the solid
        for name in ['uf', 'vf', 'wf']:
            solid.add_property(name)

        # dummy velocities for the solid wall
        # required for the no-slip BC
        for name in ['ug', 'vg', 'wg']:
            solid.add_property(name)

        # advection velocities and accelerations
        for name in ('uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 'au', 'av', 'aw'):
            fluid.add_property(name)

        # kernel summation correction for the solid
        solid.add_property('wij')

        # imposed accelerations on the solid
        solid.add_property('ax')
        solid.add_property('ay')
        solid.add_property('az')

        # magnitude of velocity
        fluid.add_property('vmag2')

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * rho0
        solid.m[:] = volume * rho0

        # volume is set as dx^2
        fluid.V[:] = 1./volume
        solid.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = hdx * dx
        solid.h[:] = hdx * dx

        # imposed horizontal velocity on the lid
        solid.u[:] = 0.0
        solid.v[:] = 0.0
        for i in range(solid.get_number_of_particles()):
            if solid.y[i] > L:
                solid.u[i] = Umax

        # set the output arrays
        fluid.set_output_arrays( ['x', 'y', 'u', 'v', 'vmag2', 'rho', 'p',
                                  'V', 'm', 'h', 'gid'] )

        solid.set_output_arrays( ['x', 'y', 'u', 'rho', 'p', 'gid'] )

        return [fluid, solid]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)

        integrator = PECIntegrator(fluid=TransportVelocityStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        tf=self.tf, dt=self.dt, pfreq=1000,
                        adaptive_timestep=False)
        return solver

    def create_equations(self):
        nu = self.nu
        equations = [

            # Summation density along with volume summation for the fluid
            # phase. This is done for all local and remote particles. At the
            # end of this group, the fluid phase has the correct density
            # taking into consideration the fluid and solid
            # particles.
            Group(
                equations=[
                    SummationDensity(dest='fluid', sources=['fluid','solid']),
                    ], real=False),


            # Once the fluid density is computed, we can use the EOS to set
            # the fluid pressure. Additionally, the dummy velocity for the
            # channel is set, which is later used in the no-slip wall BC.
            Group(
                equations=[
                    StateEquation(dest='fluid', sources=None, p0=p0, rho0=rho0, b=1.0),
                    SetWallVelocity(dest='solid', sources=['fluid']),
                    ], real=False),

            # Once the pressure for the fluid phase has been updated, we can
            # extrapolate the pressure to the ghost particles. After this
            # group, the fluid density, pressure and the boundary pressure has
            # been updated and can be used in the integration equations.
            Group(
                equations=[
                    SolidWallPressureBC(dest='solid', sources=['fluid'], b=1.0, rho0=rho0, p0=p0),
                    ], real=False),

            # The main accelerations block. The acceleration arrays for the
            # fluid phase are upadted in this stage for all local particles.
            Group(
                equations=[
                    # Pressure gradient terms
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid', 'solid'], pb=p0),

                    # fluid viscosity
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid'], nu=nu),

                    # No-slip boundary condition. This is effectively a
                    # viscous interaction of the fluid with the ghost
                    # particles.
                    SolidWallNoSlipBC(
                        dest='fluid', sources=['solid'], nu=nu),

                    # Artificial stress for the fluid phase
                    MomentumEquationArtificialStress(dest='fluid', sources=['fluid']),

                    ], real=True),
        ]
        return equations

    def post_process(self, info_fname):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        if self.rank > 0:
            return
        info = self.read_info(info_fname)
        t, ke = self._plot_ke_history()
        x, ui, vi, ui_c, vi_c = self._plot_velocity()
        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res, t=t, ke=ke, x=x, u=ui, v=vi, u_c=ui_c, v_c=vi_c)

    def _plot_ke_history(self):
        from pysph.tools.pprocess import get_ke_history
        from matplotlib import pyplot as plt
        t, ke = get_ke_history(self.output_files, 'fluid')
        plt.clf()
        plt.plot(t, ke)
        plt.xlabel('t'); plt.ylabel('Kinetic energy')
        fig = os.path.join(self.output_dir, "ke_history.png")
        plt.savefig(fig, dpi=300)
        return t, ke

    def _plot_velocity(self):
        from pysph.tools.interpolator import Interpolator
        from pysph.solver.utils import load
        from pysph.examples.ghia_cavity_data import get_u_vs_y, get_v_vs_x
        # interpolated velocities
        _x = np.linspace(0,1,101)
        xx, yy = np.meshgrid(_x, _x)

        # take the last solution data
        fname = self.output_files[-1]
        data = load(fname)
        tf = data['solver_data']['t']
        interp = Interpolator(list(data['arrays'].values()), x=xx, y=yy)

        ui = interp.interpolate('u')
        vi = interp.interpolate('v')

        ui.shape = 101,101
        vi.shape = 101,101

        # velocity magnitude
        self.vmag = vmag = np.sqrt( ui**2 + vi**2 )
        import matplotlib.pyplot as plt

        f = plt.figure()

        plt.streamplot(
            xx, yy, ui, vi, density=(2, 2), #linewidth=5*vmag/vmag.max(),
            color=vmag
        )
        plt.colorbar()
        plt.xlabel('$x$'); plt.ylabel('$y$')
        plt.title('Streamlines at %s seconds'%tf)
        fig = os.path.join(self.output_dir, 'streamplot.png')
        plt.savefig(fig, dpi=300)

        f = plt.figure()

        ui_c = ui[:, 50]
        vi_c = vi[50]

        s1 = plt.subplot(211)
        s1.plot(ui_c, _x, label='Computed')

        y, data = get_u_vs_y()
        if self.re in data:
            s1.plot(data[self.re], y, 'o', fillstyle='none',
                    label='Ghia et al.')
        s1.set_xlabel(r'$v_x$')
        s1.set_ylabel(r'$y$')
        s1.legend()

        s2 = plt.subplot(212)
        s2.plot(_x, vi_c, label='Computed')
        x, data = get_v_vs_x()
        if self.re in data:
            s2.plot(x, data[self.re], 'o', fillstyle='none',
                    label='Ghia et al.')
        s2.set_xlabel(r'$x$')
        s2.set_ylabel(r'$v_y$')
        s2.legend()

        fig = os.path.join(self.output_dir, 'centerline.png')
        plt.savefig(fig, dpi=300)
        return _x, ui, vi, ui_c, vi_c

if __name__ == '__main__':
    app = LidDrivenCavity()
    app.run()
    app.post_process(app.info_filename)
