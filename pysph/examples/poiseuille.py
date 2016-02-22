"""Poiseuille flow using the transport velocity formulation (5 minutes).
"""
import os

# numpy
import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.solver.utils import load

from pysph.sph.scheme import TVFScheme


# Numerical setup
dx = 1.0/60.0
ghost_extent = 5 * dx
hdx = 1.0

# adaptive time steps
h0 = hdx * dx

class PoiseuilleFlow(Application):
    def initialize(self):
        self.d = 0.5
        self.Ly = 2*self.d
        self.Lx = 0.4*self.Ly
        self.rho0 = 1.0
        self.nu = 0.01

    def add_user_options(self, group):
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=0.0125,
            help="Reynolds number of flow."
        )
        group.add_argument(
            "--remesh", action="store", type=float, dest="remesh", default=0,
            help="Remeshing frequency (setting it to zero disables it)."
        )

    def consume_user_options(self):
        self.re = self.options.re
        self.Vmax = self.nu*self.re/(2*self.d)
        self.c0 = 10*self.Vmax
        self.p0 = self.c0**2*self.rho0

        # The body force is adjusted to give the Required Reynold's number
        # based on the steady state maximum velocity Vmax:
        # Vmax = fx/(2*nu)*(d^2) at the centerline

        self.fx = self.Vmax * 2*self.nu/(self.d**2)
        # Setup default parameters.
        dt_cfl = 0.25 * h0/( self.c0 + self.Vmax )
        dt_viscous = 0.125 * h0**2/self.nu
        dt_force = 0.25 * np.sqrt(h0/self.fx)

        self.dt = min(dt_cfl, dt_viscous, dt_force)

    def configure_scheme(self):
        tf = 100.0
        scheme = self.scheme
        scheme.configure(c0=self.c0, p0=self.p0, pb=self.p0, gx=self.fx)
        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=1000)
        print("dt = %g"%self.dt)

    def create_scheme(self):
        s = TVFScheme(
            ['fluid'], ['channel'], dim=2, rho0=self.rho0, c0=None,
            nu=self.nu, p0=None, pb=None, h0=h0, gx=None
        )
        return s

    def create_domain(self):
        return DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)

    def create_particles(self):
        Lx = self.Lx
        Ly = self.Ly
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

        print("Poiseuille flow :: Re = %g, nfluid = %d, nchannel=%d"%(
            self.re, fluid.get_number_of_particles(),
            channel.get_number_of_particles()))

        # add requisite properties to the arrays:
        self.scheme.setup_properties([fluid, channel])

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * self.rho0
        channel.m[:] = volume * self.rho0

        # Set the default rho.
        fluid.rho[:] = self.rho0
        channel.rho[:] = self.rho0

        # volume is set as dx^2
        fluid.V[:] = 1./volume
        channel.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = hdx * dx
        channel.h[:] = hdx * dx

        # return the particle list
        return [fluid, channel]

    def create_tools(self):
        tools = []
        if self.options.remesh > 0:
            from pysph.solver.tools import SimpleRemesher
            remesher = SimpleRemesher(
                self, 'fluid', props=['u', 'v', 'uhat', 'vhat'],
                freq=self.options.remesh
            )
            tools.append(remesher)
        return tools

    def post_process(self, info_fname):
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        import matplotlib
        matplotlib.use('Agg')

        y_ex, u_ex, y, u = self._plot_u_vs_y()
        t, ke = self._plot_ke_history()
        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res, t=t, ke=ke, y=y, u=u, y_ex=y_ex, u_ex=u_ex)

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

    def _plot_u_vs_y(self):
        files = self.output_files

        # take the last solution data
        fname = files[-1]
        data = load(fname)
        tf = data['solver_data']['t']
        fluid = data['arrays']['fluid']
        u = fluid.u.copy()
        y = fluid.y.copy()

        # exact parabolic profile for the u-velocity
        d = self.d
        fx = self.fx
        nu = self.nu

        ye = np.arange(-d, d+1e-3, 0.01)
        ue = -0.5 * fx/nu * (ye**2 - d*d)
        ye += d
        from matplotlib import pyplot as plt
        plt.clf()
        plt.plot(ye, ue, label="exact")
        plt.plot(y, u, 'ko', fillstyle='none', label="computed")
        plt.xlabel('y'); plt.ylabel('u')
        plt.legend()
        plt.title('Velocity profile at %s'%tf)
        fig = os.path.join(self.output_dir, "comparison.png")
        plt.savefig(fig, dpi=300)
        return ye, ue, y, u


if __name__ == '__main__':
    app = PoiseuilleFlow()
    app.run()
    app.post_process(app.info_filename)
