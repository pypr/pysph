"""Lid driven cavity using the Transport Velocity formulation. (10 minutes)
"""

import os

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application

from pysph.sph.scheme import TVFScheme

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
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=50,
            help="Number of points along x direction."
        )
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=100,
            help="Reynolds number (defaults to 100)."
        )
        self.n_avg = 5
        group.add_argument(
            "--n-vel-avg", action="store", type=int, dest="n_avg",
            default=None,
            help="Average velocities over these many saved timesteps."
        )

    def consume_user_options(self):
        nx = self.options.nx
        if self.options.n_avg is not None:
            self.n_avg = self.options.n_avg
        self.dx = L/nx
        self.re = self.options.re
        h0 = hdx * self.dx
        self.nu = Umax*L/self.re
        dt_cfl = 0.25 * h0/( c0 + Umax )
        dt_viscous = 0.125 * h0**2/self.nu
        dt_force = 1.0
        self.tf = 10.0
        self.dt = min(dt_cfl, dt_viscous, dt_force)

    def configure_scheme(self):
        h0 = hdx * self.dx
        self.scheme.configure(h0=h0, nu=self.nu)
        self.scheme.configure_solver(tf=self.tf, dt=self.dt, pfreq=500)

    def create_scheme(self):
        s = TVFScheme(
            ['fluid'], ['solid'], dim=2, rho0=rho0, c0=c0, nu=None,
            p0=p0, pb=p0, h0=hdx
        )
        return s

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
        self.scheme.setup_properties([fluid, solid])

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * rho0
        solid.m[:] = volume * rho0
        # Set a reference rho also, some schemes will overwrite this with a
        # summation density.
        fluid.rho[:] = rho0
        solid.rho[:] = rho0

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

        return [fluid, solid]

    def post_process(self, info_fname):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        if self.rank > 0:
            return
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return
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
        ui = np.zeros_like(xx)
        vi = np.zeros_like(xx)
        # Average out the velocities over the last n_avg timesteps
        for fname in self.output_files[-self.n_avg:]:
            data = load(fname)
            tf = data['solver_data']['t']
            interp.update_particle_arrays(list(data['arrays'].values()))
            _u = interp.interpolate('u')
            _v = interp.interpolate('v')
            _u.shape = 101,101
            _v.shape = 101,101
            ui += _u
            vi += _v

        ui /= self.n_avg
        vi /= self.n_avg

        # velocity magnitude
        self.vmag = vmag = np.sqrt( ui**2 + vi**2 )
        import matplotlib.pyplot as plt

        f = plt.figure()

        plt.streamplot(
            xx, yy, ui, vi, density=(2, 2), #linewidth=5*vmag/vmag.max(),
            color=vmag
        )
        plt.xlim(0, 1); plt.ylim(0, 1)
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
