"""Taylor Green vortex flow (5 minutes).
"""

import numpy as np
import os

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.application import Application

from pysph.sph.scheme import TVFScheme, WCSPHScheme, SchemeChooser


# domain and constants
L = 1.0; U = 1.0
rho0 = 1.0; c0 = 10 * U
p0 = c0**2 * rho0


def exact_velocity(U, b, t, x, y):
    pi = np.pi; sin = np.sin; cos = np.cos
    factor = U * np.exp(b*t)

    u = -cos( 2 * pi * x ) * sin( 2 * pi * y)
    v = sin( 2 * pi * x ) * cos( 2 * pi * y)

    return factor * u, factor * v


class TaylorGreen(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--init", action="store", type=str, default=None,
            help="Initialize particle positions from given file."
        )
        group.add_argument(
            "--perturb", action="store", type=float, dest="perturb", default=0,
            help="Random perturbation of initial particles as a fraction "\
                "of dx (setting it to zero disables it, the default)."
        )
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=50,
            help="Number of points along x direction. (default 50)"
        )
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=100,
            help="Reynolds number (defaults to 100)."
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.0,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--pb-factor", action="store", type=float, dest="pb_factor",
            default=1.0,
            help="Use fraction of the background pressure (default: 1.0)."
        )

    def consume_user_options(self):
        nx = self.options.nx
        re = self.options.re

        self.nu = nu = U*L/re

        self.dx = dx = L/nx
        self.volume = dx*dx
        self.hdx = self.options.hdx
        self.gamma = self.options.gamma

        h0 = self.hdx * self.dx
        dt_cfl = 0.25 * h0/( c0 + U )
        dt_viscous = 0.125 * h0**2/nu
        dt_force = 0.25 * 1.0

        self.tf = 5.0
        self.dt = min(dt_cfl, dt_viscous, dt_force)

        scheme = self.scheme
        scheme.update(nu=self.nu, h0=h0)
        if self.options.scheme == 'tvf':
            scheme.update(pb=self.options.pb_factor*p0)
        else:
            scheme.update(hdx=self.hdx)
        kernel = QuinticSpline(dim=2)
        scheme.configure_solver(kernel=kernel, tf=self.tf, dt=self.dt)

    def create_scheme(self):
        h0 = None
        hdx = None
        wcsph = WCSPHScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, h0=h0,
            hdx=hdx, nu=None, gamma=7.0, alpha=0.1, beta=0.0
        )
        tvf = TVFScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, nu=None,
            p0=p0, pb=None, h0=h0
        )
        s = SchemeChooser(default='tvf', wcsph=wcsph, tvf=tvf)
        return s

    def create_domain(self):
        return DomainManager(
            xmin=0, xmax=L, ymin=0, ymax=L, periodic_in_x=True,
            periodic_in_y=True
        )

    def create_particles(self):
        # create the particles
        dx = self.dx
        _x = np.arange( dx/2, L, dx )
        x, y = np.meshgrid(_x, _x); x = x.ravel(); y = y.ravel()
        if self.options.init is not None:
            fname = self.options.init
            from pysph.solver.utils import load
            data = load(fname)
            _f = data['arrays']['fluid']
            x, y = _f.x.copy(), _f.y.copy()

        if self.options.perturb > 0:
            np.random.seed(1)
            factor = dx*self.options.perturb
            x += np.random.random(x.shape)*factor
            y += np.random.random(x.shape)*factor
        h = np.ones_like(x) * dx

        # create the arrays

        fluid = get_particle_array(name='fluid', x=x, y=y, h=h)

        self.scheme.setup_properties([fluid])

        # add the requisite arrays
        fluid.add_property('color')
        fluid.add_output_arrays(['color'])

        print("Taylor green vortex problem :: nfluid = %d, dt = %g"%(
            fluid.get_number_of_particles(), self.dt))

        # setup the particle properties
        pi = np.pi; cos = np.cos; sin=np.sin

        # color
        fluid.color[:] = cos(2*pi*x) * cos(4*pi*y)

        # velocities
        fluid.u[:] = -U * cos(2*pi*x) * sin(2*pi*y)
        fluid.v[:] = +U * sin(2*pi*x) * cos(2*pi*y)
        fluid.p[:] = -U*U*(np.cos(4*np.pi*x) + np.cos(4*np.pi*y))*0.25

        # mass is set to get the reference density of each phase
        fluid.rho[:] = rho0
        fluid.m[:] = self.volume * fluid.rho

        # volume is set as dx^2
        if self.options.scheme == 'tvf':
            fluid.V[:] = 1./self.volume

        # smoothing lengths
        fluid.h[:] = self.hdx * dx

        # return the particle list
        return [fluid]

    def post_process(self, info_fname):
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        decay_rate = -8.0 * np.pi**2/self.options.re

        files = self.output_files
        t, ke, ke_ex, decay, linf, l1 = [], [], [], [], [], []
        for sd, array in iter_output(files, 'fluid'):
            _t = sd['t']
            t.append(_t)
            m, u, v, x, y = array.get('m', 'u', 'v', 'x', 'y')
            u_e, v_e = exact_velocity(U, decay_rate, _t, x, y)
            vmag2 = u**2 + v**2
            vmag = np.sqrt(vmag2)
            ke.append(0.5*np.sum(m*vmag2))
            vmag2_e = u_e**2 + v_e**2
            vmag_e = np.sqrt(vmag2_e)
            ke_ex.append(0.5*np.sum(m*vmag2_e))

            vmag_max = vmag.max()
            decay.append(vmag_max)
            theoretical_max = U * np.exp(decay_rate * _t)
            linf.append(abs( (vmag_max - theoretical_max)/theoretical_max ))

            l1_err = np.average(np.abs(vmag - vmag_e))
            avg_vmag_e = np.average(np.abs(vmag_e))
            # scale the error by the maximum velocity.
            l1.append(l1_err/avg_vmag_e)

        t, ke, ke_ex, decay, l1, linf = list(map(
            np.asarray, (t, ke, ke_ex, decay, l1, linf))
        )
        decay_ex = U*np.exp(decay_rate*t)
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, t=t, ke=ke, ke_ex=ke_ex, decay=decay, linf=linf, l1=l1,
                 decay_ex=decay_ex)

        import matplotlib
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.clf()
        plt.semilogy(t, decay_ex, label="exact")
        plt.semilogy(t, decay, label="computed")
        plt.xlabel('t'); plt.ylabel('max velocity')
        plt.legend()
        fig = os.path.join(self.output_dir, "decay.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, linf)
        plt.xlabel('t'); plt.ylabel(r'$L_\infty$ error')
        fig = os.path.join(self.output_dir, "linf_error.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, l1, label="error")
        plt.xlabel('t'); plt.ylabel(r'$L_1$ error')
        fig = os.path.join(self.output_dir, "l1_error.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = TaylorGreen()
    app.run()
    app.post_process(app.info_filename)
