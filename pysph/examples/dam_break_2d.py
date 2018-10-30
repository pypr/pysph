"""Two-dimensional dam break over a dry bed.  (30 minutes)

The case is described in "State of the art classical SPH for free surface
flows", Moncho Gomez-Gesteira, Benedict D Rogers, Robert A, Dalrymple and Alex
J.C Crespo, Journal of Hydraulic Research, Vol 48, Extra Issue (2010), pp
6-27. DOI:10.1080/00221686.2010.9641242

"""

import os
import numpy as np

from pysph.base.kernels import WendlandQuintic, QuinticSpline
from pysph.examples._db_geometry import DamBreak2DGeometry
from pysph.solver.application import Application
from pysph.sph.scheme import WCSPHScheme, SchemeChooser, AdamiHuAdamsScheme
from pysph.sph.wc.edac import EDACScheme
from pysph.sph.iisph import IISPHScheme
from pysph.sph.equation import Group
from pysph.sph.wc.kernel_correction import (GradientCorrectionPreStep,
                                            GradientCorrection,
                                            MixedKernelCorrectionPreStep)
from pysph.sph.wc.crksph import CRKSPHPreStep, CRKSPH


fluid_column_height = 2.0
fluid_column_width = 1.0
container_height = 4.0
container_width = 4.0
nboundary_layers = 2
nu = 0.0
h = 0.039
g = 9.81
ro = 1000.0
co = 10.0 * np.sqrt(2 * 9.81 * fluid_column_height)
gamma = 7.0
alpha = 0.1
beta = 0.0
B = co * co * ro / gamma
p0 = 1000.0


class DamBreak2D(Application):

    def add_user_options(self, group):
        self.hdx = 1.3
        corrections = ['', 'mixed-corr', 'grad-corr', 'kernel-corr', 'crksph']
        group.add_argument(
            "--h-factor", action="store", type=float, dest="h_factor",
            default=1.0,
            help="Divide default h by this factor to change resolution"
        )
        group.add_argument(
            "--kernel-corr", action="store", type=str, dest='kernel_corr',
            default='', help="Type of Kernel Correction", choices=corrections
        )

    def consume_user_options(self):
        self.h = h / self.options.h_factor
        self.kernel_corr = self.options.kernel_corr
        print("Using h = %f" % self.h)
        if self.options.scheme == 'wcsph':
            self.hdx = self.hdx
        else:
            self.hdx = 1.0
        self.dx = self.h / self.hdx

    def configure_scheme(self):
        tf = 2.5
        kw = dict(
            tf=tf, output_at_times=[0.4, 0.6, 0.8, 1.0]
        )
        if self.options.scheme == 'wcsph':
            self.scheme.configure(h0=self.h, hdx=self.hdx)
            kernel = WendlandQuintic(dim=2)
            from pysph.sph.integrator import EPECIntegrator
            kw.update(
                dict(
                    integrator_cls=EPECIntegrator,
                    kernel=kernel, adaptive_timestep=True, n_damp=50,
                    fixed_h=False
                )
            )
        elif self.options.scheme == 'aha':
            self.scheme.configure(h0=self.h)
            kernel = QuinticSpline(dim=2)
            dt = 0.125 * self.h / co
            kw.update(
                dict(
                    kernel=kernel, dt=dt
                )
            )
            print("dt = %f" % dt)
        elif self.options.scheme == 'edac':
            self.scheme.configure(h=self.h)
            kernel = QuinticSpline(dim=2)
            dt = 0.125 * self.h / co
            kw.update(
                dict(
                    kernel=kernel, dt=dt
                )
            )
            print("dt = %f" % dt)
        elif self.options.scheme == 'iisph':
            kernel = QuinticSpline(dim=2)
            dt = 0.125 * 10 * self.h / co
            kw.update(
                dict(
                    kernel=kernel, dt=dt, adaptive_timestep=True
                )
            )
            print("dt = %f" % dt)

        self.scheme.configure_solver(**kw)

    def create_scheme(self):
        wcsph = WCSPHScheme(
            ['fluid'], ['boundary'], dim=2, rho0=ro, c0=co,
            h0=h, hdx=1.3, gy=-9.81, alpha=alpha, beta=beta,
            gamma=gamma, hg_correction=True, update_h=True
        )
        aha = AdamiHuAdamsScheme(
            fluids=['fluid'], solids=['boundary'], dim=2, c0=co, nu=nu,
            rho0=ro, h0=h, p0=0.0, gy=-g, gamma=1.0, tdamp=0.0, alpha=alpha
        )
        edac = EDACScheme(
            fluids=['fluid'], solids=['boundary'], dim=2, c0=co, nu=nu,
            rho0=ro, h=h, pb=0.0, gy=-g, eps=0.0, clamp_p=True
        )
        iisph = IISPHScheme(
            fluids=['fluid'], solids=['boundary'], dim=2, nu=nu,
            rho0=ro, gy=-g
        )
        s = SchemeChooser(default='wcsph', wcsph=wcsph, aha=aha, edac=edac,
                          iisph=iisph)
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()
        if self.options.scheme == 'iisph':
            return eqns
        n = len(eqns)
        if self.kernel_corr == 'grad-corr':
            eqn1 = Group(equations=[
                GradientCorrectionPreStep('fluid', ['fluid', 'boundary'])
            ], real=False)
            for i in range(n):
                eqn2 = GradientCorrection('fluid', ['fluid', 'boundary'])
                eqns[i].equations.insert(0, eqn2)
            eqns.insert(0, eqn1)
        elif self.kernel_corr == 'mixed-corr':
            eqn1 = Group(equations=[
                MixedKernelCorrectionPreStep('fluid', ['fluid', 'boundary'])
            ], real=False)
            for i in range(n):
                eqn2 = GradientCorrection('fluid', ['fluid', 'boundary'])
                eqns[i].equations.insert(0, eqn2)
            eqns.insert(0, eqn1)
        elif self.kernel_corr == 'crksph':
            eqn1 = Group(equations=[
                CRKSPHPreStep('fluid', ['fluid', 'boundary']),
                CRKSPHPreStep('boundary', ['fluid', 'boundary'])
            ], real=False)
            for i in range(n):
                eqn2 = CRKSPH('fluid', ['fluid', 'boundary'])
                eqn3 = CRKSPH('boundary', ['fluid', 'boundary'])
                eqns[i].equations.insert(0, eqn3)
                eqns[i].equations.insert(0, eqn2)
            eqns.insert(0, eqn1)
        return eqns

    def create_particles(self):
        if self.options.scheme == 'wcsph':
            nboundary_layers = 2
            nfluid_offset = 2
            wall_hex_pack = True
        else:
            nboundary_layers = 4
            nfluid_offset = 1
            wall_hex_pack = False
        geom = DamBreak2DGeometry(
            container_width=container_width, container_height=container_height,
            fluid_column_height=fluid_column_height,
            fluid_column_width=fluid_column_width, dx=self.dx, dy=self.dx,
            nboundary_layers=1, ro=ro, co=co,
            with_obstacle=False, wall_hex_pack=wall_hex_pack,
            beta=1.0, nfluid_offset=1, hdx=self.hdx)
        fluid, boundary = geom.create_particles(
            nboundary_layers=nboundary_layers, hdx=self.hdx,
            nfluid_offset=nfluid_offset,
        )
        self.scheme.setup_properties([fluid, boundary])
        if self.options.scheme == 'iisph':
            # the default position tends to cause the particles to be pushed
            # away from the wall, so displacing it by a tiny amount helps.
            fluid.x += self.dx / 4

        # Adding extra properties for kernel correction
        corr = self.kernel_corr
        if corr == 'kernel-corr' or corr == 'mixed-corr':
            fluid.add_property('cwij')
            boundary.add_property('cwij')
        if corr == 'mixed-corr' or corr == 'grad-corr':
            fluid.add_property('m_mat', stride=9)
            boundary.add_property('m_mat', stride=9)
        elif corr == 'crksph':
            fluid.add_property('ai')
            boundary.add_property('ai')
            fluid.add_property('gradbi', stride=9)
            boundary.add_property('gradbi', stride=9)
            for prop in ['gradai', 'bi']:
                fluid.add_property(prop, stride=3)
                boundary.add_property(prop, stride=3)

        return [fluid, boundary]

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        t, x_max = [], []
        factor = np.sqrt(2.0 * 9.81 / fluid_column_width)
        files = self.output_files
        for sd, array in iter_output(files, 'fluid'):
            t.append(sd['t'] * factor)
            x = array.get('x')
            x_max.append(x.max())

        t, x_max = list(map(np.asarray, (t, x_max)))
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, t=t, x_max=x_max)

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from pysph.examples import db_exp_data as dbd
        plt.plot(t, x_max, label='Computed')
        te, xe = dbd.get_koshizuka_oka_data()
        plt.plot(te, xe, 'o', label='Koshizuka & Oka (1996)')
        plt.xlim(0, 0.7 * factor)
        plt.ylim(0, 4.5)
        plt.xlabel('$T$')
        plt.ylabel('$Z/L$')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(self.output_dir, 'x_vs_t.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    app = DamBreak2D()
    app.run()
    app.post_process(app.info_filename)
