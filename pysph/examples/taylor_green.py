"""Taylor Green vortex flow (5 minutes).
"""

import os
import numpy as np
from numpy import pi, sin, cos, exp

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.application import Application

from pysph.sph.equation import Group, Equation
from pysph.sph.scheme import TVFScheme, WCSPHScheme, SchemeChooser
from pysph.sph.wc.edac import ComputeAveragePressure, EDACScheme
from pysph.sph.iisph import IISPHScheme

from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep, GradientCorrection,
    MixedKernelCorrectionPreStep, MixedGradientCorrection
)
from pysph.sph.wc.crksph import CRKSPHPreStep, CRKSPH, CRKSPHScheme
from pysph.sph.wc.gtvf import GTVFScheme
from pysph.sph.wc.pcisph import PCISPHScheme
from pysph.sph.wc.shift import ShiftPositions
from pysph.sph.isph.sisph import SISPHScheme


# domain and constants
L = 1.0
U = 1.0
rho0 = 1.0
c0 = 10 * U
p0 = c0**2 * rho0


def m4p(x=0.0):
    """From the paper by Chaniotis et al. (JCP 2002).
    """
    if x < 0.0:
        return 0.0
    elif x < 1.0:
        return 1.0 - 0.5*x*x*(5.0 - 3.0*x)
    elif x < 2.0:
        return (1 - x)*(2 - x)*(2 - x)*0.5
    else:
        return 0.0


class M4(Equation):
    '''An equation to be used for remeshing.
    '''
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def _get_helpers_(self):
        return [m4p]

    def loop(self, s_idx, d_idx, s_temp_prop, d_prop, d_h, XIJ):
        xij = abs(XIJ[0]/d_h[d_idx])
        yij = abs(XIJ[1]/d_h[d_idx])
        d_prop[d_idx] += m4p(xij)*m4p(yij)*s_temp_prop[s_idx]


def exact_solution(U, b, t, x, y):
    factor = U * exp(b*t)

    u = -cos(2*pi*x) * sin(2*pi*y)
    v = sin(2*pi*x) * cos(2*pi*y)
    p = -0.25 * (cos(4*pi*x) + cos(4*pi*y))

    return factor * u, factor * v, factor * factor * p


class TaylorGreen(Application):

    def add_user_options(self, group):
        group.add_argument(
            "--init", action="store", type=str, default=None,
            help="Initialize particle positions from given file."
        )
        group.add_argument(
            "--perturb", action="store", type=float, dest="perturb", default=0,
            help="Random perturbation of initial particles as a fraction "
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
        corrections = ['', 'mixed', 'gradient', 'crksph']
        group.add_argument(
            "--kernel-correction", action="store", type=str,
            dest='kernel_correction',
            default='', help="Type of Kernel Correction", choices=corrections
        )
        group.add_argument(
            "--remesh", action="store", type=int, dest="remesh", default=0,
            help="Remeshing frequency (setting it to zero disables it)."
        )
        remesh_types = ['m4', 'sph']
        group.add_argument(
            "--remesh-eq", action="store", type=str, dest="remesh_eq",
            default='m4', choices=remesh_types,
            help="Remeshing strategy to use."
        )
        group.add_argument(
            "--shift-freq", action="store", type=int, dest="shift_freq",
            default=0,
            help="Particle position shift frequency.(set zero to disable)."
        )
        shift_types = ['simple', 'fickian']
        group.add_argument(
            "--shift-kind", action="store", type=str, dest="shift_kind",
            default='simple', choices=shift_types,
            help="Use of fickian shift in positions."
        )
        group.add_argument(
            "--shift-parameter", action="store", type=float,
            dest="shift_parameter", default=None,
            help="Constant used in shift, range for 'simple' is 0.01-0.1"
            "range 'fickian' is 1-10."
        )
        group.add_argument(
            "--shift-correct-vel", action="store_true",
            dest="correct_vel", default=False,
            help="Correct velocities after shifting (defaults to false)."
        )

    def consume_user_options(self):
        nx = self.options.nx
        re = self.options.re

        self.nu = nu = U * L / re

        self.dx = dx = L / nx
        self.volume = dx * dx
        self.hdx = self.options.hdx

        h0 = self.hdx * self.dx
        if self.options.scheme == 'iisph' or self.options.scheme == 'pcisph':
            dt_cfl = 0.25 * h0 / U
        if self.options.scheme == 'sisph':
            dt_cfl = 0.25 * h0 / U
        else:
            dt_cfl = 0.25 * h0 / (c0 + U)
        dt_viscous = 0.125 * h0**2 / nu
        dt_force = 0.25 * 1.0

        self.dt = min(dt_cfl, dt_viscous, dt_force)
        self.tf = 5.0
        self.kernel_correction = self.options.kernel_correction

    def configure_scheme(self):
        scheme = self.scheme
        h0 = self.hdx * self.dx
        pfreq = 100
        kernel = QuinticSpline(dim=2)
        if self.options.scheme == 'tvf':
            scheme.configure(pb=self.options.pb_factor * p0, nu=self.nu, h0=h0)
        elif self.options.scheme == 'wcsph':
            scheme.configure(hdx=self.hdx, nu=self.nu, h0=h0)
        elif self.options.scheme == 'edac':
            scheme.configure(h=h0, nu=self.nu, pb=self.options.pb_factor * p0)
        elif self.options.scheme == 'iisph' or self.options.scheme == 'pcisph':
            scheme.configure(nu=self.nu)
            pfreq = 10
        elif self.options.scheme == 'crksph':
            scheme.configure(h0=h0, nu=self.nu)
        elif self.options.scheme == 'gtvf':
            scheme.configure(pref=p0, nu=self.nu, h0=h0)
        elif self.options.scheme == 'sisph':
            pfreq = 10
            scheme.configure(nu=self.nu)
        scheme.configure_solver(kernel=kernel, tf=self.tf, dt=self.dt,
                                pfreq=pfreq)

    def create_scheme(self):
        h0 = None
        hdx = None
        wcsph = WCSPHScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, h0=h0,
            hdx=hdx, nu=None, gamma=7.0, alpha=0.0, beta=0.0
        )
        tvf = TVFScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, nu=None,
            p0=p0, pb=None, h0=h0
        )
        edac = EDACScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, nu=None,
            pb=p0, h=h0
        )
        iisph = IISPHScheme(
            fluids=['fluid'], solids=[], dim=2, nu=None,
            rho0=rho0, has_ghosts=True
        )
        crksph = CRKSPHScheme(
            fluids=['fluid'], dim=2, nu=None,
            rho0=rho0, h0=h0, c0=c0, p0=0.0
        )
        gtvf = GTVFScheme(
            fluids=['fluid'], solids=[], dim=2, rho0=rho0, c0=c0,
            nu=None, h0=None, pref=None
        )
        pcisph = PCISPHScheme(
            fluids=['fluid'], dim=2, rho0=rho0, nu=None
        )
        sisph = SISPHScheme(
            fluids=['fluid'], solids=[], dim=2, nu=None, rho0=rho0,
            c0=c0, alpha=0.0, has_ghosts=True, pref=p0,
            rho_cutoff=0.2, internal_flow=True, gtvf=True
        )
        s = SchemeChooser(
            default='tvf', wcsph=wcsph, tvf=tvf, edac=edac, iisph=iisph,
            crksph=crksph, gtvf=gtvf, pcisph=pcisph, sisph=sisph
        )
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()
        # This tolerance needs to be fixed.
        tol = 0.5
        if self.kernel_correction == 'gradient':
            cls1 = GradientCorrectionPreStep
            cls2 = GradientCorrection
        elif self.kernel_correction == 'mixed':
            cls1 = MixedKernelCorrectionPreStep
            cls2 = MixedGradientCorrection
        elif self.kernel_correction == 'crksph':
            cls1 = CRKSPHPreStep
            cls2 = CRKSPH

        if self.kernel_correction:
            g1 = Group(equations=[cls1('fluid', ['fluid'], dim=2)])
            eq2 = cls2(dest='fluid', sources=['fluid'], dim=2, tol=tol)

            if self.options.scheme == 'wcsph':
                eqns.insert(1, g1)
                eqns[2].equations.insert(0, eq2)
            elif self.options.scheme == 'tvf':
                eqns[1].equations.append(g1.equations[0])
                eqns[2].equations.insert(0, eq2)
            elif self.options.scheme == 'edac':
                eqns.insert(1, g1)
                eqns[2].equations.insert(0, eq2)

        return eqns

    def create_domain(self):
        return DomainManager(
            xmin=0, xmax=L, ymin=0, ymax=L, periodic_in_x=True,
            periodic_in_y=True
        )

    def create_particles(self):
        # create the particles
        dx = self.dx
        _x = np.arange(dx / 2, L, dx)
        x, y = np.meshgrid(_x, _x)
        if self.options.init is not None:
            fname = self.options.init
            from pysph.solver.utils import load
            data = load(fname)
            _f = data['arrays']['fluid']
            x, y = _f.x.copy(), _f.y.copy()

        if self.options.perturb > 0:
            np.random.seed(1)
            factor = dx * self.options.perturb
            x += np.random.random(x.shape) * factor
            y += np.random.random(x.shape) * factor

        # Initialize
        m = self.volume * rho0
        h = self.hdx * dx
        re = self.options.re
        b = -8.0*pi*pi / re
        u0, v0, p0 = exact_solution(U=U, b=b, t=0, x=x, y=y)
        color0 = cos(2*pi*x) * cos(4*pi*y)

        # create the arrays
        fluid = get_particle_array(name='fluid', x=x, y=y, m=m, h=h, u=u0,
                                   v=v0, rho=rho0, p=p0, color=color0)

        self.scheme.setup_properties([fluid])

        print("Taylor green vortex problem :: nfluid = %d, dt = %g" % (
            fluid.get_number_of_particles(), self.dt))

        # volume is set as dx^2
        if self.options.scheme == 'sisph':
            nfp = fluid.get_number_of_particles()
            fluid.gid[:] = np.arange(nfp)
            fluid.add_output_arrays(['gid'])
        if self.options.scheme == 'tvf':
            fluid.V[:] = 1. / self.volume
        if self.options.scheme == 'iisph':
            # These are needed to update the ghost particle properties.
            nfp = fluid.get_number_of_particles()
            fluid.orig_idx[:] = np.arange(nfp)
            fluid.add_output_arrays(['orig_idx'])

        corr = self.kernel_correction
        if corr in ['mixed', 'crksph']:
            fluid.add_property('cwij')
        if corr == 'mixed' or corr == 'gradient':
            fluid.add_property('m_mat', stride=9)
            fluid.add_property('dw_gamma', stride=3)
        elif corr == 'crksph':
            fluid.add_property('ai')
            fluid.add_property('gradbi', stride=4)
            for prop in ['gradai', 'bi']:
                fluid.add_property(prop, stride=2)

        if self.options.shift_freq > 0:
            fluid.add_constant('vmax', [0.0])
            fluid.add_property('dpos', stride=3)
            fluid.add_property('gradv', stride=9)

        return [fluid]

    def create_tools(self):
        tools = []
        options = self.options
        if options.remesh > 0:
            if options.remesh_eq == 'm4':
                equations = [M4(dest='interpolate', sources=['fluid'])]
            else:
                equations = None
            from pysph.solver.tools import SimpleRemesher
            if options.scheme == 'wcsph' or options.scheme == 'crksph':
                props = ['u', 'v', 'au', 'av', 'ax', 'ay', 'arho']
            elif options.scheme == 'pcisph':
                props = ['u', 'v', 'p']
            elif options.scheme == 'tvf':
                props = ['u', 'v', 'uhat', 'vhat',
                         'au', 'av', 'auhat', 'avhat']
            elif options.scheme == 'edac':
                if 'uhat' in self.particles[0].properties:
                    props = ['u', 'v', 'uhat', 'vhat', 'p',
                             'au', 'av', 'auhat', 'avhat', 'ap']
                else:
                    props = ['u', 'v', 'p', 'au', 'av', 'ax', 'ay', 'ap']
            elif options.scheme == 'iisph':
                # The accelerations are not really needed since the current
                # stepper is a single stage stepper.
                props = ['u', 'v', 'p']
            elif options.scheme == 'gtvf':
                props = [
                    'uhat', 'vhat', 'what', 'rho0', 'rhodiv', 'p0',
                    'auhat', 'avhat', 'awhat', 'arho', 'arho0'
                ]

            remesher = SimpleRemesher(
                self, 'fluid', props=props,
                freq=self.options.remesh, equations=equations
            )
            tools.append(remesher)

        if options.shift_freq > 0:
            shift = ShiftPositions(
                self, 'fluid', freq=self.options.shift_freq,
                shift_kind=self.options.shift_kind,
                correct_velocity=self.options.correct_vel,
                parameter=self.options.shift_parameter
            )
            tools.append(shift)

        return tools

    # The following are all related to post-processing.
    def _get_post_process_props(self, array):
        """Return x, y, m, u, v, p.
        """
        if 'pavg' not in array.properties or \
           'pavg' not in array.output_property_arrays:
            self._add_extra_props(array)
            sph_eval = self._get_sph_evaluator(array)
            sph_eval.update_particle_arrays([array])
            sph_eval.evaluate()

        x, y, m, u, v, p, pavg = array.get(
            'x', 'y', 'm', 'u', 'v', 'p', 'pavg'
        )
        return x, y, m, u, v, p - pavg

    def _add_extra_props(self, array):
        extra = ['pavg', 'nnbr']
        for prop in extra:
            if prop not in array.properties:
                array.add_property(prop)
        array.add_output_arrays(extra)

    def _get_sph_evaluator(self, array):
        if not hasattr(self, '_sph_eval'):
            from pysph.tools.sph_evaluator import SPHEvaluator
            equations = [
                ComputeAveragePressure(dest='fluid', sources=['fluid'])
            ]
            dm = self.create_domain()
            sph_eval = SPHEvaluator(
                arrays=[array], equations=equations, dim=2,
                kernel=QuinticSpline(dim=2), domain_manager=dm
            )
            self._sph_eval = sph_eval
        return self._sph_eval

    def post_process(self, info_fname):
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        decay_rate = -8.0 * np.pi**2 / self.options.re

        files = self.output_files
        t, ke, ke_ex, decay, linf, l1, p_l1 = [], [], [], [], [], [], []
        for sd, array in iter_output(files, 'fluid'):
            _t = sd['t']
            t.append(_t)
            x, y, m, u, v, p = self._get_post_process_props(array)
            u_e, v_e, p_e = exact_solution(U, decay_rate, _t, x, y)
            vmag2 = u**2 + v**2
            vmag = np.sqrt(vmag2)
            ke.append(0.5 * np.sum(m * vmag2))
            vmag2_e = u_e**2 + v_e**2
            vmag_e = np.sqrt(vmag2_e)
            ke_ex.append(0.5 * np.sum(m * vmag2_e))

            vmag_max = vmag.max()
            decay.append(vmag_max)
            theoretical_max = U * np.exp(decay_rate * _t)
            linf.append(abs((vmag_max - theoretical_max) / theoretical_max))

            l1_err = np.average(np.abs(vmag - vmag_e))
            avg_vmag_e = np.average(np.abs(vmag_e))
            # scale the error by the maximum velocity.
            l1.append(l1_err / avg_vmag_e)

            p_e_max = np.abs(p_e).max()
            p_error = np.average(np.abs(p - p_e)) / p_e_max
            p_l1.append(p_error)

        t, ke, ke_ex, decay, l1, linf, p_l1 = list(map(
            np.asarray, (t, ke, ke_ex, decay, l1, linf, p_l1))
        )
        decay_ex = U * np.exp(decay_rate * t)
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(
            fname, t=t, ke=ke, ke_ex=ke_ex, decay=decay, linf=linf, l1=l1,
            p_l1=p_l1, decay_ex=decay_ex
        )

        import matplotlib
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.clf()
        plt.semilogy(t, decay_ex, label="exact")
        plt.semilogy(t, decay, label="computed")
        plt.xlabel('t')
        plt.ylabel('max velocity')
        plt.legend()
        fig = os.path.join(self.output_dir, "decay.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, linf)
        plt.xlabel('t')
        plt.ylabel(r'$L_\infty$ error')
        fig = os.path.join(self.output_dir, "linf_error.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, l1, label="error")
        plt.xlabel('t')
        plt.ylabel(r'$L_1$ error')
        fig = os.path.join(self.output_dir, "l1_error.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, p_l1, label="error")
        plt.xlabel('t')
        plt.ylabel(r'$L_1$ error for $p$')
        fig = os.path.join(self.output_dir, "p_l1_error.png")
        plt.savefig(fig, dpi=300)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')


if __name__ == '__main__':
    app = TaylorGreen()
    app.run()
    app.post_process(app.info_filename)
