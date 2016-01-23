"""Abstract class to define the API for an SPH scheme.  The idea is that
one can define a scheme and thereafter one simply instantiates a suitable
scheme, gives it a bunch of particles and runs the application.
"""

class Scheme(object):

    """An API for an SPH scheme.
    """

    #### Public protocol ###################################################
    def __init__(self, fluids, solids, dim):
        """
        Parameters
        ----------

        fluids: list
            List of names of fluid particle arrays.
        solids: list
            List of names of solid particle arrays (or boundaries).
        dim: int
            Dimensionality of the problem.
        """
        self.fluids = fluids
        self.solids = solids
        self.dim = dim

    @classmethod
    def add_user_options(klass, group, **defaults):
        pass

    def consume_user_options(self, options):
        pass

    def configure_solver(self, kernel=None, integrator_cls=None,
                           extra_steppers=None, **kw):
        """Configure the solver to be generated.

        Parameters
        ----------

        kernel : Kernel instance.
            Kernel to use, if none is passed a default one is used.
        integrator_cls : pysph.sph.integrator.Integrator
            Integrator class to use, use sensible default if none is
            passed.
        extra_steppers : dict
            Additional integration stepper instances as a dict.
        **kw : extra arguments
            Any additional keyword args are passed to the solver instance.
        """
        raise NotImplementedError()

    def get_equations(self):
        raise NotImplementedError()

    def get_solver(self):
        raise NotImplementedError()

    def setup_properties(self, particles, clean=True):
        """Setup the particle arrays so they have the right set of properties
        for this scheme.

        Parameters
        ----------

        particles : list
            List of particle arrays.

        clean : bool
            If True, removes any unnecessary properties.
        """
        raise NotImplementedError()

    #### Private protocol ###################################################

    def _ensure_properties(self, pa, desired_props, clean=True):
        """Given a particle array and a set of properties desired,
        this removes unnecessary properties (if `clean=True`), and
        adds the desired properties.

        Parameters
        ----------

        pa : ParticleArray
            Desired particle array.
        desired_props : sequence
            Desired properties to have in the array.
        clean : bool
            Remove undesirable properties.
        """
        all_props = set(desired_props)
        if clean:
            to_remove = set(pa.properties.keys()) - all_props
            for prop in to_remove:
                pa.remove_property(prop)

        to_add = all_props - set(pa.properties.keys())
        for prop in to_add:
            pa.add_property(prop)


############################################################################

def add_bool_argument(group, arg, dest, help, default):
    group.add_argument(
        '--%s'%arg, action="store_true", dest=dest, help=help
    )
    neg_help = 'Do not ' + help[0].lower() + help[1:]
    group.add_argument(
        '--no-%s'%arg, action="store_false", dest=dest, help=neg_help
    )
    group.set_defaults(**{dest: default})


class WCSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, c0, h0, hdx, gamma=7.0,
                  gx=0.0, gy=0.0, gz=0.0, alpha=0.1, beta=0.0, delta=0.1,
                  tensile_correction=False, hg_correction=False,
                  update_h=False, delta_sph=False):
        """
        Parameters
        ----------

        fluids: list
            List of names of fluid particle arrays.
        solids: list
            List of names of solid particle arrays (or boundaries).
        dim: int
            Dimensionality of the problem.
        rho0: float
            Reference density.
        c0: float
            Reference speed of sound.
        gamma: float
            Gamma for the equation of state.
        h0: float
            Reference smoothing length.
        hdx: float
            Ratio of h/dx.
        gx, gy, gz: float
            Body force acceleration components.
        alpha: float
            Coefficient for artificial viscosity.
        beta: float
            Coefficient for artificial viscosity.
        delta: float
            Coefficient used to control the intensity of diffusion of density
        tensile_correction: bool
            Use tensile correction.
        hg_correction: bool
            Use the Hughes-Graham correction.
        update_h: bool
            Update the smoothing length as per Ferrari et al.
        delta_sph: bool
            Use the delta-SPH correction terms.

        References
        ----------

        .. [Hughes2010] J. P. Hughes and D. I. Graham, "Comparison of incompressible
           and weakly-compressible SPH models for free-surface water flows",
           Journal of Hydraulic Research, 48 (2010), pp. 105-117.

        .. [Marrone2011] S. Marrone et al., "delta-SPH model for simulating
           violent impact flows", Computer Methods in Applied Mechanics and
           Engineering, 200 (2011), pp 1526--1542.

        .. [Cherfils2012] J. M. Cherfils et al., "JOSEPHINE: A parallel SPH code for
           free-surface flows", Computer Physics Communications, 183 (2012),
           pp 1468--1480.

        """
        self.fluids = fluids
        self.solids = solids
        self.solver = None
        self.rho0 = rho0
        self.c0 = c0
        self.gamma = gamma
        self.dim = dim
        self.h0 = h0
        self.hdx = hdx
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.tensile_correction = tensile_correction
        self.hg_correction = hg_correction
        self.update_h = update_h
        self.delta_sph = delta_sph

    @classmethod
    def add_user_options(klass, group, **defaults):
        group.add_argument(
            "--alpha", action="store", type=float, dest="alpha",
            default=defaults.get('alpha', 0.1),
            help="Alpha for the artificial viscosity."
        )
        group.add_argument(
            "--beta", action="store", type=float, dest="beta",
            default=defaults.get('beta', 0.0),
            help="Beta for the artificial viscosity."
        )
        group.add_argument(
            "--delta", action="store", type=float, dest="delta",
            default=defaults.get('delta', 0.1),
            help="Delta for the delta-SPH."
        )
        group.add_argument(
            "--gamma", action="store", type=float, dest="gamma",
            default=defaults.get('gamma', 7.0),
            help="Gamma for the state equation."
        )
        add_bool_argument(
            group, 'tensile-correction', dest='tensile_correction',
            help="Use tensile instability correction.",
            default=defaults.get('tensile_correction', False)
        )
        add_bool_argument(
            group, "hg-correction", dest="hg_correction",
            help="Use the Hughes Graham correction.",
            default=defaults.get('hg_correction', False)
        )
        add_bool_argument(
            group, "update-h", dest="update_h",
            help="Update the smoothing length as per Ferrari et al.",
            default=defaults.get('update_h', False)
        )
        add_bool_argument(
            group, "delta-sph", dest="delta_sph",
            help="Use the delta-SPH correction terms.",
            default=defaults.get('delta_sph', False)
        )

    def consume_user_options(self, options):
        vars = ['gamma', 'tensile_correction', 'hg_correction',
                'update_h', 'delta_sph', 'alpha', 'beta']
        for var in vars:
            setattr(self, var, getattr(options, var))

    def get_timestep(self, cfl=0.5):
        return cfl*self.h0/self.c0

    def configure_solver(self, kernel=None, integrator_cls=None,
                           extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import PECIntegrator, TVDRK3Integrator
        from pysph.sph.integrator_step import WCSPHStep, WCSPHTVDRK3Step

        cls = integrator_cls if integrator_cls is not None else PECIntegrator
        step_cls = WCSPHTVDRK3Step if cls is TVDRK3Integrator else WCSPHStep
        for fluid in self.fluids:
            steppers[fluid] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        if 'dt' not in kw:
            kw['dt'] = self.get_timestep()
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.wc.basic import (MomentumEquation, TaitEOS,
             TaitEOSHGCorrection, UpdateSmoothingLengthFerrari)
        from pysph.sph.wc.basic import (ContinuityEquationDeltaSPH,
                                        MomentumEquationDeltaSPH)
        from pysph.sph.basic_equations import (ContinuityEquation,
             XSPHCorrection)
        equations = []
        g1 = []
        all = self.fluids + self.solids
        for name in self.fluids:
            g1.append(TaitEOS(
                dest=name, sources=None, rho0=self.rho0, c0=self.c0,
                gamma=self.gamma
            ))

        if self.hg_correction:
            # This correction applies only to solids.
            for name in self.solids:
                g1.append(TaitEOSHGCorrection(
                    dest=name, sources=None, rho0=self.rho0, c0=self.c0,
                    gamma=self.gamma
                ))

        equations.append(Group(equations=g1, real=False))

        g2 = []
        for name in self.solids:
            g2.append(ContinuityEquation(dest=name, sources=self.fluids))

        for name in self.fluids:
            if self.delta_sph:
                other = all[:]
                other.remove(name)
                g2.extend([
                    ContinuityEquationDeltaSPH(dest=name, sources=[name]),
                    ContinuityEquation(dest=name, sources=other),
                    MomentumEquationDeltaSPH(
                        dest=name, sources=[name], rho0=self.rho0, c0=self.c0,
                        alpha=self.alpha, gx=self.gx, gy=self.gy, gz=self.gz,
                    ),
                    MomentumEquation(
                        dest=name, sources=other, c0=self.c0,
                        alpha=self.alpha, beta=self.beta,
                        gx=self.gx, gy=self.gy, gz=self.gz,
                        tensile_correction=self.tensile_correction
                    ),
                    XSPHCorrection(dest=name, sources=[name])
                ])
            else:
                g2.extend([
                    ContinuityEquation(dest=name, sources=all),
                    MomentumEquation(
                        dest=name, sources=all, alpha=self.alpha, beta=self.beta,
                        gx=self.gx, gy=self.gy, gz=self.gz, c0=self.c0,
                        tensile_correction=self.tensile_correction
                    ),
                    XSPHCorrection(dest=name, sources=[name])
                ])
        equations.append(Group(equations=g2))

        if self.update_h:
            g3 = [
                UpdateSmoothingLengthFerrari(
                    dest=x, sources=None, dim=self.dim, hdx=self.hdx
                ) for x in self.fluids
            ]
            equations.append(Group(equations=g3, real=False))

        return equations

    def get_solver(self):
        return self.solver

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_wcsph, DEFAULT_PROPS
        dummy = get_particle_array_wcsph(name='junk')
        props = list(dummy.properties.keys())
        output_props = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                        'pid', 'gid', 'tag', 'p']
        for pa in particles:
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
