"""Abstract class to define the API for an SPH scheme.  The idea is that
one can define a scheme and thereafter one simply instantiates a suitable
scheme, gives it a bunch of particles and runs the application.
"""


class Scheme(object):

    """An API for an SPH scheme.
    """

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
        self.solver = None
        self.attributes_changed()

    # Public protocol ###################################################
    def add_user_options(self, group):
        pass

    def attributes_changed(self):
        """Overload this to compute any properties that depend on others.

        This is automatically called when configure is called.
        """
        pass

    def configure(self, **kw):
        """Configure the scheme with given parameters.

        Overload this to do any scheme specific stuff.
        """
        for k, v in kw.items():
            if not hasattr(self, k):
                msg = 'Parameter {param} not defined for {scheme}.'.format(
                    param=k, scheme=self.__class__.__name__
                )
                raise RuntimeError(msg)
            setattr(self, k, v)
        self.attributes_changed()

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
        return self.solver

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

    # Private protocol ###################################################

    def _ensure_properties(self, pa, desired_props, clean=True):
        """Given a particle array and a set of properties desired,
        this removes unnecessary properties (if `clean=True`), and
        adds the desired properties.

        Parameters
        ----------

        pa : ParticleArray
            Desired particle array.
        desired_props : sequence
            Desired properties to have in the array, can be a list of strings
            or dicts with stride info or both.
        clean : bool
            Remove undesirable properties.
        """
        all_props = {}
        for p in desired_props:
            if isinstance(p, dict):
                all_props.update({p['name']: p})
            elif p not in all_props:
                all_props.update({p: {'name': p}})

        pa_props = set(pa.properties.keys())
        if clean:
            to_remove = pa_props - set(all_props.keys())
            for prop in to_remove:
                pa.remove_property(prop)

        to_add = set(all_props.keys()) - pa_props
        for prop in to_add:
            pa.add_property(**all_props[prop])

    def _smart_getattr(self, obj, var):
        res = getattr(obj, var)
        if res is None:
            return getattr(self, var)
        else:
            return res


class SchemeChooser(Scheme):
    def __init__(self, default, **schemes):
        """
        Parameters
        ----------

        default: str
            Name of the default scheme to use.
        **schemes: kwargs
            The schemes to choose between.
        """
        self.default = default
        self.schemes = dict(schemes)
        self.scheme = schemes[default]

    def add_user_options(self, group):
        for scheme in self.schemes.values():
            scheme.add_user_options(group)
        choices = list(self.schemes.keys())
        group.add_argument(
            "--scheme", action="store", dest="scheme",
            default=self.default, choices=choices,
            help="Specify scheme to use (one of %s)." % choices
        )

    def attributes_changed(self):
        self.scheme.attributes_changed()

    def configure(self, **kw):
        self.scheme.configure(**kw)

    def consume_user_options(self, options):
        self.scheme = self.schemes[options.scheme]
        self.scheme.consume_user_options(options)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        self.scheme.configure_solver(
            kernel=kernel, integrator_cls=integrator_cls,
            extra_steppers=extra_steppers, **kw
        )

    def get_equations(self):
        return self.scheme.get_equations()

    def get_solver(self):
        return self.scheme.get_solver()

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
        self.scheme.setup_properties(particles, clean)


############################################################################

def add_bool_argument(group, arg, dest, help, default):
    group.add_argument(
        '--%s' % arg, action="store_true", dest=dest, help=help
    )
    neg_help = 'Do not ' + help[0].lower() + help[1:]
    group.add_argument(
        '--no-%s' % arg, action="store_false", dest=dest, help=neg_help
    )
    group.set_defaults(**{dest: default})


class WCSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, c0, h0, hdx, gamma=7.0,
                 gx=0.0, gy=0.0, gz=0.0, alpha=0.1, beta=0.0, delta=0.1,
                 nu=0.0, tensile_correction=False, hg_correction=False,
                 update_h=False, delta_sph=False, summation_density=False):
        """Parameters
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
        nu: float
            Real viscosity of the fluid, defaults to no viscosity.
        tensile_correction: bool
            Use tensile correction.
        hg_correction: bool
            Use the Hughes-Graham correction.
        update_h: bool
            Update the smoothing length as per Ferrari et al.
        delta_sph: bool
            Use the delta-SPH correction terms.
        summation_density: bool
            Use summation density instead of continuity.

        References
        ----------

        .. [Hughes2010] J. P. Hughes and D. I. Graham, "Comparison of
           incompressible and weakly-compressible SPH models for free-surface
           water flows", Journal of Hydraulic Research, 48 (2010), pp. 105-117.

        .. [Marrone2011] S. Marrone et al., "delta-SPH model for simulating
           violent impact flows", Computer Methods in Applied Mechanics and
           Engineering, 200 (2011), pp 1526--1542.

        .. [Cherfils2012] J. M. Cherfils et al., "JOSEPHINE: A parallel SPH
           code for free-surface flows", Computer Physics Communications, 183
           (2012), pp 1468--1480.

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
        self.nu = nu
        self.tensile_correction = tensile_correction
        self.hg_correction = hg_correction
        self.update_h = update_h
        self.delta_sph = delta_sph
        self.summation_density = summation_density

    def add_user_options(self, group):
        group.add_argument(
            "--alpha", action="store", type=float, dest="alpha",
            default=None,
            help="Alpha for the artificial viscosity."
        )
        group.add_argument(
            "--beta", action="store", type=float, dest="beta",
            default=None,
            help="Beta for the artificial viscosity."
        )
        group.add_argument(
            "--delta", action="store", type=float, dest="delta",
            default=None,
            help="Delta for the delta-SPH."
        )
        group.add_argument(
            "--gamma", action="store", type=float, dest="gamma",
            default=None,
            help="Gamma for the state equation."
        )
        add_bool_argument(
            group, 'tensile-correction', dest='tensile_correction',
            help="Use tensile instability correction.",
            default=None
        )
        add_bool_argument(
            group, "hg-correction", dest="hg_correction",
            help="Use the Hughes Graham correction.",
            default=None
        )
        add_bool_argument(
            group, "update-h", dest="update_h",
            help="Update the smoothing length as per Ferrari et al.",
            default=None
        )
        add_bool_argument(
            group, "delta-sph", dest="delta_sph",
            help="Use the delta-SPH correction terms.",
            default=None
        )
        add_bool_argument(
            group, "summation-density", dest="summation_density",
            help="Use summation density instead of continuity.",
            default=None
        )

    def consume_user_options(self, options):
        vars = ['gamma', 'tensile_correction', 'hg_correction',
                'update_h', 'delta_sph', 'alpha', 'beta',
                'summation_density']

        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

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
        for name in self.fluids + self.solids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        if 'dt' not in kw:
            kw['dt'] = self.get_timestep()
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.wc.basic import (
            MomentumEquation, TaitEOS, TaitEOSHGCorrection,
            UpdateSmoothingLengthFerrari
        )
        from pysph.sph.wc.basic import (ContinuityEquationDeltaSPH,
                                        ContinuityEquationDeltaSPHPreStep,
                                        MomentumEquationDeltaSPH)
        from pysph.sph.basic_equations import \
            (ContinuityEquation, SummationDensity, XSPHCorrection)
        from pysph.sph.wc.viscosity import (LaminarViscosity,
                                            LaminarViscosityDeltaSPH)
        from pysph.sph.wc.kernel_correction import (GradientCorrectionPreStep,
                                                    GradientCorrection)

        equations = []
        g1 = []
        all = self.fluids + self.solids

        if self.summation_density:
            g0 = []
            for name in self.fluids:
                g0.append(SummationDensity(dest=name, sources=all))
            equations.append(Group(equations=g0, real=False))

        for name in self.fluids:
            g1.append(TaitEOS(
                dest=name, sources=None, rho0=self.rho0, c0=self.c0,
                gamma=self.gamma
            ))

        # This correction applies only to solids.
        for name in self.solids:
            if self.hg_correction:
                g1.append(TaitEOSHGCorrection(
                    dest=name, sources=None, rho0=self.rho0, c0=self.c0,
                    gamma=self.gamma
                ))
            else:
                g1.append(TaitEOS(
                    dest=name, sources=None, rho0=self.rho0, c0=self.c0,
                    gamma=self.gamma
                ))
        equations.append(Group(equations=g1, real=False))

        if self.delta_sph and not self.summation_density:
            eq2_pre = []
            for name in self.fluids:
                eq2_pre.append(
                    GradientCorrectionPreStep(dest=name, sources=[name],
                                              dim=self.dim)
                )
            equations.append(Group(equations=eq2_pre, real=False))

            eq2 = []
            for name in self.fluids:
                eq2.extend([
                    GradientCorrection(dest=name, sources=[name]),
                    ContinuityEquationDeltaSPHPreStep(
                        dest=name, sources=[name]
                    )])
            equations.append(Group(equations=eq2))

        g2 = []
        for name in self.solids:
            g2.append(ContinuityEquation(dest=name, sources=self.fluids))

        for name in self.fluids:
            if not self.summation_density:
                g2.append(
                    ContinuityEquation(dest=name, sources=all)
                )
            if self.delta_sph and not self.summation_density:
                g2.append(
                    ContinuityEquationDeltaSPH(
                        dest=name, sources=[name], c0=self.c0,
                        delta=self.delta
                    ))
            # This is required since MomentumEquation (ME) adds artificial
            # viscosity (AV), so make alpha 0.0 for ME and enable delta sph AV.
            alpha = 0.0 if self.delta_sph else self.alpha
            g2.append(
                MomentumEquation(
                    dest=name, sources=all, c0=self.c0,
                    alpha=alpha, beta=self.beta,
                    gx=self.gx, gy=self.gy, gz=self.gz,
                    tensile_correction=self.tensile_correction
                ))
            if self.delta_sph:
                g2.append(
                    MomentumEquationDeltaSPH(
                        dest=name, sources=[name], rho0=self.rho0, c0=self.c0,
                        alpha=self.alpha
                    ))
            g2.append(XSPHCorrection(dest=name, sources=[name]))

            if abs(self.nu) > 1e-14:
                if self.delta_sph:
                    eq = LaminarViscosityDeltaSPH(
                        dest=name, sources=all, dim=self.dim, rho0=self.rho0,
                        nu=self.nu
                    )
                else:
                    eq = LaminarViscosity(
                        dest=name, sources=all, nu=self.nu
                    )
                g2.insert(-1, eq)
        equations.append(Group(equations=g2))

        if self.update_h:
            g3 = [
                UpdateSmoothingLengthFerrari(
                    dest=x, sources=None, dim=self.dim, hdx=self.hdx
                ) for x in self.fluids
            ]
            equations.append(Group(equations=g3, real=False))

        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_wcsph
        dummy = get_particle_array_wcsph(name='junk')
        props = list(dummy.properties.keys())
        output_props = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                        'pid', 'gid', 'tag', 'p']
        if self.delta_sph:
            delta_sph_props = [
                {'name': 'm_mat', 'stride': 9},
                {'name': 'gradrho', 'stride': 3},
            ]
            props += delta_sph_props
        for pa in particles:
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)


class TVFScheme(Scheme):
    def __init__(self, fluids, solids, dim, rho0, c0, nu, p0, pb, h0,
                 gx=0.0, gy=0.0, gz=0.0, alpha=0.0, tdamp=0.0):
        self.fluids = fluids
        self.solids = solids
        self.solver = None
        self.rho0 = rho0
        self.c0 = c0
        self.pb = pb
        self.p0 = p0
        self.nu = nu
        self.dim = dim
        self.h0 = h0
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.alpha = alpha
        self.tdamp = 0.0

    def add_user_options(self, group):
        group.add_argument(
            "--alpha", action="store", type=float, dest="alpha",
            default=None,
            help="Alpha for the artificial viscosity."
        )
        group.add_argument(
            "--tdamp", action="store", type=float, dest="tdamp",
            default=None,
            help="Time for which the accelerations are damped."
        )

    def consume_user_options(self, options):
        vars = ['alpha', 'tdamp']
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def get_timestep(self, cfl=0.25):
        dt_cfl = cfl * self.h0/self.c0
        if self.nu > 1e-12:
            dt_viscous = 0.125 * self.h0**2/self.nu
        else:
            dt_viscous = 1.0
        dt_force = 1.0

        return min(dt_cfl, dt_viscous, dt_force)

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
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.integrator_step import TransportVelocityStep
        from pysph.sph.integrator import PECIntegrator
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = TransportVelocityStep
        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        cls = integrator_cls if integrator_cls is not None else PECIntegrator
        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.wc.transport_velocity import (
            SummationDensity, StateEquation, MomentumEquationPressureGradient,
            MomentumEquationArtificialViscosity,
            MomentumEquationViscosity, MomentumEquationArtificialStress,
            SolidWallPressureBC, SolidWallNoSlipBC, SetWallVelocity
        )
        equations = []
        all = self.fluids + self.solids
        g1 = []
        for fluid in self.fluids:
            g1.append(SummationDensity(dest=fluid, sources=all))

        equations.append(Group(equations=g1, real=False))

        g2 = []
        for fluid in self.fluids:
            g2.append(StateEquation(
                dest=fluid, sources=None, p0=self.p0, rho0=self.rho0, b=1.0
            ))
        for solid in self.solids:
            g2.append(SetWallVelocity(dest=solid, sources=self.fluids))

        if len(g2) > 0:
            equations.append(Group(equations=g2, real=False))

        g3 = []
        for solid in self.solids:
            g3.append(SolidWallPressureBC(
                dest=solid, sources=self.fluids, b=1.0, rho0=self.rho0,
                p0=self.p0, gx=self.gx, gy=self.gy, gz=self.gz
            ))

        if len(g3) > 0:
            equations.append(Group(equations=g3, real=False))

        g4 = []
        for fluid in self.fluids:
            g4.append(
                MomentumEquationPressureGradient(
                    dest=fluid, sources=all, pb=self.pb, gx=self.gx,
                    gy=self.gy, gz=self.gz, tdamp=self.tdamp
                )
            )
            if self.alpha > 0.0:
                g4.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=all, c0=self.c0,
                        alpha=self.alpha
                    )
                )
            if self.nu > 0.0:
                g4.append(
                    MomentumEquationViscosity(
                        dest=fluid, sources=self.fluids, nu=self.nu
                    )
                )
                if len(self.solids) > 0:
                    g4.append(
                        SolidWallNoSlipBC(
                            dest=fluid, sources=self.solids, nu=self.nu
                        )
                    )

            g4.append(
                MomentumEquationArtificialStress(
                    dest=fluid, sources=self.fluids)
            )

        equations.append(Group(equations=g4))
        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_tvf_fluid, \
            get_particle_array_tvf_solid
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_tvf_fluid(name='junk')
        props = list(dummy.properties.keys())
        output_props = dummy.output_property_arrays
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)

        dummy = get_particle_array_tvf_solid(name='junk')
        props = list(dummy.properties.keys())
        output_props = dummy.output_property_arrays
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)


class AdamiHuAdamsScheme(TVFScheme):
    """This is a scheme similiar to that in the paper:

    Adami, S., Hu, X., Adams, N. A generalized wall boundary condition for
    smoothed particle hydrodynamics.  Journal of Computational Physics
    2012;231(21):7057-7075.

    The major difference is in how the equations are integrated.  The paper
    has a different scheme that does not quite fit in with how things are done
    in PySPH readily so we simply use the WCSPHStep which works well.
    """
    def __init__(self, fluids, solids, dim, rho0, c0, nu, h0,
                 gx=0.0, gy=0.0, gz=0.0, p0=0.0, gamma=7.0,
                 tdamp=0.0, alpha=0.0):
        self.fluids = fluids
        self.solids = solids
        self.solver = None
        self.rho0 = rho0
        self.c0 = c0
        self.h0 = h0
        self.p0 = p0
        self.nu = nu
        self.dim = dim
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.alpha = alpha
        self.gamma = float(gamma)
        self.tdamp = tdamp
        self.attributes_changed()

    def add_user_options(self, group):
        super(AdamiHuAdamsScheme, self).add_user_options(group)
        group.add_argument(
            "--gamma", action="store", type=float, dest="gamma",
            default=None,
            help="Gamma for the state equation."
        )

    def attributes_changed(self):
        self.B = self.c0*self.c0*self.rho0/self.gamma

    def consume_user_options(self, options):
        vars = ['alpha', 'tdamp', 'gamma']
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

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
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.integrator_step import WCSPHStep
        from pysph.sph.integrator import PECIntegrator
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = WCSPHStep
        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        cls = integrator_cls if integrator_cls is not None else PECIntegrator
        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.wc.basic import TaitEOS
        from pysph.sph.basic_equations import XSPHCorrection
        from pysph.sph.wc.transport_velocity import (
            ContinuityEquation, ContinuitySolid,
            MomentumEquationPressureGradient,
            MomentumEquationViscosity, MomentumEquationArtificialViscosity,
            SolidWallPressureBC, SolidWallNoSlipBC, SetWallVelocity,
            VolumeSummation
        )

        equations = []
        all = self.fluids + self.solids

        g2 = []
        for fluid in self.fluids:
            g2.append(VolumeSummation(dest=fluid, sources=all))
            g2.append(TaitEOS(
                dest=fluid, sources=None, rho0=self.rho0, c0=self.c0,
                gamma=self.gamma, p0=self.p0
            ))
        for solid in self.solids:
            g2.append(VolumeSummation(dest=solid, sources=all))
            g2.append(SetWallVelocity(dest=solid, sources=self.fluids))

        equations.append(Group(equations=g2, real=False))

        g3 = []
        for solid in self.solids:
            g3.append(SolidWallPressureBC(
                dest=solid, sources=self.fluids, b=1.0, rho0=self.rho0,
                p0=self.B, gx=self.gx, gy=self.gy, gz=self.gz
            ))

        equations.append(Group(equations=g3, real=False))

        g4 = []
        for fluid in self.fluids:
            g4.append(
                ContinuityEquation(dest=fluid, sources=self.fluids)
            )
            if self.solids:
                g4.append(
                    ContinuitySolid(dest=fluid, sources=self.solids)
                )
            g4.append(
                MomentumEquationPressureGradient(
                    dest=fluid, sources=all, pb=0.0, gx=self.gx,
                    gy=self.gy, gz=self.gz, tdamp=self.tdamp
                )
            )
            if self.alpha > 0.0:
                g4.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=all, c0=self.c0,
                        alpha=self.alpha
                    )
                )
            if self.nu > 0.0:
                g4.append(
                    MomentumEquationViscosity(
                        dest=fluid, sources=self.fluids, nu=self.nu
                    )
                )
                if len(self.solids) > 0:
                    g4.append(
                        SolidWallNoSlipBC(
                            dest=fluid, sources=self.solids, nu=self.nu
                        )
                    )
            g4.append(XSPHCorrection(dest=fluid, sources=[fluid]))

        equations.append(Group(equations=g4))
        return equations

    def setup_properties(self, particles, clean=True):
        super(AdamiHuAdamsScheme, self).setup_properties(particles, clean)
        particle_arrays = dict([(p.name, p) for p in particles])
        props = ['cs', 'arho', 'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                 'ax', 'ay', 'az']
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            for prop in props:
                pa.add_property(prop)


class GasDScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma, kernel_factor, alpha1=1.0,
                 alpha2=0.1, beta=2.0, adaptive_h_scheme='mpm',
                 update_alpha1=False, update_alpha2=False,
                 max_density_iterations=250,
                 density_iteration_tolerance=1e-3, has_ghosts=False):
        """
        Parameters
        ----------

        fluids: list
            List of names of fluid particle arrays.
        solids: list
            List of names of solid particle arrays (or boundaries), currently
            not supported
        dim: int
            Dimensionality of the problem.
        gamma: float
            Gamma for Equation of state.
        kernel_factor: float
            Kernel scaling factor.
        alpha1: float
            Artificial viscosity parameter.
        alpha2: float
            Artificial viscosity parameter.
        beta: float
            Artificial viscosity parameter.
        adaptive_h_scheme: str
            Adaptive h scheme to use. One of ['mpm', 'gsph']
        update_alpha1: bool
            Update the alpha1 parameter dynamically.
        update_alpha2: bool
            Update the alpha2 parameter dynamically.
        max_density_iterations: int
            Maximum number of iterations to run for one density step
        density_iteration_tolerance: float
            Maximum difference allowed in two successive density iterations
        has_ghosts: bool
            if ghost particles (either mirror or periodic) is used
        """
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.update_alpha1 = update_alpha1
        self.update_alpha2 = update_alpha2
        self.beta = beta
        self.kernel_factor = kernel_factor
        self.adaptive_h_scheme = adaptive_h_scheme
        self.density_iteration_tolerance = density_iteration_tolerance
        self.max_density_iterations = max_density_iterations
        self.has_ghosts = has_ghosts

    def add_user_options(self, group):
        choices = ['gsph', 'mpm']
        group.add_argument(
            "--adaptive-h", action="store", dest="adaptive_h_scheme",
            default=None, choices=choices,
            help="Specify scheme for adaptive smoothing lengths %s" % choices
        )
        group.add_argument(
            "--alpha1", action="store", type=float, dest="alpha1",
            default=None,
            help="Alpha1 for the artificial viscosity."
        )
        group.add_argument(
            "--beta", action="store", type=float, dest="beta",
            default=None,
            help="Beta for the artificial viscosity."
        )
        group.add_argument(
            "--alpha2", action="store", type=float, dest="alpha2",
            default=None,
            help="Alpha2 for artificial viscosity"
        )
        group.add_argument(
            "--gamma", action="store", type=float, dest="gamma",
            default=None,
            help="Gamma for the state equation."
        )
        add_bool_argument(
            group, "update-alpha1", dest="update_alpha1",
            help="Update the alpha1 parameter.",
            default=None
        )
        add_bool_argument(
            group, "update-alpha2", dest="update_alpha2",
            help="Update the alpha2 parameter.",
            default=None
        )

    def consume_user_options(self, options):
        vars = ['gamma', 'alpha2', 'alpha1', 'beta', 'update_alpha1',
                'update_alpha2', 'adaptive_h_scheme']
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

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
        from pysph.base.kernels import Gaussian
        if kernel is None:
            kernel = Gaussian(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import PECIntegrator
        from pysph.sph.integrator_step import GasDFluidStep

        cls = integrator_cls if integrator_cls is not None else PECIntegrator
        step_cls = GasDFluidStep
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.gas_dynamics.basic import (
            ScaleSmoothingLength, UpdateSmoothingLengthFromVolume,
            SummationDensity, IdealGasEOS, MPMAccelerations,
            MPMUpdateGhostProps
        )
        from pysph.sph.gas_dynamics.boundary_equations import WallBoundary

        equations = []
        # Find the optimal 'h'
        if self.adaptive_h_scheme == 'mpm':
            g1 = []
            for fluid in self.fluids:
                g1.append(
                    SummationDensity(
                        dest=fluid, sources=self.fluids, k=self.kernel_factor,
                        density_iterations=True, dim=self.dim,
                        htol=self.density_iteration_tolerance
                    )
                )

            equations.append(Group(
                equations=g1, update_nnps=True, iterate=True,
                max_iterations=self.max_density_iterations
            ))

        elif self.adaptive_h_scheme == 'gsph':
            group = []
            for fluid in self.fluids:
                group.append(
                    ScaleSmoothingLength(dest=fluid, sources=None, factor=2.0)
                )
            equations.append(Group(equations=group, update_nnps=True))

            group = []
            for fluid in self.fluids:
                group.append(
                    SummationDensity(
                        dest=fluid, sources=self.fluids, dim=self.dim
                    )
                )
            equations.append(Group(equations=group, update_nnps=False))

            group = []
            for fluid in self.fluids:
                group.append(
                    UpdateSmoothingLengthFromVolume(
                        dest=fluid, sources=None, k=self.kernel_factor,
                        dim=self.dim
                    )
                )
            equations.append(Group(equations=group, update_nnps=True))

            group = []
            for fluid in self.fluids:
                group.append(
                    SummationDensity(
                        dest=fluid, sources=self.fluids, dim=self.dim
                    )
                )
            equations.append(Group(equations=group, update_nnps=False))
        # Done with finding the optimal 'h'

        g2 = []
        for fluid in self.fluids:
            g2.append(IdealGasEOS(dest=fluid, sources=None, gamma=self.gamma))

        equations.append(Group(equations=g2))

        g3 = []
        for solid in self.solids:
            g3.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=g3))

        if self.has_ghosts:
            gh = []
            for fluid in self.fluids:
                gh.append(
                    MPMUpdateGhostProps(dest=fluid, sources=None)
                )
            equations.append(Group(equations=gh, real=False))

        g4 = []
        for fluid in self.fluids:
            g4.append(MPMAccelerations(
                dest=fluid, sources=self.fluids + self.solids,
                alpha1_min=self.alpha1,
                alpha2_min=self.alpha2, beta=self.beta,
                update_alpha1=self.update_alpha1,
                update_alpha2=self.update_alpha2
            ))
        equations.append(Group(equations=g4))

        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_gasd
        import numpy
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_gasd(name='junk')
        props = list(dummy.properties.keys())
        output_props = dummy.output_property_arrays
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.add_property('orig_idx', type='int')
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.set_output_arrays(output_props)

        solid_props = set(props) | set('div cs wij htmp'.split(' '))
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, solid_props, clean)
            pa.set_output_arrays(output_props)


class GSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma, kernel_factor, g1=0.0,
                 g2=0.0, rsolver=2, interpolation=1, monotonicity=1,
                 interface_zero=True, hybrid=False, blend_alpha=5.0, tf=1.0,
                 niter=20, tol=1e-6, has_ghosts=False):
        """
        Parameters
        ----------

        fluids: list
            List of names of fluid particle arrays.
        solids: list
            List of names of solid particle arrays (or boundaries), currently
            not supported
        dim: int
            Dimensionality of the problem.
        gamma: float
            Gamma for Equation of state.
        kernel_factor: float
            Kernel scaling factor.
        g1, g2 : double
            ADKE style thermal conduction parameters
        rsolver: int
            Riemann solver to use.  See pysph.sph.gas_dynamics.gsph for
            valid options.
        interpolation: int
            Kind of interpolation for the specific volume integrals.
        monotonicity : int
            Type of monotonicity algorithm to use:
            0 : First order GSPH
            1 : I02 algorithm
            2 : IwIn algorithm
        interface_zero : bool
            Set Interface position s^*_{ij} = 0 for the Riemann problem.
        hybrid, blend_alpha : bool, double
            Hybrid scheme and blending alpha value
        tf: double
            Final time used for blending.
        niter: int
            Max number of iterations for iterative Riemann solvers.
        tol: double
            Tolerance for iterative Riemann solvers.
        has_ghosts: bool
            if ghost particles (either mirror or periodic) is used
        """
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.kernel_factor = kernel_factor
        self.g1 = g1
        self.g2 = g2
        self.rsolver = rsolver
        self.interpolation = interpolation
        self.monotonicity = monotonicity
        self.interface_zero = interface_zero
        self.hybrid = hybrid
        self.blend_alpha = blend_alpha
        self.tf = tf
        self.niter = niter
        self.tol = tol
        self.has_ghosts = has_ghosts

    def add_user_options(self, group):
        group.add_argument(
            "--rsolver", action="store", type=int, dest="rsolver",
            default=None,
            help="Riemann solver to use."
        )
        group.add_argument(
            "--interpolation", action="store", type=int, dest="interpolation",
            default=None,
            help="Interpolation algorithm to use."
        )
        group.add_argument(
            "--monotonicity", action="store", type=int, dest="monotonicity",
            default=None,
            help="Monotonicity algorithm to use."
        )
        group.add_argument(
            "--g1", action="store", type=float, dest="g1",
            default=None,
            help="ADKE style thermal conduction parameter."
        )
        group.add_argument(
            "--g2", action="store", type=float, dest="g2",
            default=None,
            help="ADKE style thermal conduction parameter."
        )
        group.add_argument(
            "--gamma", action="store", type=float, dest="gamma",
            default=None,
            help="Gamma for the state equation."
        )
        group.add_argument(
            "--blend-alpha", action="store", type=float, dest="blend_alpha",
            default=None,
            help="Blending factor for hybrid scheme."
        )
        add_bool_argument(
            group, "interface-zero", dest="interface_zero",
            help="Set interface position to zero for Riemann problem.",
            default=None
        )
        add_bool_argument(
            group, "hybrid", dest="hybrid",
            help="Use the hybrid scheme.",
            default=None
        )

    def consume_user_options(self, options):
        vars = ['gamma', 'g1', 'g2', 'rsolver', 'interpolation',
                'monotonicity', 'interface_zero', 'hybrid',
                'blend_alpha']
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

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
        from pysph.base.kernels import Gaussian
        if kernel is None:
            kernel = Gaussian(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import EulerIntegrator
        from pysph.sph.integrator_step import GSPHStep

        cls = integrator_cls if integrator_cls is not None else EulerIntegrator
        step_cls = GSPHStep
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )
        if 'tf' in kw:
            self.tf = kw['tf']

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.gas_dynamics.basic import (
            ScaleSmoothingLength, UpdateSmoothingLengthFromVolume,
            SummationDensity, IdealGasEOS
        )
        from pysph.sph.gas_dynamics.boundary_equations import WallBoundary
        from pysph.sph.gas_dynamics.gsph import (
            GSPHGradients, GSPHAcceleration, GSPHUpdateGhostProps
        )
        equations = []
        # Find the optimal 'h'
        group = []
        for fluid in self.fluids:
            group.append(
                ScaleSmoothingLength(dest=fluid, sources=None, factor=2.0)
            )
        equations.append(Group(equations=group, update_nnps=True))

        group = []
        for solid in self.solids:
            group.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=group))

        all_pa = self.fluids + self.solids
        group = []
        for fluid in self.fluids:
            group.append(
                SummationDensity(
                    dest=fluid, sources=all_pa, dim=self.dim
                )
            )
        equations.append(Group(equations=group, update_nnps=False))

        group = []
        for solid in self.solids:
            group.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=group))

        group = []
        for fluid in self.fluids:
            group.append(
                UpdateSmoothingLengthFromVolume(
                    dest=fluid, sources=None, k=self.kernel_factor,
                    dim=self.dim
                )
            )
        equations.append(Group(equations=group, update_nnps=True))

        group = []
        for fluid in self.fluids:
            group.append(
                SummationDensity(
                    dest=fluid, sources=all_pa, dim=self.dim
                )
            )
        equations.append(Group(equations=group, update_nnps=False))
        # Done with finding the optimal 'h'

        group = []
        for fluid in self.fluids:
            group.append(IdealGasEOS(dest=fluid, sources=None,
                                     gamma=self.gamma))
        equations.append(Group(equations=group))

        group = []
        for solid in self.solids:
            group.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=group))

        g2 = []
        for fluid in self.fluids:
            g2.append(GSPHGradients(dest=fluid, sources=all_pa))

        equations.append(Group(equations=g2))

        if self.has_ghosts:
            g3 = []
            for fluid in self.fluids:
                g3.append(GSPHUpdateGhostProps(dest=fluid, sources=None))
            equations.append(Group(
                equations=g3, update_nnps=False, real=False
                ))

        g4 = []
        for fluid in self.fluids:
            g4.append(GSPHAcceleration(
                dest=fluid, sources=all_pa, g1=self.g1,
                g2=self.g2, monotonicity=self.monotonicity,
                rsolver=self.rsolver, interpolation=self.interpolation,
                interface_zero=self.interface_zero,
                hybrid=self.hybrid, blend_alpha=self.blend_alpha,
                gamma=self.gamma, niter=self.niter, tol=self.tol
            ))
        equations.append(Group(equations=g4))
        return equations

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array_gasd
        import numpy
        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_gasd(name='junk')
        props = (list(dummy.properties.keys()) +
                 'px py pz ux uy uz vx vy vz wx wy wz'.split())

        output_props = dummy.output_property_arrays
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.add_property('orig_idx', type='int')
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.set_output_arrays(output_props)

        solid_props = set(props) | set(('wij', 'htmp'))
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, solid_props, clean)
            pa.set_output_arrays(output_props)


class ADKEScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma=1.4, alpha=1.0, beta=2.0,
                 k=1.0, eps=0.0, g1=0, g2=0, has_ghosts=False):
        """
        Parameters
        ----------

        fluids: list
            a list with names of fluid particle arrays
        solids: list
            a list with names of solid (or boundary) particle arrays
        dim: int
            dimensionality of the problem
        gamma: double
            Gamma for equation of state
        alpha: double
            artificial viscosity parameter
        beta: double
            artificial viscosity parameter
        k: double
            kernel scaling parameter
        eps: double
            kernel scaling parameter
        g1: double
            artificial heat conduction parameter
        g2: double
            artificial heat conduction parameter
        has_ghosts: bool
            if problem uses ghost particles (periodic or mirror)
        """
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.eps = eps
        self.g1 = g1
        self.g2 = g2
        self.has_ghosts = has_ghosts

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import SummationDensity
        from pysph.sph.gas_dynamics.basic import (
            IdealGasEOS, ADKEAccelerations, SummationDensityADKE,
            ADKEUpdateGhostProps
        )
        from pysph.sph.gas_dynamics.boundary_equations import WallBoundary

        equations = []
        g1 = []
        for solid in self.solids:
            g1.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=g1))

        g2 = []
        for fluid in self.fluids:
            g2.append(
                SummationDensityADKE(
                    fluid, sources=self.fluids + self.solids, k=self.k,
                    eps=self.eps
                )
            )
        equations.append(Group(g2, update_nnps=True, iterate=False))

        g3 = []
        for solid in self.solids:
            g3.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=g3))

        g4 = []
        for fluid in self.fluids:
            g4.append(SummationDensity(fluid, self.fluids+self.solids))
        equations.append(Group(g4))

        g5 = []
        for solid in self.solids:
            g5.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=g5))

        g6 = []
        for elem in self.fluids+self.solids:
            g6.append(IdealGasEOS(elem, sources=None, gamma=self.gamma))
        equations.append(Group(equations=g6))

        if self.has_ghosts:
            gh = []
            for fluid in self.fluids:
                gh.append(
                    ADKEUpdateGhostProps(dest=fluid, sources=None)
                )
            equations.append(Group(equations=gh, real=False))

        g7 = []
        for fluid in self.fluids:
            g7.append(
                ADKEAccelerations(
                    dest=fluid, sources=self.fluids + self.solids,
                    alpha=self.alpha, beta=self.beta, g1=self.g1, g2=self.g2,
                    k=self.k, eps=self.eps
                )
            )

        equations.append(Group(equations=g7))
        return equations

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
        from pysph.base.kernels import Gaussian
        if kernel is None:
            kernel = Gaussian(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import PECIntegrator
        from pysph.sph.integrator_step import ADKEStep

        cls = integrator_cls if integrator_cls is not None else PECIntegrator
        step_cls = ADKEStep
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array
        import numpy
        particle_arrays = dict([(p.name, p) for p in particles])
        required_props = [
                'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'cs', 'p',
                'e', 'au', 'av', 'aw', 'arho', 'ae', 'am', 'ah', 'x0', 'y0',
                'z0', 'u0', 'v0', 'w0', 'rho0', 'e0', 'h0', 'div',  'h0',
                'wij', 'htmp', 'logrho']

        dummy = get_particle_array(additional_props=required_props,
                                   name='junk')
        dummy.set_output_arrays(
            ['x', 'y', 'u', 'v', 'rho', 'm', 'h',
             'cs', 'p', 'e', 'au', 'av', 'ae', 'pid', 'gid', 'tag']
        )

        props = list(dummy.properties.keys())
        output_props = dummy.output_property_arrays
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)

        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.add_property('orig_idx', type='int')
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.set_output_arrays(output_props)
