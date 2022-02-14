"""
References
-----------

    .. [Hopkins2013] Hopkins, Philip F. “A General Class of Lagrangian
        Smoothed Particle Hydrodynamics Methods and Implications for Fluid
        Mixing Problems.” Monthly Notices of the Royal Astronomical Society
        428, no. 4 (February 1, 2013): 2840–56.
        https://doi.org/10.1093/mnras/sts210.

    .. [Hopkins2015] Hopkins, Philip F. “A New Class of Accurate,
        Mesh-Free Hydrodynamic Simulation Methods.” Monthly Notices of the
        Royal Astronomical Society 450, no. 1 (June 11, 2015): 53–110.
        https://doi.org/10.1093/mnras/stv195.
"""

from pysph.sph.scheme import Scheme


class TSPHScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma, hfact, beta=2.0, fkern=1.0,
                 max_density_iterations=250, alphaav=1.0,
                 density_iteration_tolerance=1e-3, has_ghosts=False):
        """
        Density-energy formulation [Hopkins2013]_ including Balsara's
        artificial viscocity switch with modifications,
        as presented in Appendix F1 of [Hopkins2015]_ .

        Notes
        -----
        Is this exactly in accordance with what is proposed in [Hopkins2015]_ ?
            Not quite.

        What is different then?
            #. Adapting smoothing length using MPM [KP14]_ procedure from
               :class:`SummationDensity
               <pysph.sph.gas_dynamics.basic.SummationDensity>`. In this,
               calculation of grad-h terms are changed to that specified for
               this scheme.
            #. Using the PEC integrator step. No individual
               adaptive time-stepping.
            #. Using :class:`Gaussian Kernel <pysph.base.kernels.Gaussian>`
               by default instead of Cubic Spline with radius scale 1.

        Tip: Reduce the number of points if particle penetration is
        encountered. This has to be done while running
        ``gas_dynamics.wc_blastwave`` and ``gas_dynamics.robert``.

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
            :math:`\\gamma` for Equation of state.
        hfact: float
            :math:`h_{fact}` for smoothing length adaptivity, also referred to
            as kernel_factor in other gas dynamics schemes.
        beta : float, optional
            :math:`\\beta` for artificial viscosity, by default 2.0
        fkern : float, optional
            :math:`f_{kern}`, Factor to scale smoothing length for equivalence
            with classic kernel when using kernel with altered
            `radius_scale` is being used, by default 1.
        max_density_iterations : int, optional
            Maximum number of iterations to run for one density step,
            by default 250.
        density_iteration_tolerance : float, optional
            Maximum difference allowed in two successive density iterations,
            by default 1e-3
        has_ghosts : bool, optional
            if ghost particles (either mirror or periodic) is used, by default
            False
        alphaav : float, optional
            :math:`\\alpha_{av}` for artificial viscosity switch, by default
            1.0
        """

        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.beta = beta
        self.hfact = hfact
        self.density_iteration_tolerance = density_iteration_tolerance
        self.max_density_iterations = max_density_iterations
        self.has_ghosts = has_ghosts
        self.fkern = fkern
        self.alphaav = alphaav

    def add_user_options(self, group):
        group.add_argument("--alphaav", action="store", type=float,
                           dest="alphaav", default=None,
                           help="alpha_av for the artificial viscosity "
                                "switch.")

        group.add_argument("--beta", action="store", type=float, dest="beta",
                           default=None,
                           help="Beta for the artificial viscosity.")

        group.add_argument("--gamma", action="store", type=float, dest="gamma",
                           default=None, help="Gamma for the state equation.")

    def consume_user_options(self, options):
        vars = ['gamma', 'alphaav', 'beta']
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):

        from pysph.base.kernels import Gaussian
        if kernel is None:
            kernel = Gaussian(dim=self.dim)

        if hasattr(kernel, 'fkern'):
            self.fkern = kernel.fkern
        else:
            self.fkern = 1.0

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import PECIntegrator
        from pysph.sph.gas_dynamics.tsph.pec_step import PECStep

        cls = integrator_cls if integrator_cls is not None else PECIntegrator
        step_cls = PECStep
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.gas_dynamics.tsph.equations import (
            SummationDensity, MomentumAndEnergy, VelocityGradDivC1,
            BalsaraSwitch, UpdateGhostProps, WallBoundary, IdealGasEOS)

        all_pa = self.fluids + self.solids
        equations = []

        # Find the optimal 'h'
        g1 = []
        for fluid in self.fluids:
            g1.append(SummationDensity(
                dest=fluid, sources=all_pa, hfact=self.hfact,
                density_iterations=True, dim=self.dim,
                htol=self.density_iteration_tolerance))
            equations.append(Group(
                equations=g1, update_nnps=True, iterate=True,
                max_iterations=self.max_density_iterations))

        g2 = []
        for fluid in self.fluids:
            g2.append(IdealGasEOS(dest=fluid, sources=None, gamma=self.gamma))
        equations.append(Group(equations=g2))

        g3 = []
        for fluid in self.fluids:
            g3.append(VelocityGradDivC1(dest=fluid, sources=all_pa,
                                        dim=self.dim))
            g3.append(BalsaraSwitch(dest=fluid, sources=None,
                                    alphaav=self.alphaav, fkern=self.fkern))
        equations.append(Group(equations=g3))

        g4 = []
        for solid in self.solids:
            g4.append(WallBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=g4))

        if self.has_ghosts:
            gh = []
            for fluid in self.fluids:
                gh.append(UpdateGhostProps(dest=fluid, sources=None))
            equations.append(Group(equations=gh, real=False))

        g5 = []
        for fluid in self.fluids:
            g5.append(MomentumAndEnergy(dest=fluid, sources=all_pa,
                                        dim=self.dim, beta=self.beta,
                                        fkern=self.fkern))
        equations.append(Group(equations=g5))

        return equations

    def setup_properties(self, particles, clean=True):
        import numpy
        particle_arrays = dict([(p.name, p) for p in particles])

        props = ['rho', 'm', 'x', 'y', 'z', 'u', 'v', 'w', 'h', 'cs', 'p', 'e',
                 'au', 'av', 'aw', 'ae', 'pid', 'gid', 'tag', 'dwdh', 'h0',
                 'converged', 'ah', 'arho', 'dt_cfl', 'e0', 'rho0', 'u0', 'v0',
                 'w0', 'x0', 'y0', 'z0', 'alpha']
        more_props = ['drhosumdh', 'n', 'dndh', 'prevn', 'prevdndh',
                      'prevdrhosumdh', 'divv', 'an', 'n0']
        props.extend(more_props)
        output_props = 'rho p u v w x y z e n divv h alpha'.split(' ')
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.add_property('orig_idx', type='int')
            # Guess for number density.
            pa.add_property('n', data=pa.rho / pa.m)
            pa.add_property('gradv', stride=9)
            pa.add_property('invtt', stride=9)
            nfp = pa.get_number_of_particles()
            pa.orig_idx[:] = numpy.arange(nfp)
            pa.set_output_arrays(output_props)

        solid_props = set(props) | set('m0 wij htmp'.split(' '))
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, solid_props, clean)
            pa.set_output_arrays(output_props)
