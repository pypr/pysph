
class Tool(object):
    """A tool is typically an object that can be used to perform a
    specific task on the solver's pre_step/post_step or post_stage callbacks.
    This can be used for a variety of things.  For example, one could save a
    plot, print debug statistics or perform remeshing etc.

    To create a new tool, simply subclass this class and overload any of its
    desired methods.
    """

    def pre_step(self, solver):
        """If overloaded, this is called automatically before each integrator
        step.  The method is passed the solver instance.
        """
        pass

    def post_stage(self, current_time, dt, stage):
        """If overloaded, this is called automatically after each integrator
        stage, i.e. if the integrator is a two stage integrator it will be
        called after the first and second stages.

        The method is passed (current_time, dt, stage).  See the the
        `Integrator.one_timestep` methods for examples of how this is called.
        """
        pass

    def post_step(self, solver):
        """If overloaded, this is called automatically after each integrator
        step.  The method is passed the solver instance.
        """
        pass


class SimpleRemesher(Tool):
    """A simple tool to periodically remesh a given array of particles onto an
    initial set of points.
    """

    def __init__(self, app, array_name, props, freq=100, xi=None, yi=None,
                 zi=None, kernel=None, equations=None):
        """Constructor.

        Parameters
        ----------

        app : pysph.solver.application.Application
            The application instance.
        array_name: str
            Name of the particle array that needs to be remeshed.
        props : list(str)
            List of properties to interpolate.
        freq : int
            Frequency of remeshing operation.
        xi, yi, zi : ndarray
            Positions to remesh the properties onto.  If not specified they
            are taken from the particle arrays at the time of construction.
        kernel: any kernel from pysph.base.kernels

        equations: list or None
            Equations to use for the interpolation, passed to the interpolator.

        """
        from pysph.solver.utils import get_array_by_name
        self.app = app
        self.particles = app.particles
        self.array = get_array_by_name(self.particles, array_name)
        self.props = props
        if xi is None:
            xi = self.array.x
        if yi is None:
            yi = self.array.y
        if zi is None:
            zi = self.array.z
        self.xi, self.yi, self.zi = xi.copy(), yi.copy(), zi.copy()
        self.freq = freq
        from pysph.tools.interpolator import Interpolator
        if kernel is None:
            kernel = app.solver.kernel
        self.interp = Interpolator(
            self.particles, x=self.xi, y=self.yi, z=self.zi,
            kernel=kernel,
            domain_manager=app.create_domain(),
            equations=equations
        )

    def post_step(self, solver):
        if solver.count % self.freq == 0 and solver.count > 0:
            self.interp.nnps.update()
            data = dict(x=self.xi, y=self.yi, z=self.zi)
            for prop in self.props:
                data[prop] = self.interp.interpolate(prop)
            self.array.set(**data)
            self.interp.nnps.update_domain()


class FiberIntegrator(Tool):
    def __init__(self, all_particles, scheme, domain=None, innerloop=True,
                 updates=True, parallel=False, steps=None, D=0):
        """The second integrator is a simple Euler-Integrator (accurate
        enough due to very small time steps; very fast) using EBGSteps.
        EBGSteps are basically the same as EulerSteps, exept for the fact
        that they work with an intermediate ebg velocity [eu, ev, ew].
        This velocity does not interfere with the actual velocity, which
        is neseccery to not disturb the real velocity through artificial
        damping in this step. The ebg velocity is initialized for each
        inner loop again and reset in the outer loop."""
        from math import ceil
        from pysph.base.kernels import CubicSpline
        from pysph.sph.integrator_step import EBGStep
        from compyle.config import get_config
        from pysph.sph.integrator import EulerIntegrator
        from pysph.sph.scheme import BeadChainScheme
        from pysph.sph.equation import Group
        from pysph.sph.fiber.utils import (HoldPoints, Contact,
                                           ComputeDistance)
        from pysph.sph.fiber.beadchain import (Tension, Bending,
                                               ArtificialDamping)
        from pysph.base.nnps import DomainManager, LinkedListNNPS
        from pysph.sph.acceleration_eval import AccelerationEval
        from pysph.sph.sph_compiler import SPHCompiler

        if not isinstance(scheme, BeadChainScheme):
            raise TypeError("Scheme must be BeadChainScheme")

        self.innerloop = innerloop
        self.dt = scheme.dt
        self.fiber_dt = scheme.fiber_dt
        self.domain_updates = updates
        self.steps = steps
        self.D = D
        self.eta0 = scheme.rho0 * scheme.nu

        # if there are more than 1 particles involved, elastic equations are
        # iterated in an inner loop.
        if self.innerloop:
            # second integrator
            # self.fiber_integrator = EulerIntegrator(fiber=EBGStep())
            steppers = {}
            for f in scheme.fibers:
                steppers[f] = EBGStep()
            self.fiber_integrator = EulerIntegrator(**steppers)
            # The type of spline has no influence here. It must be large enough
            # to contain the next particle though.
            kernel = CubicSpline(dim=scheme.dim)
            equations = []
            g1 = []
            for fiber in scheme.fibers:
                g1.append(ComputeDistance(dest=fiber, sources=[fiber]))
            equations.append(Group(equations=g1))

            g2 = []
            for fiber in scheme.fibers:
                g2.append(Tension(dest=fiber,
                                  sources=None,
                                  ea=scheme.E*scheme.A))
                g2.append(Bending(dest=fiber,
                                  sources=None,
                                  ei=scheme.E*scheme.Ip))
                g2.append(Contact(dest=fiber,
                                  sources=scheme.fibers,
                                  E=scheme.E,
                                  d=scheme.dx,
                                  dim=scheme.dim,
                                  k=scheme.k,
                                  lim=scheme.lim,
                                  eta0=self.eta0))
                g2.append(ArtificialDamping(dest=fiber,
                                            sources=None,
                                            d=self.D))
            equations.append(Group(equations=g2))

            g3 = []
            for fiber in scheme.fibers:
                g3.append(HoldPoints(dest=fiber, sources=None, tag=100))
            equations.append(Group(equations=g3))

            # These equations are applied to fiber particles only - that's the
            # reason for computational speed up.
            particles = [p for p in all_particles if p.name in scheme.fibers]
            # A seperate DomainManager is needed to ensure that particles don't
            # leave the domain.
            if domain:
                xmin = domain.manager.xmin
                ymin = domain.manager.ymin
                zmin = domain.manager.zmin
                xmax = domain.manager.xmax
                ymax = domain.manager.ymax
                zmax = domain.manager.zmax
                periodic_in_x = domain.manager.periodic_in_x
                periodic_in_y = domain.manager.periodic_in_y
                periodic_in_z = domain.manager.periodic_in_z
                gamma_yx = domain.manager.gamma_yx
                gamma_zx = domain.manager.gamma_zx
                gamma_zy = domain.manager.gamma_zy
                n_layers = domain.manager.n_layers
                N = self.steps or int(ceil(self.dt/self.fiber_dt))
                # dt = self.dt/N
                self.domain = DomainManager(xmin=xmin, xmax=xmax, ymin=ymin,
                                            ymax=ymax, zmin=zmin, zmax=zmax,
                                            periodic_in_x=periodic_in_x,
                                            periodic_in_y=periodic_in_y,
                                            periodic_in_z=periodic_in_z,
                                            gamma_yx=gamma_yx,
                                            gamma_zx=gamma_zx,
                                            gamma_zy=gamma_zy,
                                            n_layers=n_layers,
                                            dt=self.dt,
                                            calls_per_step=N
                                            )
            else:
                self.domain = None
            # A seperate list for the nearest neighbourhood search is
            # benefitial since it is much smaller than the original one.
            nnps = LinkedListNNPS(dim=scheme.dim, particles=particles,
                                  radius_scale=kernel.radius_scale,
                                  domain=self.domain,
                                  fixed_h=False, cache=False, sort_gids=False)
            # The acceleration evaluator needs to be set up in order to compile
            # it together with the integrator.
            if parallel:
                self.acceleration_eval = AccelerationEval(
                            particle_arrays=particles,
                            equations=equations,
                            kernel=kernel)
            else:
                self.acceleration_eval = AccelerationEval(
                            particle_arrays=particles,
                            equations=equations,
                            kernel=kernel,
                            mode='serial')
            # Compilation of the integrator not using openmp, because the
            # overhead is too large for those few fiber particles.
            comp = SPHCompiler(self.acceleration_eval, self.fiber_integrator)
            if parallel:
                comp.compile()
            else:
                config = get_config()
                config.use_openmp = False
                comp.compile()
                config.use_openmp = True
            self.acceleration_eval.set_nnps(nnps)

            # Connecting neighbourhood list to integrator.
            self.fiber_integrator.set_nnps(nnps)

    def post_stage(self, current_time, dt, stage):
        """This post stage function gets called after each outer loop and
        starts an inner loop for the fiber iteration."""
        from math import ceil
        if self.innerloop:
            # 1) predictor
            # 2) post stage 1:
            if stage == 1:
                N = self.steps or int(ceil(self.dt/self.fiber_dt))
                for n in range(0, N):
                    self.fiber_integrator.step(current_time, dt/N)
                    current_time += dt/N
                    if self.domain_updates and self.domain:
                        self.domain.update()
            # 3) Evaluation
            # 4) post stage 2


class DensityCorrection(Tool):
    """
    A tool to reinitialize the density of the fluid particles
    """

    def __init__(self, app, arr_names, corr='shepard', freq=10, kernel=None):
        """
        Parameters
        ----------

        app : pysph.solver.application.Application.
            The application instance.
        arr_names : array
            Names of the particle arrays whose densities needs to be
            reinitialized.
        corr : str
            Name of the density reinitialization operation.
            corr='shepard' for using zeroth order shepard filter
        freq : int
            Frequency of reinitialization.
        kernel: any kernel from pysph.base.kernels

        """
        from pysph.solver.utils import get_array_by_name
        self.freq = freq
        self.corr = corr
        self.names = arr_names
        self.count = 1
        self._sph_eval = None
        self.kernel = kernel
        self.dim = app.solver.dim
        self.particles = app.particles
        self.arrs = [get_array_by_name(self.particles, i) for i in self.names]
        options = ['shepard', 'mls2d_1', 'mls3d_1']
        assert self.corr in options, 'corr should be one of %s' % options

    def _get_sph_eval_shepard(self):
        from pysph.sph.wc.density_correction import ShepardFilter
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.sph.equation import Group
        if self._sph_eval is None:
            arrs = self.arrs
            eqns = []
            for arr in arrs:
                name = arr.name
                arr.add_property('rhotmp')
                eqns.append(Group(equations=[
                            ShepardFilter(name, [name])], real=False))
            sph_eval = SPHEvaluator(
                arrays=arrs, equations=eqns, dim=self.dim,
                kernel=self.kernel(dim=self.dim))
            return sph_eval
        else:
            return self._sph_eval

    def _get_sph_eval_mls2d_1(self):
        from pysph.sph.wc.density_correction import MLSFirstOrder2D
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.sph.equation import Group
        if self._sph_eval is None:
            arrs = self.arrs
            eqns = []
            for arr in arrs:
                name = arr.name
                arr.add_property('rhotmp')
                eqns.append(Group(equations=[
                            MLSFirstOrder2D(name, [name])], real=False))
            sph_eval = SPHEvaluator(
                arrays=arrs, equations=eqns, dim=self.dim,
                kernel=self.kernel(dim=self.dim))
            return sph_eval
        else:
            return self._sph_eval

    def _get_sph_eval_mls3d_1(self):
        from pysph.sph.wc.density_correction import MLSFirstOrder3D
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.sph.equation import Group
        if self._sph_eval is None:
            arrs = self.arrs
            eqns = []
            for arr in arrs:
                name = arr.name
                arr.add_property('rhotmp')
                eqns.append(Group(equations=[
                            MLSFirstOrder3D(name, [name])], real=False))
            sph_eval = SPHEvaluator(
                arrays=arrs, equations=eqns, dim=self.dim,
                kernel=self.kernel(dim=self.dim))
            return sph_eval
        else:
            return self._sph_eval

    def _get_sph_eval(self, corr):
        if corr == 'shepard':
            return self._get_sph_eval_shepard()
        elif corr == 'mls2d_1':
            return self._get_sph_eval_mls2d_1()
        elif corr == 'mls3d_1':
            return self._get_sph_eval_mls3d_1()
        else:
            pass

    def post_step(self, solver):
        if self.freq == 0:
            pass
        elif self.count % self.freq == 0:
            self._sph_eval = self._get_sph_eval(self.corr)
            self._sph_eval.update()
            self._sph_eval.evaluate()
        self.count += 1
