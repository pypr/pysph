
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
