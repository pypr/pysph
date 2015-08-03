
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
    def __init__(self, app, array_name, props, freq=100, xi=None, yi=None, zi=None):
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
        self.interp = Interpolator(
            self.particles, x=self.xi, y=self.yi, z=self.zi,
            kernel=app.solver.kernel,
            domain_manager=app.create_domain()
        )

    def post_step(self, solver):
        if solver.count%self.freq == 0 and solver.count > 0:
            self.interp.nnps.update()
            data = dict(x=self.xi, y=self.yi, z=self.zi)
            for prop in self.props:
                data[prop] = self.interp.interpolate(prop)
            self.array.set(**data)
