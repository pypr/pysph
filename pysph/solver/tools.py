
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


