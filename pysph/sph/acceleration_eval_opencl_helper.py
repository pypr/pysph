from mako.template import Template
from os.path import dirname, join

from .acceleration_eval_cython_helper import (
    get_all_array_names, get_known_types_for_arrays
)


class OpenCLAccelerationEval(object):
    """Does the actual work of performing the evaluation.
    """
    def __init__(self, helper):
        self.helper = helper

    def compute(self, t, dt):
        pass

    def set_nnps(self, nnps):
        pass

    def update_particle_arrays(self, arrays):
        pass


class AccelerationEvalOpenCLHelper(object):
    def __init__(self, acceleration_eval):
        self.object = acceleration_eval
        self.all_array_names = get_all_array_names(
            self.object.particle_arrays
        )
        self.known_types = get_known_types_for_arrays(
            self.all_array_names
        )

    ##########################################################################
    # Public interface.
    ##########################################################################
    def get_code(self):
        path = join(dirname(__file__), 'acceleration_eval_opencl.mako')
        template = Template(filename=path)
        main = template.render(helper=self)
        return main

    def setup_compiled_module(self, module=None):
        object = self.object
        acceleration_eval = OpenCLAccelerationEval(self)
        object.set_compiled_object(acceleration_eval)

    ##########################################################################
    # Mako interface.
    ##########################################################################
