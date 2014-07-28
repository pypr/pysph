from pysph.base.ext_module import ExtModule
from pysph.sph.integrator_cython_helper import IntegratorCythonHelper
from pysph.sph.acceleration_eval_cython_helper import AccelerationEvalCythonHelper


###############################################################################
class SPHCompiler(object):
    def __init__(self, acceleration_eval, integrator):
        self.acceleration_eval = acceleration_eval
        self.acceleration_eval_helper = AccelerationEvalCythonHelper(
            self.acceleration_eval
        )
        self.integrator = integrator
        self.integrator_helper = IntegratorCythonHelper(integrator)
        self.ext_mod = None

    ##########################################################################
    # Public interface.
    ##########################################################################
    def get_code(self):
        main = self.acceleration_eval_helper.get_code()
        if self.integrator is not None:
            integrator_code = self.integrator_helper.get_code()
        else:
            integrator_code = ''
        return main + integrator_code

    def set_nnps(self, nnps):
        if self.ext_mod is None:
            self.setup()
        self.acceleration_eval.set_nnps(nnps)
        if self.integrator is not None:
            self.integrator.set_nnps(nnps)

    def setup(self):
        """Always call this first.
        """
        code = self.get_code()
        self.ext_mod = ExtModule(code, verbose=True)
        mod = self.ext_mod.load()
        self.acceleration_eval_helper.setup_compiled_module(mod)
        calc = self.acceleration_eval.calc
        if self.integrator is not None:
            self.integrator_helper.setup_compiled_module(mod, calc)
