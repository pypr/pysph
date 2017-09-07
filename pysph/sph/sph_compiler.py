from pysph.base.config import get_config
from pysph.base.ext_module import ExtModule


###############################################################################
class SPHCompiler(object):
    def __init__(self, acceleration_eval, integrator, backend=None):
        """Compiles the acceleration evaluator and integrator to produce a
        fast version using one of the supported backends. If the backend is
        not given, one is automatically chosen based on the configuration.

        Parameters
        ----------

        acceleration_eval: .acceleration_eval.AccelerationEval instance
        integrator: .integrator.Integrator instance
        backend: str: indicates the backend to use.
            one of ('opencl', 'cython', '', None)

        """
        assert backend in ('opencl', 'cython', '', None)
        self.acceleration_eval = acceleration_eval
        self.integrator = integrator
        self.backend = self._get_backend(backend)
        self._setup_helpers()
        self.ext_mod = None
        self.module = None

    # Public interface. ####################################################
    def compile(self):
        """Compile the generated code to an extension module and
        setup the objects that need this by calling their
        setup_compiled_module.
        """
        if self.ext_mod is not None:
            return
        code = self._get_code()
        if self.backend == 'cython':
            # Note, we do not add carray or particle_array as nnps_base would
            # have been rebuilt anyway if they changed.
            depends = ["pysph.base.nnps_base"]
            self.ext_mod = ExtModule(code, verbose=True, depends=depends)
            mod = self.ext_mod.load()
            self.module = mod

            self.acceleration_eval_helper.setup_compiled_module(mod)
            cython_a_eval = self.acceleration_eval.c_acceleration_eval
            if self.integrator is not None:
                self.integrator_helper.setup_compiled_module(
                    mod, cython_a_eval
                )
        elif self.backend == 'opencl':
            self.acceleration_eval_helper.setup_compiled_module()

    # Private interface. ####################################################
    def _get_code(self):
        main = self.acceleration_eval_helper.get_code()
        # FIXME
        if self.integrator_helper is None:
            integrator_code = ''
        else:
            integrator_code = self.integrator_helper.get_code()
        return main + integrator_code

    def _get_backend(self, backend):
        if not backend:
            cfg = get_config()
            if cfg.use_opencl:
                backend = 'opencl'
            else:
                backend = 'cython'
        return backend

    def _setup_helpers(self):
        if self.backend == 'cython':
            from .integrator_cython_helper import \
                IntegratorCythonHelper
            from .acceleration_eval_cython_helper import \
                AccelerationEvalCythonHelper
            self.acceleration_eval_helper = AccelerationEvalCythonHelper(
                self.acceleration_eval
            )
            self.integrator_helper = IntegratorCythonHelper(
                self.integrator, self.acceleration_eval_helper
            )
        elif self.backend == 'opencl':
            from .acceleration_eval_opencl_helper import \
                AccelerationEvalOpenCLHelper
            # FIXME
            # from .integrator_opencl_helper import IntegratorOpenCLHelper

            self.acceleration_eval_helper = AccelerationEvalOpenCLHelper(
                self.acceleration_eval
            )
            # FIXME
            self.integrator_helper = None
