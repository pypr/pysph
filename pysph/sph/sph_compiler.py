class SPHCompiler(object):
    def __init__(self, acceleration_eval, integrator):
        """Compiles the acceleration evaluator and integrator to produce a
        fast version using one of the supported backends. If the backend is
        not given, one is automatically chosen based on the configuration.

        Parameters
        ----------

        acceleration_eval: .acceleration_eval.AccelerationEval instance
        integrator: .integrator.Integrator instance
        """
        self.acceleration_eval = acceleration_eval
        self.integrator = integrator
        self.backend = acceleration_eval.backend
        self._setup_helpers()
        self.module = None

    # Public interface. ####################################################
    def compile(self):
        """Compile the generated code to an extension module and
        setup the objects that need this by calling their
        setup_compiled_module.
        """
        if self.module is not None:
            return
        code = self._get_code()
        mod = self.acceleration_eval_helper.compile(code)
        self.module = mod
        self.acceleration_eval_helper.setup_compiled_module(mod)
        if self.backend == 'cython':
            cython_a_eval = self.acceleration_eval.c_acceleration_eval
            if self.integrator is not None:
                self.integrator_helper.setup_compiled_module(
                    mod, cython_a_eval
                )
        elif self.backend == 'opencl':
            if self.integrator is not None:
                c_a_eval = self.acceleration_eval.c_acceleration_eval
                self.integrator_helper.setup_compiled_module(
                    mod, c_a_eval
                )

    # Private interface. ####################################################
    def _get_code(self):
        main = self.acceleration_eval_helper.get_code()
        integrator_code = self.integrator_helper.get_code()
        return main + integrator_code

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
            from .integrator_opencl_helper import IntegratorOpenCLHelper

            self.acceleration_eval_helper = AccelerationEvalOpenCLHelper(
                self.acceleration_eval
            )
            self.integrator_helper = IntegratorOpenCLHelper(
                self.integrator, self.acceleration_eval_helper
            )
