class SPHCompiler(object):
    def __init__(self, acceleration_evals, integrator):
        """Compiles the acceleration evaluator and integrator to produce a
        fast version using one of the supported backends. If the backend is
        not given, one is automatically chosen based on the configuration.

        Parameters
        ----------

        acceleration_eval: .acceleration_eval.AccelerationEval instance
            or list of instances.
        integrator: .integrator.Integrator instance
        """
        if not isinstance(acceleration_evals, (list, tuple)):
            acceleration_evals = [acceleration_evals]

        self.acceleration_evals = list(acceleration_evals)
        self.integrator = integrator
        if integrator is not None:
            integrator.set_acceleration_evals(self.acceleration_evals)
        self.backend = acceleration_evals[0].backend
        self._setup_helpers()
        self.module = None

    # Public interface. ####################################################
    def compile(self):
        """Compile the generated code to extension modules and
        setup the objects that need this by calling their
        setup_compiled_module.
        """
        if self.module is not None:
            return

        # We compile the first acceleration eval along with the integrator.
        # The rest of the acceleration evals (if present) are independent.
        code0 = self._get_code()
        helper0 = self.acceleration_eval_helpers[0]
        mod = helper0.compile(code0)
        helper0.setup_compiled_module(mod)

        self.module = mod
        c_accel_eval0 = self.acceleration_evals[0].c_acceleration_eval

        if self.backend == 'cython':
            if self.integrator is not None:
                self.integrator_helper.setup_compiled_module(
                    mod, c_accel_eval0
                )
        elif self.backend == 'opencl' or self.backend == 'cuda':
            if self.integrator is not None:
                self.integrator_helper.setup_compiled_module(
                    mod, c_accel_eval0
                )

        # Setup the remaining acceleration evals.
        for helper in self.acceleration_eval_helpers[1:]:
            mod = helper.compile(helper.get_code())
            helper.setup_compiled_module(mod)

    # Private interface. ####################################################
    def _get_code(self):
        main = self.acceleration_eval_helpers[0].get_code()
        integrator_code = self.integrator_helper.get_code()
        return main + integrator_code

    def _setup_helpers(self):
        if self.backend == 'cython':
            from .acceleration_eval_cython_helper import \
                AccelerationEvalCythonHelper
            cls = AccelerationEvalCythonHelper
        elif self.backend == 'opencl' or self.backend == 'cuda':
            from .acceleration_eval_gpu_helper import \
                AccelerationEvalGPUHelper
            cls = AccelerationEvalGPUHelper

        self.acceleration_eval_helpers = [
            cls(a_eval)
            for a_eval in self.acceleration_evals
        ]
        self._setup_integrator_helper()

    def _setup_integrator_helper(self):
        a_helper0 = self.acceleration_eval_helpers[0]
        if self.backend == 'cython':
            from .integrator_cython_helper import \
                IntegratorCythonHelper
            self.integrator_helper = IntegratorCythonHelper(
                self.integrator, a_helper0
            )
        elif self.backend == 'opencl' or self.backend == 'cuda':
            from .integrator_gpu_helper import IntegratorGPUHelper
            self.integrator_helper = IntegratorGPUHelper(
                self.integrator, a_helper0
            )
