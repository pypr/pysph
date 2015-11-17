"""A convenience class that combines an AccelerationEval and an SPHCompiler to
allow a user to specify particle arrays, equations, an optional domain and
kernel to produce an SPH evaluation.

This is handy for post-processing.

"""

from pysph.base.kernels import Gaussian
from pysph.base.nnps import LinkedListNNPS as NNPS
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.sph_compiler import SPHCompiler


class SPHEvaluator(object):
    def __init__(self, arrays, equations, dim, kernel=None,
                 domain_manager=None):
        """Constructor.

        Parameters
        ----------
        arrays: list(ParticleArray)
        equations: list
        dim: int
        kernel: kernel instance.
        domain_manager: DomainManager
        """
        self.arrays = arrays
        self.equations = equations
        self.domain_manager = domain_manager
        self.dim = dim
        if kernel is None:
            self.kernel = Gaussian(dim=dim)
        else:
            self.kernel = kernel

        self.func_eval = AccelerationEval(arrays, equations, self.kernel)
        compiler = SPHCompiler(self.func_eval, None)
        compiler.compile()
        self._create_nnps(arrays)

    def evaluate(self, t=0.0, dt=0.1):
        """Evalute the SPH equations, dummy t and dt values can
        be passed.
        """
        self.func_eval.compute(t, dt)

    def update_particle_arrays(self, arrays):
        self._create_nnps(arrays)
        self.func_eval.update_particle_arrays(arrays)

    #### Private protocol ###################################################
    def _create_nnps(self, arrays):
        self.nnps = NNPS(dim=self.kernel.dim, particles=arrays,
                         radius_scale=self.kernel.radius_scale,
                         domain=self.domain_manager,
                         cache=True)
        self.nnps.update()
        self.func_eval.set_nnps(self.nnps)
