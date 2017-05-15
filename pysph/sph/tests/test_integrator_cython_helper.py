import unittest

import numpy as np

from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline

from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.acceleration_eval_cython_helper import AccelerationEvalCythonHelper
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.integrator_cython_helper import IntegratorCythonHelper

from pysph.sph.basic_equations import SummationDensity


class TestIntegratorCythonHelper(unittest.TestCase):
    def test_invalid_kwarg_raises_error(self):
        # Given
        x = np.linspace(0, 1, 10)
        pa = get_particle_array(name='fluid', x=x)
        equations = [SummationDensity(dest='fluid', sources=['fluid'])]
        kernel = QuinticSpline(dim=1)
        a_eval = AccelerationEval([pa], equations, kernel=kernel)
        a_helper = AccelerationEvalCythonHelper(a_eval)

        # When/Then
        integrator = PECIntegrator(f=WCSPHStep())
        self.assertRaises(
            RuntimeError,
            IntegratorCythonHelper,
            integrator, a_helper
        )


if __name__ == '__main__':
    unittest.main()
