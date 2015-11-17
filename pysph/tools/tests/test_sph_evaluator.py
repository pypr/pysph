import unittest

import numpy as np

from pysph.base.utils import get_particle_array
from pysph.base.nnps import DomainManager
from pysph.sph.basic_equations import SummationDensity
from pysph.tools.sph_evaluator import SPHEvaluator


class TestSPHEvaluator(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0, 1, 10)
        dx = x[1] - x[0]
        self.dx = dx
        m = np.ones_like(x)
        h = np.ones_like(x)*dx
        self.src = get_particle_array(name='src', x=x, m=m, h=h)
        self.equations = [SummationDensity(dest='dest', sources=['src'])]

    def test_evaluation(self):
        # Given
        xd = [0.5]
        hd = self.src.h[:1]
        dest = get_particle_array(name='dest', x=xd, h=hd)
        sph_eval = SPHEvaluator(
            arrays=[dest, self.src], equations=self.equations, dim=1
        )

        # When.
        sph_eval.evaluate()

        # Then.
        self.assertAlmostEqual(dest.rho[0], 9.0, places=2)

    def test_evaluation_with_domain_manager(self):
        # Given
        xd = [0.0]
        hd = self.src.h[:1]
        dest = get_particle_array(name='dest', x=xd, h=hd)
        dx = self.dx
        dm = DomainManager(xmin=-dx/2, xmax=1.0+dx/2, periodic_in_x=True)
        sph_eval = SPHEvaluator(
            arrays=[dest, self.src], equations=self.equations, dim=1,
            domain_manager=dm
        )

        # When.
        sph_eval.evaluate()

        # Then.
        self.assertAlmostEqual(dest.rho[0], 9.0, places=2)

    def test_updating_particle_arrays(self):
        # Given
        xd = [0.5]
        hd = self.src.h[:1]
        dest = get_particle_array(name='dest', x=xd, h=hd)
        sph_eval = SPHEvaluator(
            [dest, self.src], equations=self.equations, dim=1
        )
        sph_eval.evaluate()
        rho0 = dest.rho[0]

        # When.
        dest.x[0] = 0.0
        sph_eval.update_particle_arrays([dest, self.src])
        sph_eval.evaluate()

        # Then.
        self.assertNotEqual(rho0, dest.rho[0])
        self.assertAlmostEqual(dest.rho[0], 7.0, places=1)


if __name__ == '__main__':
    unittest.main()
