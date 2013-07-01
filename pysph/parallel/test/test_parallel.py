"""Tests for the PySPH parallel module"""

try:
    import mpi4py.MPI as mpi
except ImportError:
    import nose.plugins.skip as skip
    reason = "mpi4py not installed"
    raise skip.SkipTest(reason)

import unittest
from test_parallel_run import _run_example_script

class ParticleArrayExchangeTestCase(unittest.TestCase):
    def test_lb_exchange(self):
        _run_example_script('lb_exchange.py', nprocs=4)
            
    def test_remote_exchange(self):
        _run_example_script('remote_exchange.py', nprocs=4)

class SummationDensityTestCase(unittest.TestCase):
    def test_sd(self):
        _run_example_script('summation_density_test.py', nprocs=4)
        
if __name__ == '__main__':
    unittest.main()
