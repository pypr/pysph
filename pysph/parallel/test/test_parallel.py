"""Tests for the PySPH parallel module"""
from pysph.tools import run_parallel_script

import unittest
from nose.plugins.attrib import attr

path = run_parallel_script.get_directory(__file__)

class ParticleArrayExchangeTestCase(unittest.TestCase):

    @attr(slow=False, parallel=True)
    def test_lb_exchange(self):
        run_parallel_script.run(filename='./lb_exchange.py', nprocs=4, path=path)

    @attr(slow=False, parallel=True)
    def test_remote_exchange(self):
        run_parallel_script.run(filename='./remote_exchange.py', nprocs=4, path=path)

class SummationDensityTestCase(unittest.TestCase):

    @attr(slow=True, parallel=True)
    def test_summation_density(self):
        run_parallel_script.run(filename='./summation_density.py', nprocs=4,
                                path=path)

if __name__ == '__main__':
    unittest.main()
