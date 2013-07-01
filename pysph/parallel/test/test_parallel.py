"""Tests for the PySPH parallel module"""
from test_parallel_run import _run_example_script

import unittest
from nose.plugins.attrib import attr

class ParticleArrayExchangeTestCase(unittest.TestCase):

    @attr(slow=False, parallel=True)
    def test_lb_exchange(self):
        _run_example_script('./lb_exchange.py', nprocs=4)

    @attr(slow=False, parallel=True)            
    def test_remote_exchange(self):
        _run_example_script('./remote_exchange.py', nprocs=4)

class SummationDensityTestCase(unittest.TestCase):
    
    @attr(slow=False, parallel=True)
    def test_summation_density(self):
        _run_example_script('./summation_density.py', nprocs=4)
        
if __name__ == '__main__':
    unittest.main()
