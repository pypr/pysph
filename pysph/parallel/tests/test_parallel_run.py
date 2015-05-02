""" Module to run the example files and report their success/failure results

Add a function to the ExampleTest class corresponding to an example script to
be tested.
This is done till better strategy for parallel testing is implemented

"""

import unittest

from nose.plugins.attrib import attr

from pysph.tools import run_parallel_script
from example_test_case import ExampleTestCase

run_parallel_script.skip_if_no_mpi4py()


class ParallelTests(ExampleTestCase):

    @attr(slow=True, very_slow=True, parallel=True)
    def test_3Ddam_break_example(self):
        dt = 1e-5; tf = 100*dt
        serial_kwargs = dict(timestep=dt, tf=tf)
        extra_parallel_kwargs = dict(ghost_layers=1, lb_freq=5)
        self.run_example(
            'dambreak3D.py', nprocs=4, atol=1e-4,
            serial_kwargs=serial_kwargs,
            extra_parallel_kwargs=extra_parallel_kwargs
        )

    @attr(slow=True, parallel=True)
    def test_ldcavity_example(self):
        dt=1e-4; tf=200*dt
        serial_kwargs = dict(timestep=dt, tf=tf)
        extra_parallel_kwargs = dict(ghost_layers=3, lb_freq=5)
        self.run_example(
            'cavity.py', nprocs=4, atol=1e-4, serial_kwargs=serial_kwargs,
            extra_parallel_kwargs=extra_parallel_kwargs
        )

if __name__ == "__main__":
    unittest.main()
