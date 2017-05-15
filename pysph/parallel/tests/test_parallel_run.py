""" Module to run the example files and report their success/failure results

Add a function to the ExampleTest class corresponding to an example script to
be tested.
This is done till better strategy for parallel testing is implemented

"""

from pytest import mark, importorskip

from pysph.tools import run_parallel_script
from pysph.parallel.tests.example_test_case import ExampleTestCase, get_example_script


class ParallelTests(ExampleTestCase):

    @classmethod
    def setup_class(cls):
        importorskip("mpi4py.MPI")
        importorskip("pyzoltan.core.zoltan")

    @mark.slow
    @mark.parallel
    def test_3Ddam_break_example(self):
        serial_kwargs = dict(
            max_steps=50, pfreq=200, sort_gids=None, test=None
        )
        extra_parallel_kwargs = dict(ghost_layers=1, lb_freq=5)
        self.run_example(
            get_example_script('sphysics/dambreak_sphysics.py'),
            nprocs=4, atol=1e-12,
            serial_kwargs=serial_kwargs,
            extra_parallel_kwargs=extra_parallel_kwargs
        )

    @mark.slow
    @mark.parallel
    def test_elliptical_drop_example(self):
        serial_kwargs = dict(sort_gids=None, kernel='CubicSpline', tf=0.0038)
        extra_parallel_kwargs = dict(ghost_layers=1, lb_freq=5)
        self.run_example(
            'elliptical_drop.py', nprocs=2, atol=1e-11,
            serial_kwargs=serial_kwargs,
            extra_parallel_kwargs=extra_parallel_kwargs
        )

    @mark.parallel
    def test_ldcavity_example(self):
        max_steps = 150
        serial_kwargs = dict(max_steps=max_steps, pfreq=500, sort_gids=None)
        extra_parallel_kwargs = dict(ghost_layers=2, lb_freq=5)
        self.run_example(
            'cavity.py', nprocs=4, atol=1e-14, serial_kwargs=serial_kwargs,
            extra_parallel_kwargs=extra_parallel_kwargs
        )

if __name__ == '__main__':
    import unittest
    unittest.main()
