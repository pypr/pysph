"""Tests for the PySPH parallel module"""


import os
import shutil
import tempfile
import unittest
from pytest import mark, importorskip
from pysph.tools import run_parallel_script


path = run_parallel_script.get_directory(__file__)


class ParticleArrayExchangeTestCase(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        importorskip("mpi4py.MPI")
        importorskip("pyzoltan.core.zoltan")

    @mark.parallel
    def test_lb_exchange(self):
        run_parallel_script.run(filename='lb_exchange.py', nprocs=4, path=path)

    @mark.parallel
    def test_remote_exchange(self):
        run_parallel_script.run(filename='remote_exchange.py', nprocs=4, path=path)


class SummationDensityTestCase(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        importorskip("mpi4py.MPI")
        importorskip("pyzoltan.core.zoltan")

    @mark.slow
    @mark.parallel
    def test_summation_density(self):
        run_parallel_script.run(filename='summation_density.py', nprocs=4,
                                path=path)


class MPIReduceArrayTestCase(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        importorskip("mpi4py.MPI")
        importorskip("pyzoltan.core.zoltan")

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    @mark.parallel
    def test_mpi_reduce_array(self):
        run_parallel_script.run(
            filename='reduce_array.py', nprocs=4, path=path
        )

    @mark.parallel
    def test_parallel_reduce(self):
        args = ['--directory=%s'%self.root]
        run_parallel_script.run(
            filename='simple_reduction.py', args=args, nprocs=4, path=path
        )


class DumpLoadTestCase(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        importorskip("mpi4py.MPI")
        importorskip("pyzoltan.core.zoltan")

    @mark.parallel
    def test_dump_and_load_work_in_parallel(self):
        run_parallel_script.run(
            filename='check_dump_load.py', nprocs=4, path=path
        )


if __name__ == '__main__':
    unittest.main()
