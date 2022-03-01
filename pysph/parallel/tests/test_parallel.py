"""Tests for the PySPH parallel module"""

import shutil
import tempfile
import unittest

import numpy as np
from pytest import mark, importorskip
from pysph.tools import run_parallel_script

path = run_parallel_script.get_directory(__file__)


class ParticleArrayTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        importorskip("pysph.parallel.parallel_manager")

    def test_get_strided_indices(self):
        # Given
        from pysph.parallel.parallel_manager import get_strided_indices

        indices = np.array([1, 5, 3])

        # When
        idx = get_strided_indices(indices, 1)
        # Then
        np.testing.assert_array_equal(idx, indices)

        # When
        idx = get_strided_indices(indices, 2)
        # Then
        np.testing.assert_array_equal(
            idx, [2, 3, 10, 11, 6, 7]
        )

        # When
        idx = get_strided_indices(indices, 3)
        # Then
        np.testing.assert_array_equal(
            idx, [3, 4, 5, 15, 16, 17, 9, 10, 11]
        )


class ParticleArrayExchangeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        importorskip("mpi4py.MPI")
        importorskip("pyzoltan.core.zoltan")

    @mark.parallel
    def test_lb_exchange(self):
        run_parallel_script.run(filename='lb_exchange.py', nprocs=4, path=path)

    @mark.parallel
    def test_remote_exchange(self):
        run_parallel_script.run(
            filename='remote_exchange.py', nprocs=4, path=path
        )


class SummationDensityTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        importorskip("mpi4py.MPI")
        importorskip("pyzoltan.core.zoltan")

    @mark.slow
    @mark.parallel
    def test_summation_density(self):
        run_parallel_script.run(
            filename='summation_density.py', nprocs=4, path=path
        )


class MPIReduceArrayTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        args = ['--directory=%s' % self.root]
        run_parallel_script.run(
            filename='simple_reduction.py', args=args, nprocs=4, path=path,
            timeout=60.0
        )


class DumpLoadTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        importorskip("mpi4py.MPI")
        importorskip("pyzoltan.core.zoltan")

    @mark.parallel
    def test_dump_and_load_work_in_parallel(self):
        run_parallel_script.run(
            filename='check_dump_load.py', nprocs=4, path=path
        )


if __name__ == '__main__':
    unittest.main()
