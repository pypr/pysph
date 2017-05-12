"""Running script for Zoltan"""

import unittest
from pytest import mark, importorskip

from pysph.tools import run_parallel_script

path = run_parallel_script.get_directory(__file__)

class PyZoltanTests(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        importorskip("mpi4py.MPI")
        importorskip("pyzoltan.core.zoltan")

    @mark.parallel
    def test_zoltan_geometric_partitioner(self):
        run_parallel_script.run(
            filename='geometric_partitioner.py', nprocs=4, path=path
        )

    @mark.slow
    @mark.parallel
    def test_zoltan_partition(self):
        run_parallel_script.run(
            filename='3d_partition.py', nprocs=4, timeout=90.0, path=path
        )

    @mark.parallel
    def test_zoltan_zcomm(self):
        run_parallel_script.run(
            filename='zcomm.py', nprocs=4, path=path
        )

if __name__ == '__main__':
    unittest.main()
