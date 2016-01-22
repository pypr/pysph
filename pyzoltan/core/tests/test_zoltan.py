"""Running script for Zoltan"""

import unittest
from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest

from pysph.tools import run_parallel_script

run_parallel_script.skip_if_no_mpi4py()
try:
    from pyzoltan.core import zoltan
except ImportError:
    raise SkipTest('Build does not support Zoltan.')

path = run_parallel_script.get_directory(__file__)

class PyZoltanTests(unittest.TestCase):

    @attr(parallel=True)
    def test_zoltan_geometric_partitioner(self):
        run_parallel_script.run(
            filename='geometric_partitioner.py', nprocs=4, path=path
        )

    @attr(slow=True, parallel=True)
    def test_zoltan_partition(self):
        run_parallel_script.run(
            filename='3d_partition.py', nprocs=4, timeout=90.0, path=path
        )

    @attr(parallel=True)
    def test_zoltan_zcomm(self):
        run_parallel_script.run(
            filename='zcomm.py', nprocs=4, path=path
        )

if __name__ == '__main__':
    unittest.main()
