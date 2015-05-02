"""Running script for Zoltan"""

import unittest
from nose.plugins.attrib import attr

from pysph.tools import run_parallel_script

run_parallel_script.skip_if_no_mpi4py()
path = run_parallel_script.get_directory(__file__)

class PyZoltanTests(unittest.TestCase):

    @attr(slow=False, parallel=True)
    def _test_zoltan_geometric_partitioner(self):
        run_parallel_script.run(
            path=path, filename='geometric_partitioner.py', nprocs=4, args=['>mesh.out'])

    @attr(slow=True, parallel=True)
    def test_zoltan_partition(self):
        run_parallel_script.run(
            path=path, filename='3d_partition.py', nprocs=4, timeout=40.0)

    @attr(slow=False, parallel=True)
    def test_zoltan_zcomm(self):
        run_parallel_script.run(
            path=path, filename='zcomm.py', nprocs=4)

if __name__ == '__main__':
    unittest.main()
