"""Test if dumping and loading files works in parallel correctly.
"""

import mpi4py.MPI as mpi
import numpy as np
from os.path import join
import shutil
from tempfile import mkdtemp

from pysph.base.particle_array import ParticleArray
from pysph.solver.utils import dump, load

def assert_lists_same(l1, l2):
    expect = list(sorted(l1))
    result = list(sorted(l2))
    assert l1 == l2, "Expected %s, got %s"%(l1, l2)

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

root = mkdtemp()
filename = join(root, 'test.npz')

x = np.ones(5, dtype=float)*rank
pa = ParticleArray(name='fluid', constants={'c1': 0.0, 'c2': [0.0, 0.0]}, x=x)

try:
    dump(filename, [pa], {}, mpi_comm=comm)
    if rank == 0:
        data = load(filename)
        pa1 = data["arrays"]["fluid"]

        assert_lists_same(pa.properties.keys(), pa1.properties.keys())
        assert_lists_same(pa.constants.keys(), pa1.constants.keys())

        expect = np.ones(5*size)
        for i in range(size):
            expect[5*i:5*(i+1)] = i

        assert np.allclose(pa1.x, expect, atol=1e-14), \
            "Expected %s, got %s"%(expect, pa1.x)
finally:
    shutil.rmtree(root)
