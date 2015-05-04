"""Test the ParticleArrayExchange object with the following data

25 particles are created on a rectangular grid with the following
processor assignment:

2---- 3---- 3---- 3---- 3
|     |     |     |     |
1---- 1---- 1---- 2---- 2
|     |     |     |     |
0---- 0---- 1---- 1---- 1
|     |     |     |     |
0---- 0---- 0---- 0---- 0
|     |     |     |     |
0---- 0---- 0---- 0---- 0


We assume that after a load balancing step, we have the following
distribution:

1---- 1---- 1---- 3---- 3
|     |     |     |     |
1---- 1---- 1---- 3---- 3
|     |     |     |     |
0---- 0---- 0---- 3---- 3
|     |     |     |     |
0---- 0---- 2---- 2---- 2
|     |     |     |     |
0---- 0---- 2---- 2---- 2

We create a ParticleArray to represent the initial distribution and
use ParticleArrayExchange to move the data by manually setting the
particle import/export lists. We require that the test be run with 4
processors.

"""
import mpi4py.MPI as mpi
import numpy as np

from pysph.parallel.parallel_manager import ParticleArrayExchange
from pysph.base.utils import get_particle_array_wcsph

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 4:
    if rank == 0:
        raise RuntimeError("Run this test with 4 processors")

# create the initial distribution
if rank == 0:
    numPoints = 12
    x = np.array( [0.0, 1.0, 2.0, 3.0, 4.0,
                   0.0, 1.0, 2.0, 3.0, 4.0,
                   0.0, 1.0] )

    y = np.array( [0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 1.0, 1.0, 1.0, 1.0,
                   2.0, 2.0] )

    gid = np.array( [0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10,
                     11], dtype=np.uint32 )

    exportLocalids = np.array( [2, 3, 4, 7, 8, 9], dtype=np.uint32 )
    exportProcs = np.array( [2, 2, 2, 2, 2, 2], dtype=np.int32 )
    numExport = 6

    parts = np.ones(shape=numPoints, dtype=np.int32) * rank

if rank == 1:
    numPoints = 6
    x = np.array( [2.0, 3.0, 4.0,
                   0.0, 1.0, 2.0] )

    y = np.array( [2.0, 2.0, 2.0,
                   3.0, 3.0, 3.0] )

    gid = np.array( [12, 13, 14, 15,
                     16, 17], dtype=np.uint32 )

    exportLocalids = np.array( [0, 1, 2], dtype=np.uint32 )
    exportProcs = np.array( [0, 3, 3], dtype=np.int32 )
    numExport = 3

    parts = np.ones(shape=numPoints, dtype=np.int32) * rank

if rank == 2:
    numPoints = 3
    x = np.array( [4.0, 5.0, 0.0] )
    y = np.array( [3.0, 3.0, 4.0] )
    gid = np.array( [18, 19, 20], dtype=np.uint32 )

    exportLocalids = np.array( [0, 1, 2], dtype=np.uint32 )
    exportProcs = np.array( [3, 3, 1], dtype=np.int32 )
    numExport = 3

    parts = np.ones(shape=numPoints, dtype=np.int32) * rank

if rank == 3:
    numPoints = 4
    x = np.array( [1.0, 2.0, 3.0, 4.0] )
    y = np.array( [4.0, 4.0, 4.0, 4.0] )
    gid = np.array( [21, 22, 23, 24], dtype=np.uint32 )

    exportLocalids = np.array( [0,1], dtype=np.uint32 )
    exportProcs = np.array( [1, 1], dtype=np.int32 )
    numExport = 2

    parts = np.ones(shape=numPoints, dtype=np.int32) * rank

# Gather the Global data on root
X = np.zeros(shape=25, dtype=np.float64)
Y = np.zeros(shape=25, dtype=np.float64)
GID = np.zeros(shape=25, dtype=np.uint32)

displacements = np.array( [12, 6, 3, 4], dtype=np.int32 )

comm.Gatherv(sendbuf=[x, mpi.DOUBLE], recvbuf=[X, (displacements, None)], root=0)
comm.Gatherv(sendbuf=[y, mpi.DOUBLE], recvbuf=[Y, (displacements, None)], root=0)
comm.Gatherv(sendbuf=[gid, mpi.UNSIGNED_INT], recvbuf=[GID, (displacements, None)], root=0)

# broadcast global X, Y and GID to everyone
comm.Bcast(buf=X, root=0)
comm.Bcast(buf=Y, root=0)
comm.Bcast(buf=GID, root=0)

# create the local particle arrays and exchange objects
pa = get_particle_array_wcsph(name='test', x=x, y=y, gid=gid)
pae = ParticleArrayExchange(pa_index=0, pa=pa, comm=comm)

# set the export indices for each array
pae.reset_lists()
pae.numParticleExport = numExport
pae.exportParticleLocalids.resize(numExport)
pae.exportParticleLocalids.set_data( exportLocalids )

pae.exportParticleProcs.resize( numExport )
pae.exportParticleProcs.set_data( exportProcs )

# call lb_balance with these lists
pae.lb_exchange_data()

# All arrays must be local after lb_exchange_data
assert( pa.num_real_particles == pa.get_number_of_particles() )
assert( np.allclose(pa.tag, 0) )

# now check the data on each array
numParticles = 6
if rank == 0:
    numParticles = 7

assert( pa.num_real_particles == numParticles )

for i in range(numParticles):
    assert( abs(X[pa.gid[i]] - pa.x[i]) < 1e-15 )
    assert( abs(Y[pa.gid[i]] - pa.y[i]) < 1e-15 )
    assert( GID[pa.gid[i]] == pa.gid[i] )
