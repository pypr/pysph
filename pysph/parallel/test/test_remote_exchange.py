"""Test the remote exchange data of ParticleArrayExchange

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

Remote neighbors are those nodes which share an edge with a node
belonging to another processor. For example, the number of remote
neighbors for processor 0 in this case is 6.

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

# create the data
if rank == 0:
    numPoints = 7
    x = np.array( [0.0, 1.0, 0.0, 1.0,
                   0.0, 1.0, 2.0] )

    y = np.array( [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0] )

    gid = np.array( [0, 1, 5, 6, 10, 11, 12], dtype=np.uint32 )

    exportLocalids = np.array( [1, 3, 4, 5, 6, 6, 6], dtype=np.uint32 )
    exportProcs = np.array(    [2, 2, 1, 1, 1, 2, 3], dtype=np.int32 )

    numExport = 7
    numRemote = 6

if rank == 1:
    numPoints = 6
    x = np.array( [0.0, 1.0, 2.0, 0.0, 1.0, 2.0] )

    y = np.array( [3.0, 3.0, 3.0,
                   4.0, 4.0, 4.0] )

    gid = np.array( [15, 16, 17, 20, 21, 22], dtype=np.uint32 )

    exportLocalids = np.array( [0, 1, 2, 2, 5], dtype=np.uint32 )
    exportProcs = np.array(    [0, 0, 0, 3, 3], dtype=np.int32 )

    numExport = 5
    numRemote = 5

if rank == 2:
    numPoints = 6
    x = np.array( [2.0, 3.0, 4.0, 2.0, 3.0, 4.0] )
    y = np.array( [0.0, 0.0, 0.0, 1.0, 1.0, 1.0] )
    gid = np.array( [2, 3, 4, 7, 8, 9], dtype=np.uint32 )

    exportLocalids = np.array( [0, 3, 4, 5], dtype=np.uint32 )
    exportProcs = np.array(    [0, 0, 3, 3], dtype=np.int32 )

    numExport = 4
    numRemote = 5

if rank == 3:
    numPoints = 6
    x = np.array( [3.0, 4.0, 3.0, 4.0, 3.0, 4.0] )
    y = np.array( [2.0, 2.0, 3.0, 3.0, 4.0, 4.0] )
    gid = np.array( [13, 14, 18, 19, 23, 24], dtype=np.uint32 )

    exportLocalids = np.array( [0, 0, 1, 2, 4], dtype=np.uint32 )
    exportProcs = np.array(    [0, 2, 2, 1, 1], dtype=np.int32 )

    numExport = 5
    numRemote = 5

# Global data
X = np.array( [0,1,2,3,4,
               0,1,2,3,4,
               0,1,2,3,4,
               0,1,2,3,4,
               0,1,2,3,4,], dtype=np.float64 )

Y = np.array( [0,0,0,0,0,
               1,1,1,1,1,
               2,2,2,2,2,
               3,3,3,3,3,
               4,4,4,4,4], dtype=np.float64 )

GID = np.array( range(25), dtype=np.uint32 )

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

# call remote_exchange data with these lists
pae.remote_exchange_data()

# the added particles should be remote
tag = pa.get('tag', only_real_particles=False)
assert( pa.num_real_particles == numPoints )
assert( pa.get_number_of_particles() == numPoints + numRemote )

assert( np.allclose(tag[numPoints:], 1) )

# now check the data on each array
numParticles = numPoints + numRemote

for i in range(numParticles):
    x, y, gid = pa.get('x', 'y', 'gid', only_real_particles=False)

    assert( abs(X[gid[i]] - x[i]) < 1e-15 )
    assert( abs(Y[gid[i]] - y[i]) < 1e-15 )
    assert( GID[gid[i]] == gid[i] )
