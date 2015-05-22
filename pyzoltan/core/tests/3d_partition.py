"""3 dimensional tests for the Zoltan partitioner. The test follows
the same pattern as the 2D test.

To see the output from the script try the following::

    $ mpirun -np 4 python 3d_partition.py --plot

"""
import sys

import mpi4py.MPI as mpi
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from pyzoltan.core.carray import UIntArray, DoubleArray
from pyzoltan.core import zoltan

from numpy import random
import numpy as np


colors = ['r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood']

def plot_points(x, y, z, slice_data, title, filename):
    if '--plot' not in sys.argv:
        return

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    s1 = fig.add_subplot(111)
    s1.axes = Axes3D(fig)
    for i in range(size):
        s1.axes.plot3D(
            x[slice_data[i]], y[slice_data[i]], z[slice_data[i]],
            c=colors[i], marker='o', linestyle='None', alpha=0.5
        )

    s1.axes.set_xlabel( 'X' )
    s1.axes.set_ylabel( 'Y' )
    s1.axes.set_zlabel( 'Z' )

    plt.title(title)
    plt.savefig(filename)


numPoints = 1<<12

x = random.random( numPoints )
y = random.random( numPoints )
z = random.random( numPoints )
gid = np.arange( numPoints*size, dtype=np.uint32 )[rank*numPoints:(rank+1)*numPoints]

X = np.zeros( size * numPoints )
Y = np.zeros( size * numPoints )
Z = np.zeros( size * numPoints )
GID = np.arange( numPoints*size, dtype=np.uint32 )

comm.Gather( sendbuf=x, recvbuf=X, root=0 )
comm.Gather( sendbuf=y, recvbuf=Y, root=0 )
comm.Gather( sendbuf=z, recvbuf=Z, root=0 )

if rank == 0:
    slice_data = [slice(i*numPoints, (i+1)*numPoints) for i in range(size)]
    plot_points(
        X, Y, Z, slice_data, title="Initial Distribution",
        filename="initial.pdf"
    )
# partition the points using PyZoltan
xa = DoubleArray(numPoints); xa.set_data(x)
ya = DoubleArray(numPoints); ya.set_data(y)
za = DoubleArray(numPoints); za.set_data(z)
gida = UIntArray(numPoints); gida.set_data(gid)

# create the geometric partitioner
pz = zoltan.ZoltanGeometricPartitioner(
    dim=3, comm=comm, x=xa, y=ya, z=za, gid=gida)

# call the load balancing function
pz.set_lb_method('RIB')
pz.Zoltan_Set_Param('DEBUG_LEVEL', '1')
pz.Zoltan_LB_Balance()

# get the new assignments
my_global_ids = list( gid )

# remove points to be exported
for i in range(pz.numExport):
    my_global_ids.remove( pz.exportGlobalids[i] )

# add points to be imported
for i in range(pz.numImport):
    my_global_ids.append( pz.importGlobalids[i] )

new_gids = np.array( my_global_ids, dtype=np.uint32 )

# gather the new gids on root as a list
NEW_GIDS = comm.gather( new_gids, root=0 )

# save the new partition
if rank == 0:
    plot_points(
        X, Y, Z, NEW_GIDS,
        title='Final Distribution', filename='final.pdf'
    )
comm.barrier()
