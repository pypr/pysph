"""Clone of the Zoltan tests for the geometric partitioners (RCB, RIB
and HSFC)"""

from __future__ import print_function

from os.path import abspath, dirname, join
import sys

import mpi4py.MPI as mpi
comm = mpi.COMM_WORLD

from pyzoltan.core.carray import UIntArray, DoubleArray
from pyzoltan.core import zoltan

MY_DIR = dirname(abspath(__file__))

ack_tag = 5
count_tag = 10
id_tag = 15
x_tag = 20
y_tag = 25
ack = 0

import numpy as np
from numpy import loadtxt

def read_input_file(fname=join(MY_DIR, 'mesh.txt')):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # read the data
        data_line = 0
        data = loadtxt( fname, dtype=int )
        numGlobalPoints = data[data_line][0]; data_line += 1

        if size > 1:
            nobj = numGlobalPoints // 2
            remaining = numGlobalPoints - nobj
        else:
            nobj = numGlobalPoints
            remaining = 0

        myGlobalIds = np.ones(shape=nobj, dtype=np.uint32)
        x = np.ones(shape=nobj, dtype=np.float64)
        y = np.ones(shape=nobj, dtype=np.float64)
        numMyPoints = nobj

        for i in range(nobj):
            myGlobalIds[i], x[i], y[i] = data[data_line]; data_line += 1

        gids = np.ones(shape=(nobj+1), dtype=np.uint32)
        xcoord = np.ones(shape=(nobj+1), dtype=np.float64)
        ycoord = np.ones(shape=(nobj+1), dtype=np.float64)

        for i in range(1,size):
            if (remaining > 1):
                nobj = remaining // 2
                remaining -= nobj

            elif (remaining == 1):
                nobj = 1
                remaining = 0

            else:
                nobj = 0

            if ( (i== size-1) and (remaining > 0) ):
                nobj += remaining

            if (nobj > 0):
                for j in range(nobj):
                    gids[j],xcoord[j],ycoord[j] = data[data_line];data_line += 1

            comm.send(obj=nobj, dest=i, tag=count_tag)
            ack = comm.recv(source=i, tag=ack_tag)

            if (nobj > 0):
                comm.send(gids[:nobj], dest=i, tag=id_tag)
                comm.send(xcoord[:nobj], dest=i, tag=x_tag)
                comm.send(ycoord[:nobj], dest=i, tag=y_tag)

        ack = 0
        for i in range(1, size):
            comm.send(ack, dest=i, tag=0)

    else:
        numMyPoints = comm.recv(source=0, tag=count_tag)
        ack = 0
        if (numMyPoints > 0):
            myGlobalIds = np.ones(shape=numMyPoints, dtype=np.uint32)
            x = np.ones(shape=numMyPoints, dtype=np.float64)
            y = np.ones(shape=numMyPoints, dtype=np.float64)

            comm.send(obj=ack, dest=0, tag=ack_tag)
            myGlobalIds[:] = comm.recv(source=0, tag=id_tag)
            x[:] = comm.recv(source=0, tag=x_tag)
            y[:] = comm.recv(source=0, tag=y_tag)

        elif ( numMyPoints == 0 ):
            comm.send(obj=ack, dest=0, tag=ack_tag)

        else:
            sys.exit(-1)

        ack = comm.recv( source=0, tag=0 )
        if ( ack < 0 ):
            sys.exit(-1)

    return numMyPoints, myGlobalIds, x, y

def show_mesh(rank, numPoints, gids, parts):

    partAssign = np.zeros(shape=25, dtype=np.int32)
    allPartAssign = np.zeros(shape=25, dtype=np.int32)
    i = j = part = 0

    for i in range(numPoints):
        partAssign[ gids[i]-1 ] = parts[i]

    comm.Reduce(sendbuf=partAssign,
                recvbuf=allPartAssign, op=mpi.MAX, root=0)

    if rank==0:

        for i in range(20, -1, -5):
            for j in range(5):
                part = allPartAssign[i+j]
                if j < 4:
                    print("%d----"%(part),)
                else:
                    print("%d"%(part))
            if i > 0:
                print("|     |     |     |     |")

        print()

# read the input file and distribute objects across processors
numMyPoints, myGlobalIds, x, y = read_input_file()
rank = comm.Get_rank()

parts = np.ones(shape=numMyPoints, dtype=np.int32)
for i in range(numMyPoints):
    parts[i] = rank

# now do the load balancing
_x = np.asarray(x); _y = np.asarray(y); _gid = np.asarray(myGlobalIds)
_z = np.zeros_like(x)

x = DoubleArray(numMyPoints); x.set_data(_x)
y = DoubleArray(numMyPoints); y.set_data(_y)
z = DoubleArray(numMyPoints); z.set_data(_z)
gid = UIntArray(numMyPoints); gid.set_data(_gid)

pz = zoltan.ZoltanGeometricPartitioner(dim=2, comm=comm, x=x, y=y, z=z, gid=gid)

# set the weights to 0 by default
weights = pz.weights.get_npy_array()
weights[:] = 0

pz.set_lb_method("RCB")
pz.Zoltan_Set_Param("DEBUG_LEVEL","0")
pz.Zoltan_LB_Balance()

if rank == 0:
    print("\nMesh partition before Zoltan\n")

comm.barrier()

show_mesh(rank, numMyPoints, myGlobalIds, parts)

for i in range(pz.numExport):
    parts[ pz.exportLocalids[i] ] = pz.exportProcs[i]

if rank == 0:
    print("Mesh partition assignment after calling Zoltan")

show_mesh(rank, numMyPoints, myGlobalIds, parts)
