"""Tests for the Zoltan unstructured communication package"""
import mpi4py.MPI as mpi
import numpy as np
from numpy import random

# import the unstructured communication package
from pyzoltan.core import zoltan_comm
from pyzoltan.core import zoltan

# MPI comm, rank and size
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize the Zoltan library
zoltan_version = zoltan.Zoltan_Initialize()

# each processor creates some random data
numObjectsTotal = 1<<10

x = random.random(numObjectsTotal)
gids = np.array( np.arange(size * numObjectsTotal) )[rank*numObjectsTotal:(rank+1)*numObjectsTotal]
gids = gids.astype(np.uint32)

# arbitrarily assign some objects to be sent to some other processor
nsend = np.int32( random.random_integers(low=1, high=5) )
object_ids = random.random_integers( low=0, high=numObjectsTotal, size=nsend )
proclist = random.random_integers(low=0, high=size-1, size=nsend).astype(np.int32)

my_indices = np.where(proclist == rank)[0]
proclist[my_indices] = (rank + 1)%size

# create the ZComm object
tag = np.int32(0)
zcomm = zoltan_comm.ZComm(comm, tag=tag, nsend=nsend, proclist=proclist)

# the data to send and receive
senddata = x[ object_ids ]
recvdata = np.ones( zcomm.nreturn )

# use zoltan to exchange doubles
print("Proc %d, Sending %s to %s"%(rank, senddata, proclist))
zcomm.Comm_Do(senddata, recvdata)
print("Proc %d, Received %s"%(rank, recvdata))

# use zoltan to exchange unsigned ints
senddata = gids[ object_ids ]
recvdata = np.ones(zcomm.nreturn, dtype=np.uint32)
zcomm.set_nbytes(recvdata.itemsize, recvdata.dtype)

print("Proc %d, Sending %s to %s"%(rank, senddata, proclist))
zcomm.Comm_Do(senddata, recvdata)
print("Proc %d, Received %s"%(rank, recvdata))

# Test the Comm Reverse function
# modify the received data
recvdata[:] = rank

updated_info = np.zeros(zcomm.nsend, dtype=senddata.dtype)
print('Proc %d, sending updated data %s'%(rank, recvdata))
zcomm.Comm_Do_Reverse(recvdata, updated_info)
print('Proc %d, received updated data %s'%(rank, updated_info))
