"""Example for the Zoltan Distributed data directory"""
import mpi4py.MPI as mpi

from mpi4py.mpi_c cimport MPI_Comm
from czoltan_dd cimport *

_comm = mpi.COMM_WORLD
rank = _comm.Get_rank()
size = _comm.Get_size()

# pointer to Zoltan_DD
cdef Zoltan_DD_Directory* dd

# create a distributed data directory
cdef int ierr
cdef MPI_Comm comm = _comm.ob_mpi

ierr = Zoltan_DD_Create( &d, comm, 1, 1, 0, 0, 1)
