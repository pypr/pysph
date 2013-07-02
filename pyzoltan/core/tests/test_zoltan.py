"""Simple tests for the Zoltan wrapper"""
import mpi4py.MPI as mpi
from pyzoltan.core import zoltan

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def test_PyZoltan():
    """Basic tests for the PyZoltan wrapper"""
    pz = zoltan.PyZoltan( comm=comm )

    # test rank and size
    assert( pz.rank == rank )
    assert( pz.size == size )

    # test the zoltan arrays 
    assert( pz.exportLocalids.length == 0 )
    assert( pz.exportGlobalids.length == 0 )
    assert( pz.exportProcs.length == 0 )

    assert( pz.importLocalids.length == 0 )
    assert( pz.importGlobalids.length == 0 )
    assert( pz.importProcs.length == 0 )

    assert( pz.procs.size == size )
    assert( pz.parts.size == size )

if __name__ == '__main__':
    test_PyZoltan()

    if rank == 0:
        print "OK"
