"""Zoltan example in parallel"""
import mpi4py.MPI as mpi

import numpy
from numpy import random
from numpy import savez, load

from pyzoltan.core.point import Point
from pyzoltan.core.carray import UIntArray
from pyzoltan.sph.kernels import CubicSpline
from pyzoltan.sph.nnps import NNPSParticleGeometric
from pyzoltan.sph.nnps_utils import get_particle_array

"""Utility to compute summation density"""
def sd_evaluate(nps, mass):
    pa = nps.pa
    x, y, h, rho= pa.get('x', 'y','h', 'rho')

    neighbors = UIntArray()
    cubic = CubicSpline(dim=2)

    for i in range(x.size):
        
        xi = Point( x[i], y[i], 0.0 )
        hi = h[i]

        nps.get_nearest_particles(i, neighbors)
        nnbrs = neighbors._length

        rho_sum = 0.0
        for indexj in range(nnbrs):
            j = neighbors[indexj]

            xj = Point(x[j], y[j], 0.0)
            wij = cubic.py_function(xi, xj, hi)

            rho_sum = rho_sum + mass *  wij

        rho[i] = rho_sum

# the total number of particles
numGlobalPoints = 1<<15

# Initialize MPI and find out number of local particles
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if numGlobalPoints % size != 0:
    raise RuntimeError("Run with 2^n num procs!")

numMyPoints = int(numGlobalPoints/size)
dx = numpy.sqrt( 1.0/numGlobalPoints )
mass = dx*dx

# only root creates the data, saves it and then everybody loads it
if rank == 0:
    _global_x = random.random( numGlobalPoints )
    _global_y = random.random( numGlobalPoints )
    savez( 'random.npz', x=_global_x, y=_global_y )

comm.barrier()
data = load('random.npz'); global_x = data['x']; global_y = data['y']

global_gid = numpy.array( range(numGlobalPoints), dtype=numpy.uint32 )
global_h = numpy.ones_like(global_x) * 1.2*dx

PA = get_particle_array(x=global_x, y=global_y, h=global_h, gid=global_gid)
NPS = NNPSParticleGeometric(dim=2, pa=PA, comm=comm)

if rank == 0:
    # compute the initial RHO
    sd_evaluate(NPS, mass)
    global_rho = PA.rho

# be nice to the root...
comm.barrier()    

# create the local particle arrays
pa = get_particle_array(x=global_x[rank*numMyPoints:(rank+1)*numMyPoints],
                        y=global_y[rank*numMyPoints:(rank+1)*numMyPoints],
                        h=global_h[rank*numMyPoints:(rank+1)*numMyPoints],
                        gid=global_gid[rank*numMyPoints:(rank+1)*numMyPoints])

# create the local nnps object
nps = NNPSParticleGeometric(dim=2, comm=comm, pa=pa)

# set the Zoltan parameters (Optional)
nps.set_lb_method("RIB")
nps.Zoltan_Set_Param("RETURN_LISTS", "ALL")
nps.Zoltan_Set_Param("DEBUG_LEVEL", "0")
nps.Zoltan_Set_Param("KEEP_CUTS", "1")

# CALL NNPS update
nps.update()

# new number of points per processor
numMyPoints = nps.num_particles

# Compute summation density individually on each processor
sd_evaluate(nps, mass)

# gather the density and global ids
rho = pa.get_carray('rho')
gid = pa.get_carray('gid')
x = pa.get_carray('x')
y = pa.get_carray('y')
_rho = comm.gather(rho.get_npy_array()[:numMyPoints], root=0)
_gid = comm.gather(gid.get_npy_array()[:numMyPoints], root=0)
_x = comm.gather(x.get_npy_array()[:numMyPoints], root=0)
_y = comm.gather(y.get_npy_array()[:numMyPoints], root=0)

if rank == 0:
    rho1 = numpy.concatenate( _rho )
    gid1 = numpy.concatenate( _gid )
    x1 = numpy.concatenate(_x)
    y1 = numpy.concatenate(_y)

    for index in range(numGlobalPoints):
        i = gid1[ index ]

        # make sure we're chacking the right point
        assert abs( x1[index]-global_x[i] ) < 1e-14
        assert abs( y1[index]-global_y[i] ) < 1e-14
        
        diff = abs( global_rho[i] - rho1[index] )
        condition = diff < 1e-14
        assert condition, "diff = %g"%(diff)

if rank == 0:
    print "Summation density test: OK"
