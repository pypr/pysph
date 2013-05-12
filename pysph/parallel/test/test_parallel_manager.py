"""Summation density example using the cell based NNPS partitioner."""

import mpi4py.MPI as mpi

import numpy
from numpy import random
from numpy import savez, load

# Cython kernel function from PyZoltan
from pyzoltan.sph.kernels import CubicSpline
from pyzoltan.core.carray import UIntArray
from pyzoltan.sph.point import Point

# PySPH imports
from pysph.base.nnps import NNPS
from pysph.parallel.parallel_manager import ZoltanParallelManagerGeometric
from pysph.base.utils import get_particle_array_wcsph

"""Utility to compute summation density"""
def sd_evaluate(pm, mass, src_index, dst_index):
    # the destination particle array
    dst = pm.particles[ dst_index ]
    src = pm.particles[ src_index ]

    # particle coordinates
    dx, dy, dh, drho = dst.get('x', 'y','h', 'rho', only_real_particles=True)
    sx, sy, sh, srho = src.get('x', 'y','h', 'rho', only_real_particles=False)

    neighbors = UIntArray()
    cubic = CubicSpline(dim=2)

    # compute density for each destination particle
    num_particles = pm.num_local[dst_index]

    # the number of local particles should have tag Local
    assert( num_particles == dx.size )
    
    for i in range(num_particles):
        
        xi = Point( dx[i], dy[i], 0.0 )
        hi = dh[i]

        pm.get_nearest_particles(src_index=src_index, dst_index=dst_index, i=i,
                                  nbrs=neighbors)
        nnbrs = neighbors._length

        rho_sum = 0.0
        for indexj in range(nnbrs):
            j = neighbors[indexj]

            xj = Point(sx[j], sy[j], 0.0)
            wij = cubic.py_function(xi, xj, hi)

            rho_sum = rho_sum + mass *  wij

        drho[i] = rho_sum

# Initialize MPI and find out number of local particles
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# number of particles per array
numMyPoints = 1<<12
numGlobalPoints = size * numMyPoints

dx = numpy.sqrt( 1.0/numGlobalPoints )
mass = dx*dx
hdx = 1.3

if numGlobalPoints % size != 0:
    raise RuntimeError("Run with 2^n num procs!")

# everybody creates two particle arrays with numMyPoints
x1 = random.random( numMyPoints ); y1 = random.random( numMyPoints )
h1 = numpy.ones_like(x1) * hdx * dx

x2 = random.random( numMyPoints ); y2 = random.random( numMyPoints )
h2 = numpy.ones_like(x2) * hdx * dx

# local particle arrays
pa1 = get_particle_array_wcsph(x=x1, y=y1, h=h1)
pa2 = get_particle_array_wcsph(x=x2, y=y2, h=h2)

# gather the data on root
X1 = numpy.zeros( numGlobalPoints ); Y1 = numpy.zeros( numGlobalPoints )
H1 = numpy.ones_like(X1) * hdx * dx
comm.Gatherv( sendbuf=x1, recvbuf=X1 )
comm.Gatherv( sendbuf=y1, recvbuf=Y1 )

X2 = numpy.zeros( numGlobalPoints ); Y2 = numpy.zeros( numGlobalPoints )
H2 = numpy.ones_like(X2) * hdx * dx
comm.Gatherv( sendbuf=x2, recvbuf=X2 )
comm.Gatherv( sendbuf=y2, recvbuf=Y2 )

# create the particle arrays and PM
PA1 = get_particle_array_wcsph(x=X1, y=Y1, h=H1)
PA2 = get_particle_array_wcsph(x=X2, y=Y2, h=H2)
PM = ZoltanParallelManagerGeometric(dim=2, particles=[PA1,PA2], comm=comm)

# only root computes summation density
if rank == 0:
    sd_evaluate(PM, mass, src_index=1, dst_index=0)
    RHO1 = PA1.rho

    sd_evaluate(PM, mass, src_index=0, dst_index=1)
    RHO2 = PA2.rho

# be nice to the root...
comm.barrier()    

# create the local particle arrays
particles = [pa1, pa2]

# create the local nnps object
pm = ZoltanParallelManagerGeometric(dim=2, comm=comm, particles=particles)

# set the Zoltan parameters (Optional)
pz = pm.pz
pz.set_lb_method("RCB")
pz.Zoltan_Set_Param("DEBUG_LEVEL", "0")

# CALL NNPS update
pm.update()

# Compute summation density individually on each processor
sd_evaluate(pm, mass, src_index=0, dst_index=1)
sd_evaluate(pm, mass, src_index=1, dst_index=0)

# gather the density and global ids
rho1 = pa1.rho; tmp = comm.gather(rho1)
if rank == 0:
    global_rho1 = numpy.concatenate( tmp )
    assert( global_rho1.size == numGlobalPoints )

rho2 = pa2.rho; tmp = comm.gather(rho2)
if rank == 0:
    global_rho2 = numpy.concatenate( tmp )
    assert( global_rho2.size == numGlobalPoints )

# gather global x1 and y1
x1 = pa1.x; tmp = comm.gather( x1 )
if rank == 0 :
    global_x1 = numpy.concatenate( tmp )
    assert( global_x1.size == numGlobalPoints )

y1 = pa1.y; tmp = comm.gather( y1 )
if rank == 0 :
    global_y1 = numpy.concatenate( tmp )
    assert( global_y1.size == numGlobalPoints )

# gather global x2 and y2
x2 = pa2.x; tmp = comm.gather( x2 )
if rank == 0 :
    global_x2 = numpy.concatenate( tmp )
    assert( global_x2.size == numGlobalPoints )

y2 = pa2.y; tmp = comm.gather( y2 )
if rank == 0 :
    global_y2 = numpy.concatenate( tmp )
    assert( global_y2.size == numGlobalPoints )

# gather global indices
gid1 = pa1.gid; tmp = comm.gather( gid1 )
if rank == 0 :
    global_gid1 = numpy.concatenate( tmp )
    assert ( global_gid1.size == numGlobalPoints )

gid2 = pa2.gid; tmp = comm.gather( gid2 )
if rank == 0 :
    global_gid2 = numpy.concatenate( tmp )
    assert( global_gid2.size == numGlobalPoints )

# check rho1
if rank == 0:
    # make sure the arrays are of the same size
    assert( global_x1.size == X1.size )
    assert( global_y1.size == Y1.size )

    for i in range(numGlobalPoints):

        # make sure we're chacking the right point
        assert abs( global_x1[i] - X1[global_gid1[i]] ) < 1e-14
        assert abs( global_y1[i] - Y1[global_gid1[i]] ) < 1e-14

        diff = abs( global_rho1[i] - RHO1[global_gid1[i]] )
        condition = diff < 1e-14
        assert condition, "diff = %g"%(diff)

# check rho2
if rank == 0:
    # make sure the arrays are of the same size
    assert( global_x2.size == X2.size )
    assert( global_y2.size == Y2.size )

    for i in range(numGlobalPoints):

        # make sure we're chacking the right point
        assert abs( global_x2[i] - X2[global_gid2[i]] ) < 1e-14
        assert abs( global_y2[i] - Y2[global_gid2[i]] ) < 1e-14

        diff = abs( global_rho2[i] - RHO2[global_gid2[i]] )
        condition = diff < 1e-14
        assert condition, "diff = %g"%(diff)

if rank == 0:
    print "Summation density test: OK"
