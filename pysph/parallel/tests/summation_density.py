"""Summation density example using the cell based NNPS partitioner."""

import mpi4py.MPI as mpi

import numpy
from numpy import random

# Carray from PyZoltan
from pyzoltan.core.carray import UIntArray

# PySPH imports
from pysph.base.nnps import BoxSortNNPS
from pysph.parallel.parallel_manager import ZoltanParallelManagerGeometric
from pysph.base.utils import get_particle_array_wcsph

from pysph.base.kernels import CubicSpline, get_compiled_kernel

"""Utility to compute summation density"""
def sd_evaluate(nnps, pm, mass, src_index, dst_index):
    # the destination particle array
    dst = nnps.particles[ dst_index ]
    src = nnps.particles[ src_index ]

    # particle coordinates
    dx, dy, dz, dh, drho = dst.get('x', 'y', 'z', 'h', 'rho', only_real_particles=True)
    sx, sy, sz, sh, srho = src.get('x', 'y', 'z', 'h', 'rho', only_real_particles=False)

    neighbors = UIntArray()
    cubic = get_compiled_kernel(CubicSpline(dim=dim))

    # compute density for each destination particle
    num_particles = dst.num_real_particles

    # the number of local particles should have tag Local
    assert( num_particles == pm.num_local[dst_index] )

    for i in range(num_particles):

        hi = dh[i]

        nnps.get_nearest_particles(
            src_index, dst_index, i, neighbors)

        nnbrs = neighbors.length

        rho_sum = 0.0
        for indexj in range(nnbrs):
            j = neighbors[indexj]

            wij = cubic.kernel(dx[i], dy[i], dz[i], sx[j], sy[j], sz[j], hi)

            rho_sum = rho_sum + mass *  wij

        drho[i] += rho_sum

# Initialize MPI and find out number of local particles
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# number of particles per array
numMyPoints = 1<<10
numGlobalPoints = size * numMyPoints

dim=3
avg_vol = 1.0/numGlobalPoints
dx = numpy.power( avg_vol, 1.0/dim )
mass = avg_vol
hdx = 1.3

if numGlobalPoints % size != 0:
    raise RuntimeError("Run with 2^n num procs!")

# everybody creates two particle arrays with numMyPoints
x1 = random.random( numMyPoints ); y1 = random.random( numMyPoints )
z1 = random.random( numMyPoints ); h1 = numpy.ones_like(x1) * hdx * dx
rho1 = numpy.zeros_like(x1)

x2 = random.random( numMyPoints ); y2 = random.random( numMyPoints )
z2 = random.random( numMyPoints ); h2 = numpy.ones_like(x2) * hdx * dx
rho2 = numpy.zeros_like(x2)

#z1[:] = 1.0
#z2[:] = 0.5

# local particle arrays
pa1 = get_particle_array_wcsph(x=x1, y=y1, h=h1, rho=rho1, z=z1)
pa2 = get_particle_array_wcsph(x=x2, y=y2, h=h2, rho=rho2, z=z2)

# gather the data on root
X1 = numpy.zeros( numGlobalPoints ); Y1 = numpy.zeros( numGlobalPoints )
Z1 = numpy.zeros( numGlobalPoints ); H1 = numpy.ones_like(X1) * hdx * dx
RHO1 = numpy.zeros_like(X1)

gathers = (numpy.ones(size)*numMyPoints, None)

comm.Gatherv( sendbuf=[x1, mpi.DOUBLE], recvbuf=[X1, gathers, mpi.DOUBLE] )
comm.Gatherv( sendbuf=[y1, mpi.DOUBLE], recvbuf=[Y1, gathers, mpi.DOUBLE] )
comm.Gatherv( sendbuf=[z1, mpi.DOUBLE], recvbuf=[Z1, gathers, mpi.DOUBLE] )
comm.Gatherv( sendbuf=[rho1, mpi.DOUBLE], recvbuf=[RHO1, gathers, mpi.DOUBLE])

X2 = numpy.zeros( numGlobalPoints ); Y2 = numpy.zeros( numGlobalPoints )
Z2 = numpy.zeros( numGlobalPoints ); H2 = numpy.ones_like(X2) * hdx * dx
RHO2 = numpy.zeros_like(X2)

comm.Gatherv( sendbuf=[x2, mpi.DOUBLE], recvbuf=[X2, gathers, mpi.DOUBLE])
comm.Gatherv( sendbuf=[y2, mpi.DOUBLE], recvbuf=[Y2, gathers, mpi.DOUBLE])
comm.Gatherv( sendbuf=[z2, mpi.DOUBLE], recvbuf=[Z2, gathers, mpi.DOUBLE])
comm.Gatherv( sendbuf=[rho2, mpi.DOUBLE], recvbuf=[RHO2, gathers, mpi.DOUBLE])

# create the particle arrays and PM
PA1 = get_particle_array_wcsph(x=X1, y=Y1, z=Z1, h=H1, rho=RHO1)
PA2 = get_particle_array_wcsph(x=X2, y=Y2, z=Z2, h=H2, rho=RHO2)

# create the parallel manager
PARTICLES = [PA1, PA2]
PM = ZoltanParallelManagerGeometric(dim=dim, particles=PARTICLES, comm=comm)

# create the local NNPS object with all the particles
Nnps = BoxSortNNPS(dim=dim, particles=PARTICLES)
Nnps.update()

# only root computes summation density
if rank == 0:
    assert numpy.allclose(PA1.rho, 0)
    sd_evaluate(Nnps, PM, mass, src_index=1, dst_index=0)
    sd_evaluate(Nnps, PM, mass, src_index=0, dst_index=0)
    RHO1 = PA1.rho

    assert numpy.allclose(PA2.rho, 0)
    sd_evaluate(Nnps, PM, mass, src_index=0, dst_index=1)
    sd_evaluate(Nnps, PM, mass, src_index=1, dst_index=1)
    RHO2 = PA2.rho

# wait for the root...
comm.barrier()

# create the local particle arrays
particles = [pa1, pa2]

# create the local nnps object and parallel manager
pm = ZoltanParallelManagerGeometric(dim=dim, comm=comm, particles=particles)
nnps = BoxSortNNPS(dim=dim, particles=particles)

# set the Zoltan parameters (Optional)
pz = pm.pz
pz.set_lb_method("RCB")
pz.Zoltan_Set_Param("DEBUG_LEVEL", "0")

# Update the parallel manager (distribute particles)
pm.update()

# update the local nnps
nnps.update()

# Compute summation density individually on each processor
sd_evaluate(nnps, pm, mass, src_index=0, dst_index=1)
sd_evaluate(nnps, pm, mass, src_index=1, dst_index=1)

sd_evaluate(nnps, pm, mass, src_index=0, dst_index=0)
sd_evaluate(nnps, pm, mass, src_index=1, dst_index=0)

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

z1 = pa1.z; tmp = comm.gather( z1 )
if rank == 0 :
    global_z1 = numpy.concatenate( tmp )
    assert( global_z1.size == numGlobalPoints )

# gather global x2 and y2
x2 = pa2.x; tmp = comm.gather( x2 )
if rank == 0 :
    global_x2 = numpy.concatenate( tmp )
    assert( global_x2.size == numGlobalPoints )

y2 = pa2.y; tmp = comm.gather( y2 )
if rank == 0 :
    global_y2 = numpy.concatenate( tmp )
    assert( global_y2.size == numGlobalPoints )

z2 = pa2.z; tmp = comm.gather( z2 )
if rank == 0 :
    global_z2 = numpy.concatenate( tmp )
    assert( global_z2.size == numGlobalPoints )

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
    assert( global_z1.size == Z1.size )

    for i in range(numGlobalPoints):

        # make sure we're chacking the right point
        assert abs( global_x1[i] - X1[global_gid1[i]] ) < 1e-14
        assert abs( global_y1[i] - Y1[global_gid1[i]] ) < 1e-14
        assert abs( global_z1[i] - Z1[global_gid1[i]] ) < 1e-14

        diff = abs( global_rho1[i] - RHO1[global_gid1[i]] )
        condition = diff < 1e-14
        assert condition, "diff = %g"%(diff)

# check rho2
if rank == 0:
    # make sure the arrays are of the same size
    assert( global_x2.size == X2.size )
    assert( global_y2.size == Y2.size )
    assert( global_z2.size == Z2.size )

    for i in range(numGlobalPoints):

        # make sure we're chacking the right point
        assert abs( global_x2[i] - X2[global_gid2[i]] ) < 1e-14
        assert abs( global_y2[i] - Y2[global_gid2[i]] ) < 1e-14
        assert abs( global_z2[i] - Z2[global_gid2[i]] ) < 1e-14

        diff = abs( global_rho2[i] - RHO2[global_gid2[i]] )
        condition = diff < 1e-14
        assert condition, "diff = %g"%(diff)

if rank == 0:
    print("Summation density test: OK")
