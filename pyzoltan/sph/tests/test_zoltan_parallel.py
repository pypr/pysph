"""Zoltan example in parallel"""
from mpi4py import MPI as mpi

import numpy
from numpy import random

from pyzoltan.sph.utils import get_particle_array
from pyzoltan.core.point import Point
from pyzoltan.core.carray import UIntArray
from pyzoltan.sph.kernels import CubicSpline
from pyzoltan.sph.nnps import NNPS

"""Utility to compute summation density"""
def sd_evaluate(nps, mass):
    pa = nps.pa
    x, y, h, rho= pa.get('x', 'y','h', 'rho')

    neighbors = UIntArray(100)
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

# from pylab import subplot, savefig, close
colors = ['Aqua', 'Crimson', 'DarkOliveGreen', 'DarkOrange',
          'Fuchsia', 'Indigo', 'Yellow', 'LimeGreen']

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

# create the local points
xa = random.random(numMyPoints).astype(numpy.float64)
ya = random.random(numMyPoints).astype(numpy.float64)
gida = numpy.array( range(numGlobalPoints) ).astype(numpy.uint32)[rank*numMyPoints:(rank+1)*numMyPoints]
ha = numpy.ones_like(xa) * 1.2*dx

# create the particle array
pa = get_particle_array(x=xa, y=ya, h=ha, gid=gida)

# Save the initial data partition
_x = comm.gather(xa, root=0)
_y = comm.gather(ya, root=0)
_h = comm.gather(ha, root=0)
if rank == 0:
    # compute the initial RHO
    _X = numpy.concatenate(_x); _Y = numpy.concatenate(_y)
    _H = numpy.concatenate(_h)
    _GID = numpy.arange( numGlobalPoints ).astype(numpy.uint32)

    PA = get_particle_array(x=_X, y=_Y, h=_H, gid=_GID)
    NPS = NNPS(2, PA, radius_scale=1.2, comm=None)
    sd_evaluate(NPS, mass)
    rho0 = PA.rho

# with a valid MPI comm object, Zoltan is initialized and the
# Zoltan_Struct is created and stored with nnps
nps = NNPS(dim=2, pa=pa, radius_scale=1.2, comm=comm)    

# set the Zoltan parameters
nps.set_lb_method("RCB")
nps.Zoltan_Set_Param("RETURN_LISTS", "ALL")
nps.Zoltan_Set_Param("DEBUG_LEVEL", "0")
nps.Zoltan_Set_Param("KEEP_CUTS", "1")

# register Zoltan query functions
nps.Zoltan_Set_Num_Obj_Fn()
nps.Zoltan_Set_Obj_List_Fn()
nps.Zoltan_Set_Num_Geom_Fn()
nps.Zoltan_Set_Geom_Multi_Fn()

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

        assert abs( x1[index]-_X[i] ) < 1e-14
        assert abs( y1[index]-_Y[i] ) < 1e-14
        
        diff = abs( rho0[ gid1[index] ] - rho1[index] )
        condition = diff < 1e-14
        assert condition, "diff = %g"%(diff)

# Destroy the Zoltan struct
nps.Zoltan_Destroy()

if rank == 0:
    print "Summation density test: OK"
