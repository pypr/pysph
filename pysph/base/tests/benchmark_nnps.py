"""Benchmark numbers for NNPS"""
import numpy
from time import time
from numpy import random

from pyzoltan.core.carray import UIntArray, DoubleArray

from pysph.base.point import IntPoint, Point
from pysph.base.utils import get_particle_array
from pysph.base.nnps import BoxSortNNPS, LinkedListNNPS

# number of points. Be warned, 1<<20 is about a million particles
# which can take a while to run. Hash out appropriately for your
# machine
_numPoints = [1<<15, 1<<16, 1<<17]#, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22]

# time containers
bs_update_times = []; bs_neighbor_times = []
ll_update_times = []; ll_neighbor_times = []
ll_update_times_cell = []; ll_neighbor_times_cell = []

for numPoints in _numPoints:
    dx = numpy.power( 1./numPoints, 1.0/3.0 )
    xa = random.random(numPoints)
    ya = random.random(numPoints)
    za = random.random(numPoints)
    ha = numpy.ones_like(xa) * 2*dx
    gida = numpy.arange(numPoints).astype(numpy.uint32)
    
    # create the particle array
    pa = get_particle_array(x=xa, y=ya, z=za, h=ha, gid=gida)

    ###### Update times #######
    nnps_boxs = BoxSortNNPS(
        dim=3, particles=[pa,], radius_scale=2.0, warn=False)

    t1 = time()
    nnps_boxs.update()
    bs_update_times.append(time() - t1)
    
    nnps_llist = LinkedListNNPS(
        dim=3, particles=[pa,], radius_scale=2.0, warn=False)

    t1 = time()
    nnps_llist.update()
    ll_update_times.append(time() - t1)

    ###### Neighbor look up times #######
    nbrs = UIntArray(1000)
    t1 = time()
    for i in range(numPoints):
        nnps_boxs.get_nearest_particles(0, 0, i, nbrs)
    bs_neighbor_times.append( time() - t1 )

    nbrs = UIntArray(1000)
    t1 = time()
    for i in range(numPoints):
        nnps_llist.get_nearest_particles(0, 0, i, nbrs)
    ll_neighbor_times.append( time() - t1 )

    ncells_tot = nnps_llist.ncells_tot
    cell_indices = UIntArray(1000)
    potential_nbrs = UIntArray(1000)
    nbrs = UIntArray(1000)
    t1 = time()
    for i in range(ncells_tot):
        nnps_llist.get_cell_indices(i, 0, cell_indices)     # indices in this cell
        nnps_llist.get_cell_neighbors(i, 0, potential_nbrs) # potential neighbors

        # get the indices for each particle
        for particle_index in range( cell_indices._length ):
            nnps_llist.get_nearest_particles_by_cell(
                0, 0, particle_index, potential_nbrs, nbrs)
        
    ll_neighbor_times_cell.append( time() - t1 )
        

print "###Timing Benchmarks for a Random Distribuion###\n"
print "\t\t update:"
print "Scheme\t  N_p\t time (s) \t time/particle"
for i, numPoints in enumerate(_numPoints):
    print "BSort\t %d\t %0.5g\t %0.5g"%(numPoints, bs_update_times[i], bs_update_times[i]/numPoints)
    print "LList\t %d\t %0.5g\t %0.5g\n"%(numPoints, ll_update_times[i], ll_update_times[i]/numPoints)

print "\n\t\t get_neighbors:"
print "Scheme\t  N_p\t time (s) \t time/particle"
for i, numPoints in enumerate(_numPoints):
    print "BSort\t %d\t %0.5g\t\t %0.5g"%(numPoints, bs_neighbor_times[i], bs_neighbor_times[i]/numPoints)
    print "LList\t %d\t %0.5g\t\t %0.5g"%(numPoints, ll_neighbor_times[i], ll_neighbor_times[i]/numPoints)
    print "CLList\t %d\t %0.5g\t\t %0.5g\n"%(numPoints, ll_neighbor_times_cell[i], ll_neighbor_times_cell[i]/numPoints)


# Do the same but for a uniform distribution of particles
bs_update_times = []; bs_neighbor_times = []
ll_update_times = []; ll_neighbor_times = []
ll_update_times_cell = []; ll_neighbor_times_cell = []
for numPoints in _numPoints:
    nx = numpy.ceil(numpy.power(numPoints, 1./3))
    dx = 1./nx

    xa, ya, za = numpy.mgrid[0:1:dx, 0:1:dx, 0:1:dx]
    xa = xa.ravel(); ya = ya.ravel(); za = za.ravel()
    ha = numpy.ones_like(xa) * 2.0*dx

    gida = numpy.arange(xa.size).astype(numpy.uint32)
    
    # create the particle array
    pa = get_particle_array(x=xa, y=ya, z=za, h=ha, gid=gida)

    ###### Update times #######
    nnps_boxs = BoxSortNNPS(
        dim=3, particles=[pa,], radius_scale=2.0, warn=False)

    t1 = time()
    nnps_boxs.update()
    bs_update_times.append(time() - t1)
    
    nnps_llist = LinkedListNNPS(
        dim=3, particles=[pa,], radius_scale=2.0, warn=False)

    t1 = time()
    nnps_llist.update()
    ll_update_times.append(time() - t1)

    ###### Neighbor look up times #######
    nbrs = UIntArray(1000)
    t1 = time()
    for i in range(numPoints):
        nnps_boxs.get_nearest_particles(0, 0, i, nbrs)
    bs_neighbor_times.append( time() - t1 )

    nbrs = UIntArray(1000)
    t1 = time()
    for i in range(numPoints):
        nnps_llist.get_nearest_particles(0, 0, i, nbrs)
    ll_neighbor_times.append( time() - t1 )

    ncells_tot = nnps_llist.ncells_tot
    cell_indices = UIntArray(1000)
    potential_nbrs = UIntArray(1000)
    nbrs = UIntArray(1000)
    t1 = time()
    for i in range(ncells_tot):
        nnps_llist.get_cell_indices(i, 0, cell_indices)     # indices in this cell
        nnps_llist.get_cell_neighbors(i, 0, potential_nbrs) # potential neighbors

        # get the indices for each particle
        for particle_index in range( cell_indices._length ):
            nnps_llist.get_nearest_particles_by_cell(
                0, 0, particle_index, potential_nbrs, nbrs)
        
    ll_neighbor_times_cell.append( time() - t1 )
        

print "\n\n###Timing Benchmarks for a Uniform Distribuion###\n"
print "\t\t update:"
print "Scheme\t  N_p\t time (s) \t time/particle"
for i, numPoints in enumerate(_numPoints):
    print "BSort\t %d\t %0.5g\t %0.5g"%(numPoints, bs_update_times[i], bs_update_times[i]/numPoints)
    print "LList\t %d\t %0.5g\t %0.5g\n"%(numPoints, ll_update_times[i], ll_update_times[i]/numPoints)

print "\n\t\t get_neighbors:"
print "Scheme\t  N_p\t time (s) \t time/particle"
for i, numPoints in enumerate(_numPoints):
    print "BSort\t %d\t %0.5g\t\t %0.5g"%(numPoints, bs_neighbor_times[i], bs_neighbor_times[i]/numPoints)
    print "LList\t %d\t %0.5g\t\t %0.5g"%(numPoints, ll_neighbor_times[i], ll_neighbor_times[i]/numPoints)
    print "CLList\t %d\t %0.5g\t\t %0.5g\n"%(numPoints, ll_neighbor_times_cell[i], ll_neighbor_times_cell[i]/numPoints)
