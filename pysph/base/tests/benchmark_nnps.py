"""Benchmark numbers for NNPS"""
import numpy
from time import time
from numpy import random
import pandas as pd

from pyzoltan.core.carray import UIntArray

from pysph.base.utils import get_particle_array
from pysph.base.nnps import BoxSortNNPS, LinkedListNNPS

# number of points. Be warned, 1<<20 is about a million particles
# which can take a while to run. Hash out appropriately for your
# machine
_numPoints = [1<<15, 1<<16, 1<<17]#, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22]

def bench_nnps(particle_arrays):

    # time containers
    bs_update_times = []; bs_neighbor_times = []
    ll_update_times = []; ll_neighbor_times = []
    ll_neighbor_times_cell = []
    bs_neighbor_times_cell = []
    np = []

    for pa in particle_arrays:
        numPoints = pa.get_number_of_particles()
        np.append(numPoints)
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

        # BoxSort : Particle iteration
        nbrs = UIntArray(1000)
        t1 = time()
        for i in range(numPoints):
            nnps_boxs.get_nearest_particles(0, 0, i, nbrs)
        bs_neighbor_times.append( time() - t1 )

        # LinkedList : Particle iteration
        nbrs = UIntArray(1000)
        t1 = time()
        for i in range(numPoints):
            nnps_llist.get_nearest_particles(0, 0, i, nbrs)
        ll_neighbor_times.append( time() - t1 )

        # BoxSort : Cell iteration
        cells = nnps_boxs.cells
        cell_indices = UIntArray(1000)
        potential_nbrs = UIntArray(1000)
        nbrs = UIntArray(1000)

        t1 = time()
        for cell_index, cell in nnps_boxs.cells.iteritems():
            nnps_boxs.get_particles_in_cell(cell_index, 0, cell_indices)
            nnps_boxs.get_particles_in_neighboring_cells(cell_index, 0, potential_nbrs)

            # get the indices for each particle
            for particle_index in range( cell_indices.length ):
                nnps_boxs.get_nearest_particles_filtered(
                    0, 0, particle_index, potential_nbrs, nbrs)

        bs_neighbor_times_cell.append( time() - t1 )

        # LinkedList : Cell iteration

        ncells_tot = nnps_llist.ncells_tot
        cell_indices = UIntArray(1000)
        potential_nbrs = UIntArray(1000)
        nbrs = UIntArray(1000)

        t1 = time()
        for i in range(ncells_tot):
            nnps_llist.get_particles_in_cell(i, 0, cell_indices)     # indices in this cell
            nnps_llist.get_particles_in_neighboring_cells(i, 0, potential_nbrs) # potential neighbors

            # get the indices for each particle
            for particle_index in range( cell_indices.length ):
                nnps_llist.get_nearest_particles_filtered(
                    0, 0, particle_index, potential_nbrs, nbrs)

        ll_neighbor_times_cell.append( time() - t1 )

    data = dict(
        bs_update=bs_update_times,
        ll_update=ll_update_times,
        bs_nb=bs_neighbor_times,
        bs_cell_nb=bs_neighbor_times_cell,
        ll_nb=ll_neighbor_times,
        ll_cell_nb=ll_neighbor_times_cell
    )
    results = pd.DataFrame(data=data, index=np)
    return results


def bench_random_distribution():
    arrays = []
    for numPoints in _numPoints:
        dx = numpy.power( 1./numPoints, 1.0/3.0 )
        xa = random.random(numPoints)
        ya = random.random(numPoints)
        za = random.random(numPoints)
        ha = numpy.ones_like(xa) * 2*dx
        gida = numpy.arange(numPoints).astype(numpy.uint32)

        # create the particle array
        pa = get_particle_array(x=xa, y=ya, z=za, h=ha, gid=gida)
        arrays.append(pa)

    results = bench_nnps(arrays)
    print "###Timing Benchmarks for a Random Distribution###\n"
    print results

def bench_uniform_distribution():
    arrays = []
    for numPoints in _numPoints:
        nx = numpy.ceil(numpy.power(numPoints, 1./3))
        dx = 1./nx

        xa, ya, za = numpy.mgrid[0:1:dx, 0:1:dx, 0:1:dx]
        xa = xa.ravel(); ya = ya.ravel(); za = za.ravel()
        ha = numpy.ones_like(xa) * 2.0*dx

        gida = numpy.arange(xa.size).astype(numpy.uint32)

        # create the particle array
        pa = get_particle_array(x=xa, y=ya, z=za, h=ha, gid=gida)
        # create the particle array
        pa = get_particle_array(x=xa, y=ya, z=za, h=ha, gid=gida)
        arrays.append(pa)

    results = bench_nnps(arrays)
    print "###Timing Benchmarks for a Uniform Distribution###\n"
    print results


def main():
    bench_random_distribution()
    bench_uniform_distribution()

if __name__ == '__main__':
    main()
