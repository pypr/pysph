#cython: embedsignature=True
# Library imports.
import numpy as np
cimport numpy as np

# Cython imports
from cython.operator cimport dereference as deref, preincrement as inc
from cython.parallel import parallel, prange, threadid

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)
    void sort[Iter](Iter first, Iter last)

# cpython
from cpython.dict cimport PyDict_Clear, PyDict_Contains, PyDict_GetItem
from cpython.list cimport PyList_GetItem, PyList_SetItem, PyList_GET_ITEM

# Cython for compiler directives
cimport cython

import pyopencl as cl
import pyopencl.array
from pyopencl.scan import ExclusiveScanKernel
from pyopencl.elementwise import ElementwiseKernel

from pysph.base.nnps_base cimport *

# Particle Tag information
from pyzoltan.core.carray cimport BaseArray, aligned_malloc, aligned_free
from utils import ParticleTAGS

from nnps_base cimport *


cdef class GPUNeighborCache:
    def __init__(self, GPUNNPS nnps, int dst_index, int src_index):
        self._dst_index = dst_index
        self._src_index = src_index
        self._nnps = nnps
        self._particles = nnps.particles
        self._narrays = nnps.narrays

        self._dst_index = dst_index
        self._src_index = src_index
        self._nnps = nnps
        self._particles = nnps.particles
        self._narrays = nnps.narrays
        cdef long n_p = self._particles[dst_index].get_number_of_particles()
        cdef size_t i

        self._cached = False
        self._copied_to_cpu = False

        self._nbr_lengths_gpu = cl.array.zeros(self._nnps.queue,
                (n_p,), dtype=np.uint32)

    #### Public protocol ################################################

    cdef void get_neighbors_raw_gpu(self):
        if not self._cached:
            self._find_neighbors()

    cdef void get_neighbors_raw(self, size_t d_idx, UIntArray nbrs):
        self.get_neighbors_raw_gpu()
        if not self._copied_to_cpu:
            self.copy_to_cpu()
        nbrs.c_reset()
        nbrs.c_set_view(self._neighbors_cpu_ptr + self._start_idx_ptr[d_idx],
                self._nbr_lengths_ptr[d_idx])

    #### Private protocol ################################################

    cdef void _find_neighbors(self):
        self._nnps.find_neighbor_lengths(self._nbr_lengths_gpu)
        total_size_gpu = cl.array.sum(self._nbr_lengths_gpu)
        cdef unsigned long total_size = <unsigned long>(total_size_gpu.get())

        # Allocate _neighbors_cpu and neighbors_gpu
        self._neighbors_gpu = cl.array.empty(self._nnps.queue, (total_size,),
                dtype=np.uint32)

        self._start_idx_gpu = self._nbr_lengths_gpu.copy()

        # Do prefix sum on self._neighbor_lengths for the self._start_idx
        get_start_indices = ExclusiveScanKernel(self._nnps.ctx,
                np.uint32, scan_expr="a+b", neutral="0")

        get_start_indices(self._start_idx_gpu)
        self._nnps.find_nearest_neighbors_gpu(self._neighbors_gpu,
                self._start_idx_gpu)
        self._cached = True

    cdef void copy_to_cpu(self):
        self._copied_to_cpu = True
        self._neighbors_cpu = self._neighbors_gpu.get()
        self._neighbors_cpu_ptr = <unsigned int*> self._neighbors_cpu.data
        self._nbr_lengths = self._nbr_lengths_gpu.get()
        self._nbr_lengths_ptr = <unsigned int*> self._nbr_lengths.data
        self._start_idx = self._start_idx_gpu.get()
        self._start_idx_ptr = <unsigned int*> self._start_idx.data

    cpdef update(self):
        self._cached = False
        self._copied_to_cpu = False
        cdef long n_p = self._particles[self._dst_index].get_number_of_particles()
        self._nbr_lengths_gpu = cl.array.zeros(self._nnps.queue, n_p, dtype=np.uint32)
        self._start_idx_gpu = cl.array.empty(self._nnps.queue, n_p, dtype=np.uint32)

    cpdef get_neighbors(self, int src_index, size_t d_idx, UIntArray nbrs):
        self.get_neighbors_raw(d_idx, nbrs)

    cpdef get_neighbors_gpu(self):
        self.get_neighbors_raw_gpu()

cdef class GPUNNPS(NNPSBase):
    """Nearest neighbor query class using the box-sort algorithm.

    NNPS bins all local particles using the box sort algorithm in
    Cells. The cells are stored in a dictionary 'cells' which is keyed
    on the spatial index (IntPoint) of the cell.

    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, bint cache=True,
                 bint sort_gids=False, ctx=None):
        """Constructor for NNPS

        Parameters
        ----------

        dim : int
            Dimension (fixme: Not sure if this is really needed)

        particles : list
            The list of particle arrays we are working on.

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        domain : DomainManager, default (None)
            Optional limits for the domain

        cache : bint
            Flag to set if we want to cache neighbor calls. This costs
            storage but speeds up neighbor calculations.

        sort_gids : bint, default (False)
            Flag to sort neighbors using gids (if they are available).
            This is useful when comparing parallel results with those
            from a serial run.

        ctx : pyopencl.Context
            For testing purpose
        """
        NNPSBase.__init__(self, dim, particles, radius_scale, ghost_layers,
                domain, cache, sort_gids)

        if ctx is None:
            self.ctx = cl.create_some_context()
        else:
            self.ctx = ctx
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # The cache.
        self.use_cache = cache
        _cache = []
        for d_idx in range(len(particles)):
            for s_idx in range(len(particles)):
                _cache.append(GPUNeighborCache(self, d_idx, s_idx))
        self.cache = _cache

    cdef void get_nearest_neighbors(self, size_t d_idx, UIntArray nbrs):
        if self.use_cache:
            self.current_cache.get_neighbors_raw(d_idx, nbrs)
        else:
            nbrs.c_reset()
            self.find_nearest_neighbors(d_idx, nbrs)

    cpdef get_nearest_particles_gpu(self, int src_index, int dst_index):
        cdef int idx = dst_index*self.narrays + src_index
        if self.src_index != src_index \
            or self.dst_index != dst_index:
            self.set_context(src_index, dst_index)
        self.cache[idx].get_neighbors_gpu()

    cpdef spatially_order_particles(self, int pa_index):
        """Spatially order particles such that nearby particles have indices
        nearer each other.  This may improve pre-fetching on the CPU.
        """
        cdef LongArray indices = LongArray()
        cdef ParticleArray pa = self.pa_wrappers[pa_index].pa
        self.get_spatially_ordered_indices(pa_index, indices)
        cdef BaseArray arr

        for arr in pa.properties.values():
            arr.c_align_array(indices)

        copy_to_gpu(self.pa_wrappers[pa_index], self.queue,
                (np.float64 if self.use_double else np.float32))

    cdef void find_neighbor_lengths(self, nbr_lengths):
        raise NotImplementedError("NNPS :: find_neighbor_lengths called")

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        raise NotImplementedError("NNPS :: find_nearest_neighbors called")

    cpdef update(self):
        cdef int i, num_particles
        cdef ParticleArray pa

        cdef DomainManager domain = self.domain

        # use cell sizes computed by the domain.
        self.cell_size = domain.cell_size
        self.hmin = domain.hmin

        # compute bounds and refresh the data structure
        self._compute_bounds()
        self._refresh()

        # indices on which to bin. We bin all local particles
        for i in range(self.narrays):
            pa = self.particles[i]
            num_particles = pa.get_number_of_particles()
            indices = arange_uint(num_particles)

            # bin the particles
            self._bin(pa_index=i)

        if self.use_cache:
            for cache in self.cache:
                cache.update()

    cpdef _bin(self, int pa_index):
        raise NotImplementedError("NNPS :: _bin called")

    cpdef _refresh(self):
        raise NotImplementedError("NNPS :: _refresh called")

cdef class BruteForceNNPS(GPUNNPS):
    def __init__(self, int dim, list particles, double radius_scale=2.0,
            int ghost_layers=1, domain=None, bint cache=True,
            bint sort_gids=False, bint use_double=True):
        GPUNNPS.__init__(self, dim, particles, radius_scale, ghost_layers,
                domain, cache, sort_gids)

        self.use_double = use_double
        self.radius_scale2 = radius_scale*radius_scale
        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

        cdef str norm2 = \
                """
                #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
                """

        self.preamble = norm2

        cdef NNPSParticleArrayWrapper pa_wrapper
        for pa_wrapper in self.pa_wrappers:
            if use_double:
                pa_wrapper.copy_to_gpu(self.queue, np.float64)
            else:
                pa_wrapper.copy_to_gpu(self.queue, np.float32)

    cpdef set_context(self, int src_index, int dst_index):
        """Setup the context before asking for neighbors.  The `dst_index`
        represents the particles for whom the neighbors are to be determined
        from the particle array with index `src_index`.

        Parameters
        ----------

         src_index: int: the source index of the particle array.
         dst_index: int: the destination index of the particle array.
        """
        GPUNNPS.set_context(self, src_index, dst_index)

        self.src = self.pa_wrappers[ src_index ]
        self.dst = self.pa_wrappers[ dst_index ]

    cdef void find_neighbor_lengths(self, nbr_lengths):
        arguments = \
                """%(data_t)s* s_x, %(data_t)s* s_y, %(data_t)s* s_z,
                %(data_t)s* s_h, %(data_t)s* d_x, %(data_t)s* d_y,
                %(data_t)s* d_z, %(data_t)s* d_h, unsigned int num_particles,
                unsigned int* nbr_lengths, %(data_t)s radius_scale2
                """ % {"data_t" : ("double" if self.use_double else "float")}

        src = """
                unsigned int j;
                %(data_t)s dist;
                %(data_t)s h_i = radius_scale2*d_h[i]*d_h[i];
                %(data_t)s h_j;
                for(j=0; j<num_particles; j++)
                {
                    h_j = radius_scale2*s_h[j]*s_h[j];
                    dist = NORM2(d_x[i] - s_x[j], d_y[i] - s_y[j], d_z[i] - s_z[j]);
                    if(dist < h_i || dist < h_j)
                        nbr_lengths[i] += 1;
                }
                """ % {"data_t" : ("double" if self.use_double else "float")}

        brute_force_nbr_lengths = ElementwiseKernel(self.ctx,
                arguments, src, "brute_force_nbr_lengths", preamble=norm2)

        brute_force_nbr_lengths(self.src.gpu_x, self.src.gpu_y, self.src.gpu_z,
                self.src.gpu_h, self.dst.gpu_x, self.dst.gpu_y, self.dst.gpu_z,
                self.dst.gpu_h, self.src.get_number_of_particles(),
                nbr_lengths, self.radius_scale2)

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        arguments = \
                """%(data_t)s* s_x, %(data_t)s* s_y, %(data_t)s* s_z,
                %(data_t)s* s_h, %(data_t)s* d_x, %(data_t)s* d_y,
                %(data_t)s* d_z, %(data_t)s* d_h, unsigned int num_particles,
                unsigned int* start_indices, unsigned int* nbrs,
                %(data_t)s radius_scale2
                """ % {"data_t" : ("double" if self.use_double else "float")}

        src = """
                unsigned int j, k = 0;
                %(data_t)s dist;
                %(data_t)s h_i = radius_scale2*d_h[i]*d_h[i];
                %(data_t)s h_j;
                unsigned int idx = start_indices[i];
                for(j=0; j<num_particles; j++)
                {
                    h_j = radius_scale2*s_h[j]*s_h[j];
                    dist = NORM2(d_x[i] - s_x[j], d_y[i] - s_y[j], d_z[i] - s_z[j]);
                    if(dist < h_i || dist < h_j)
                    {
                        nbrs[idx + k] = j;
                        k += 1;
                    }
                }
                """ % {"data_t" : ("double" if self.use_double else "float")}

        norm2 = """
                #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
                """

        brute_force_nbrs = ElementwiseKernel(self.ctx,
                arguments, src, "brute_force_nbrs", preamble=norm2)

        brute_force_nbrs(self.src.gpu_x, self.src.gpu_y, self.src.gpu_z,
                self.src.gpu_h, self.dst.gpu_x, self.dst.gpu_y, self.dst.gpu_z,
                self.dst.gpu_h, self.src.get_number_of_particles(),
                start_indices, nbrs, self.radius_scale2)

    cpdef _bin(self, int pa_index):
        pass

    cpdef _refresh(self):
        pass

