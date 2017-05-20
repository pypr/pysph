#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair

from cython.operator cimport dereference as deref, preincrement as inc

from gpu_nnps_helper import GPUNNPSHelper

import pyopencl as cl
import pyopencl.array
import pyopencl.algorithm
from pyopencl.scan import GenericScanKernel
from pyopencl.scan import GenericDebugScanKernel
from pyopencl.elementwise import ElementwiseKernel

# Cython for compiler directives
cimport cython

import numpy as np
cimport numpy as np

IF UNAME_SYSNAME == "Windows":
    cdef inline double fmin(double x, double y) nogil:
        return x if x < y else y
    cdef inline double fmax(double x, double y) nogil:
        return x if x > y else y

    @cython.cdivision(True)
    cdef inline double log2(double n) nogil:
        return log(n)/log(2)


cdef class StratifiedSFCGPUNNPS(GPUNNPS):
    def __init__(self, int dim, list particles, double radius_scale=2.0,
            int ghost_layers=1, domain=None, bint fixed_h=False,
            bint cache=True, bint sort_gids=False, bint use_double=True,
            int num_levels=2, ctx=None):
        GPUNNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids, ctx
        )

        self.radius_scale2 = radius_scale*radius_scale
        self.use_double = use_double

        self.helper = GPUNNPSHelper(self.ctx, "stratified_sfc_gpu_nnps.mako", use_double)
        self.eps = 16*np.finfo(np.float32).eps

        self.num_levels = num_levels
        self.src_index = -1
        self.dst_index = -1
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    @cython.cdivision(True)
    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, j, num_particles
        self.pids = []
        self.pid_keys = []
        self.start_idx_levels = []
        self.num_particles_levels = []
        self._sorted = False

        if self.cell_size - self.hmin > self.hmin*self.eps:
            self.interval_size = \
                    (self.cell_size - self.hmin)*(1 + self.eps)/self.num_levels
        else:
            self.interval_size = self.hmin*self.eps

        for i from 0<=i<self.narrays:
            pa_wrapper = <NNPSParticleArrayWrapper>self.pa_wrappers[i]

            if self.use_double:
                copy_to_gpu(pa_wrapper, self.queue, np.float64)
            else:
                copy_to_gpu(pa_wrapper, self.queue, np.float32)

            num_particles = pa_wrapper.get_number_of_particles()

            self.pids.append(cl.array.empty(self.queue,
                num_particles, dtype=np.uint32))
            self.pid_keys.append(cl.array.empty(self.queue,
                num_particles, dtype=np.uint64))
            start_idx_i = num_particles + cl.array.zeros(self.queue,
                self.num_levels, dtype=np.uint32)
            self.start_idx_levels.append(start_idx_i)
            self.num_particles_levels.append(cl.array.zeros_like(start_idx_i))

        cdef double max_length = fmax(fmax((self.xmax[0] - self.xmin[0]),
            (self.xmax[1] - self.xmin[1])), (self.xmax[2] - self.xmin[2]))

        cdef int max_num_cells = (<int> ceil(max_length/self.hmin))

        self.max_num_bits = 1 + 3*(<int> ceil(log2(max_num_cells)))

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        cdef np.ndarray current_pids = (self.pids[pa_index].get()).astype(np.int64)
        indices.resize(current_pids.size)
        indices.set_data(current_pids)
        self._sorted = True

    cpdef _bin(self, int pa_index):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        fill_pids = self.helper.get_kernel("fill_pids")

        levels = cl.array.empty(self.queue, pa_wrapper.get_number_of_particles(),
                dtype=np.int32)

        fill_pids(pa_wrapper.gpu_x, pa_wrapper.gpu_y, pa_wrapper.gpu_z, pa_wrapper.gpu_h,
                self.interval_size, self.xmin[0], self.xmin[1], self.xmin[2], self.hmin,
                self.pid_keys[pa_index], self.pids[pa_index],
                self.radius_scale, self.max_num_bits)

        radix_sort = cl.algorithm.RadixSort(self.ctx,
                "unsigned int* pids, unsigned long* keys",
                scan_kernel=GenericScanKernel, key_expr="keys[i]",
                sort_arg_names=["pids", "keys"])

        (sorted_indices, sorted_keys), evnt = radix_sort(self.pids[pa_index],
                self.pid_keys[pa_index], key_bits=64)
        self.pids[pa_index] = sorted_indices
        self.pid_keys[pa_index] = sorted_keys

        cdef unsigned long long key = <unsigned long long> (sorted_keys[0].get())

        self.start_idx_levels[pa_index][key >> self.max_num_bits] = 0

        fill_start_indices = self.helper.get_kernel("fill_start_indices")

        fill_start_indices(self.pid_keys[pa_index], self.start_idx_levels[pa_index],
                self.max_num_bits, self.num_particles_levels[pa_index])

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

        self.src = self.pa_wrappers[src_index]
        self.dst = self.pa_wrappers[dst_index]


    cdef void find_neighbor_lengths(self, nbr_lengths):
        find_nbr_lengths = self.helper.get_kernel("find_nbr_lengths",
                sorted=self._sorted)

        make_vec = cl.array.vec.make_double3 if self.use_double \
                else cl.array.vec.make_float3

        mask_lengths = cl.array.zeros(self.queue, self.dst.get_number_of_particles(),
                dtype=np.int32)

        find_nbr_lengths(self.dst.gpu_x, self.dst.gpu_y, self.dst.gpu_z,
                self.dst.gpu_h, self.src.gpu_x, self.src.gpu_y, self.src.gpu_z,
                self.src.gpu_h,
                make_vec(self.xmin.data[0], self.xmin.data[1], self.xmin.data[2]),
                self.src.get_number_of_particles(), self.pid_keys[self.src_index],
                self.pids[self.dst_index], self.pids[self.src_index], nbr_lengths,
                self.radius_scale, self.hmin, self.interval_size,
                self.start_idx_levels[self.src_index],
                self.max_num_bits, self.num_levels, self.num_particles_levels[self.src_index])

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        find_nbrs = self.helper.get_kernel("find_nbrs",
                sorted=self._sorted)

        make_vec = cl.array.vec.make_double3 if self.use_double \
                else cl.array.vec.make_float3

        find_nbrs(self.dst.gpu_x, self.dst.gpu_y, self.dst.gpu_z,
                self.dst.gpu_h, self.src.gpu_x, self.src.gpu_y, self.src.gpu_z,
                self.src.gpu_h,
                make_vec(self.xmin.data[0], self.xmin.data[1], self.xmin.data[2]),
                self.src.get_number_of_particles(), self.pid_keys[self.src_index],
                self.pids[self.dst_index], self.pids[self.src_index],
                start_indices, nbrs, self.radius_scale, self.hmin, self.interval_size,
                self.start_idx_levels[self.src_index],
                self.max_num_bits, self.num_levels, self.num_particles_levels[self.src_index])


