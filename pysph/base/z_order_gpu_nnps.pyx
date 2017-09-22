#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.map cimport map

from cython.operator cimport dereference as deref, preincrement as inc

# Cython for compiler directives
cimport cython

import pyopencl as cl
import pyopencl.array
import pyopencl.algorithm
from pyopencl.scan import GenericScanKernel
from pyopencl.scan import GenericDebugScanKernel
from pyopencl.elementwise import ElementwiseKernel

import numpy as np
cimport numpy as np

from pysph.base.gpu_nnps_helper import GPUNNPSHelper

IF UNAME_SYSNAME == "Windows":
    cdef inline double fmin(double x, double y) nogil:
        return x if x < y else y
    cdef inline double fmax(double x, double y) nogil:
        return x if x > y else y

    @cython.cdivision(True)
    cdef inline double log2(double n) nogil:
        return log(n)/log(2)


cdef class ZOrderGPUNNPS(GPUNNPS):
    def __init__(self, int dim, list particles, double radius_scale=2.0,
            int ghost_layers=1, domain=None, bint fixed_h=False,
            bint cache=True, bint sort_gids=False, ctx=None):
        GPUNNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids, ctx
        )

        self.radius_scale2 = radius_scale*radius_scale
        self.radix_sort = None

        self.helper = GPUNNPSHelper(self.ctx, "z_order_gpu_nnps.mako",
                                    self.use_double)

        self.src_index = -1
        self.dst_index = -1
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    cpdef _bin(self, int pa_index):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        fill_pids = self.helper.get_kernel("fill_pids")

        pa_gpu = pa_wrapper.pa.gpu
        fill_pids(pa_gpu.x, pa_gpu.y, pa_gpu.z,
                self.cell_size, self.xmin[0], self.xmin[1], self.xmin[2],
                self.pid_keys[pa_index], self.pids[pa_index])

        if self.radix_sort is None:
            self.radix_sort = cl.algorithm.RadixSort(
                self.ctx,
                "unsigned int* pids, unsigned long* keys",
                scan_kernel=GenericScanKernel, key_expr="keys[i]",
                sort_arg_names=["pids", "keys"]
            )


        (sorted_indices, sorted_keys), evnt = self.radix_sort(
            self.pids[pa_index], self.pid_keys[pa_index], key_bits=64
        )
        self.pids[pa_index] = sorted_indices
        self.pid_keys[pa_index] = sorted_keys

        curr_cid = 1 + cl.array.zeros(self.queue, 1, dtype=np.uint32)

        fill_unique_cids = self.helper.get_kernel("fill_unique_cids")

        fill_unique_cids(self.pid_keys[pa_index], self.cids[pa_index],
                curr_cid)

        cdef unsigned int num_cids = <unsigned int> (curr_cid.get())
        self.cid_to_idx[pa_index] = -1 + cl.array.zeros(self.queue,
                27*num_cids, dtype=np.int32)

        self.max_cid[pa_index] = num_cids

        map_cid_to_idx = self.helper.get_kernel("map_cid_to_idx")

        make_vec = cl.array.vec.make_double3 if self.use_double \
                else cl.array.vec.make_float3

        map_cid_to_idx(
            pa_gpu.x, pa_gpu.y, pa_gpu.z,
            pa_wrapper.get_number_of_particles(), self.cell_size,
            make_vec(self.xmin.data[0], self.xmin.data[1], self.xmin.data[2]),
            self.pids[pa_index], self.pid_keys[pa_index], self.cids[pa_index],
            self.cid_to_idx[pa_index]
        )

        fill_cids = self.helper.get_kernel("fill_cids")

        fill_cids(self.pid_keys[pa_index], self.cids[pa_index],
                pa_wrapper.get_number_of_particles())


    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles
        self.pids = []
        self.pid_keys = []
        self.cids = []
        self.cid_to_idx = [None for i in range(self.narrays)]
        self.max_cid = []
        self._sorted = False

        for i from 0<=i<self.narrays:
            pa_wrapper = <NNPSParticleArrayWrapper>self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids.append(cl.array.empty(self.queue,
                num_particles, dtype=np.uint32))
            self.pid_keys.append(cl.array.empty(self.queue,
                num_particles, dtype=np.uint64))
            self.cids.append(cl.array.empty(self.queue,
                num_particles, dtype=np.uint32))
            self.max_cid.append(0)

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

        self.dst_src = src_index != dst_index

        cdef unsigned int overflow_size = 0

        self.dst_to_src = cl.array.zeros(self.queue,
                self.max_cid[dst_index], dtype=np.uint32)

        if self.dst_src:

            map_dst_to_src = self.helper.get_kernel("map_dst_to_src")

            max_cid_src = self.max_cid[src_index] + \
                    cl.array.zeros(self.queue, 1, dtype=np.int32)

            map_dst_to_src(self.dst_to_src, self.cids[dst_index],
                    self.cid_to_idx[dst_index], self.pid_keys[dst_index],
                    self.pid_keys[src_index], self.cids[src_index],
                    self.src.get_number_of_particles(), max_cid_src)

            overflow_size = <unsigned int>(max_cid_src.get()) - \
                    self.max_cid[src_index]

            self.overflow_cid_to_idx = -1 + cl.array.zeros(self.queue,
                    max(1, 27*overflow_size), dtype=np.int32)

            fill_overflow_map = self.helper.get_kernel("fill_overflow_map")

            make_vec = cl.array.vec.make_double3 if self.use_double \
                    else cl.array.vec.make_float3

            dst_gpu = self.dst.pa.gpu
            fill_overflow_map(self.dst_to_src, self.cid_to_idx[dst_index],
                    dst_gpu.x, dst_gpu.y, dst_gpu.z,
                    self.src.get_number_of_particles(), self.cell_size,
                    make_vec(self.xmin.data[0], self.xmin.data[1], self.xmin.data[2]),
                    self.pid_keys[src_index], self.pids[dst_index],
                    self.overflow_cid_to_idx, <unsigned int> self.max_cid[src_index])

        else:

            self.overflow_cid_to_idx = -1 + cl.array.zeros(self.queue,
                    1, dtype=np.int32)


    cdef void find_neighbor_lengths(self, nbr_lengths):
        z_order_nbr_lengths = self.helper.get_kernel("z_order_nbr_lengths",
                sorted=self._sorted, dst_src=self.dst_src)

        make_vec = cl.array.vec.make_double3 if self.use_double \
                else cl.array.vec.make_float3

        dst_gpu = self.dst.pa.gpu
        src_gpu = self.src.pa.gpu
        z_order_nbr_lengths(dst_gpu.x, dst_gpu.y, dst_gpu.z,
                dst_gpu.h, src_gpu.x, src_gpu.y, src_gpu.z, src_gpu.h,
                make_vec(self.xmin.data[0], self.xmin.data[1], self.xmin.data[2]),
                self.src.get_number_of_particles(), self.pid_keys[self.src_index],
                self.pids[self.dst_index], self.pids[self.src_index],
                self.max_cid[self.src_index], self.cids[self.dst_index],
                self.cid_to_idx[self.src_index], self.overflow_cid_to_idx,
                self.dst_to_src, nbr_lengths, self.radius_scale2, self.cell_size)

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        z_order_nbrs = self.helper.get_kernel("z_order_nbrs",
                sorted=self._sorted, dst_src=self.dst_src)

        make_vec = cl.array.vec.make_double3 if self.use_double \
                else cl.array.vec.make_float3

        dst_gpu = self.dst.pa.gpu
        src_gpu = self.src.pa.gpu
        z_order_nbrs(dst_gpu.x, dst_gpu.y, dst_gpu.z,
                dst_gpu.h, src_gpu.x, src_gpu.y, src_gpu.z, src_gpu.h,
                make_vec(self.xmin.data[0], self.xmin.data[1], self.xmin.data[2]),
                self.src.get_number_of_particles(), self.pid_keys[self.src_index],
                self.pids[self.dst_index], self.pids[self.src_index],
                self.max_cid[self.src_index], self.cids[self.dst_index],
                self.cid_to_idx[self.src_index], self.overflow_cid_to_idx,
                self.dst_to_src, start_indices, nbrs, self.radius_scale2, self.cell_size)
