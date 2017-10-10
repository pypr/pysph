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
from pysph.base.opencl import DeviceArray, get_config


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
        self.make_vec = cl.array.vec.make_double3 if self.use_double \
                else cl.array.vec.make_float3

        self.helper = GPUNNPSHelper(self.ctx, "z_order_gpu_nnps.mako",
                                    self.use_double)

        self.src_index = -1
        self.dst_index = -1
        self.sort_gids = sort_gids

        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles
        self.pids = []
        self.pid_keys = []
        self.cids = []
        self.cid_to_idx = []

        for i from 0<=i<self.narrays:
            pa_wrapper = <NNPSParticleArrayWrapper>self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids.append(DeviceArray(np.uint32, n=num_particles))
            self.pid_keys.append(DeviceArray(np.uint64, n=num_particles))
            self.cids.append(DeviceArray(np.uint32, n=num_particles))
            self.cid_to_idx.append(DeviceArray(np.int32))

        self.curr_cid = 1 + cl.array.zeros(self.queue, 1, dtype=np.uint32)
        self.max_cid_src = cl.array.zeros(self.queue, 1, dtype=np.int32)

        self.dst_to_src = DeviceArray(np.uint32)
        self.overflow_cid_to_idx = DeviceArray(np.int32)

        self.domain.update()
        self.update()

    cpdef _bin(self, int pa_index):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        fill_pids = self.helper.get_kernel("fill_pids")

        pa_gpu = pa_wrapper.pa.gpu
        fill_pids(pa_gpu.x, pa_gpu.y, pa_gpu.z,
                self.cell_size,
                self.make_vec(self.xmin[0], self.xmin[1], self.xmin[2]),
                self.pid_keys[pa_index].array, self.pids[pa_index].array)

        if self.radix_sort is None:
            self.radix_sort = cl.algorithm.RadixSort(
                self.ctx,
                "unsigned int* pids, unsigned long* keys",
                scan_kernel=GenericScanKernel, key_expr="keys[i]",
                sort_arg_names=["pids", "keys"]
            )


        (sorted_indices, sorted_keys), evnt = self.radix_sort(
            self.pids[pa_index].array, self.pid_keys[pa_index].array, key_bits=64
        )
        self.pids[pa_index].set_data(sorted_indices)
        self.pid_keys[pa_index].set_data(sorted_keys)

        self.curr_cid.fill(1)

        fill_unique_cids = self.helper.get_kernel("fill_unique_cids")

        fill_unique_cids(self.pid_keys[pa_index].array,
                self.cids[pa_index].array, self.curr_cid)

        cdef unsigned int num_cids = <unsigned int> (self.curr_cid.get())
        self.cid_to_idx[pa_index].resize(27 * num_cids)
        self.cid_to_idx[pa_index].fill(-1)

        self.max_cid[pa_index] = num_cids

        map_cid_to_idx = self.helper.get_kernel("map_cid_to_idx")

        map_cid_to_idx(
            pa_gpu.x, pa_gpu.y, pa_gpu.z,
            pa_wrapper.get_number_of_particles(), self.cell_size,
            self.make_vec(self.xmin[0], self.xmin[1], self.xmin[2]),
            self.pids[pa_index].array, self.pid_keys[pa_index].array,
            self.cids[pa_index].array, self.cid_to_idx[pa_index].array
        )

        fill_cids = self.helper.get_kernel("fill_cids")

        fill_cids(self.pid_keys[pa_index].array, self.cids[pa_index].array,
                pa_wrapper.get_number_of_particles())

    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles
        self.max_cid = []
        self._sorted = False

        for i from 0<=i<self.narrays:
            pa_wrapper = <NNPSParticleArrayWrapper>self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids[i].resize(num_particles)
            self.pid_keys[i].resize(num_particles)
            self.cids[i].resize(num_particles)
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

        if self.dst_src:
            self.dst_to_src.resize(self.max_cid[dst_index])

            map_dst_to_src = self.helper.get_kernel("map_dst_to_src")

            self.max_cid_src.fill(self.max_cid[src_index])

            map_dst_to_src(self.dst_to_src.array, self.cids[dst_index].array,
                    self.cid_to_idx[dst_index].array,
                    self.pid_keys[dst_index].array,
                    self.pid_keys[src_index].array, self.cids[src_index].array,
                    self.src.get_number_of_particles(), self.max_cid_src)

            overflow_size = <unsigned int>(self.max_cid_src.get()) - \
                    self.max_cid[src_index]

            self.overflow_cid_to_idx.resize(max(1, 27 * overflow_size))
            self.overflow_cid_to_idx.fill(-1)

            fill_overflow_map = self.helper.get_kernel("fill_overflow_map")

            dst_gpu = self.dst.pa.gpu
            fill_overflow_map(self.dst_to_src.array,
                    self.cid_to_idx[dst_index].array, dst_gpu.x, dst_gpu.y,
                    dst_gpu.z, self.src.get_number_of_particles(),
                    self.cell_size,
                    self.make_vec(self.xmin[0], self.xmin[1],
                        self.xmin[2]),
                    self.pid_keys[src_index].array, self.pids[dst_index].array,
                    self.overflow_cid_to_idx.array,
                    <unsigned int> self.max_cid[src_index])


    cdef void find_neighbor_lengths(self, nbr_lengths):
        z_order_nbr_lengths = self.helper.get_kernel(
                "z_order_nbr_lengths", sorted=self._sorted,
                dst_src=self.dst_src)

        dst_gpu = self.dst.pa.gpu
        src_gpu = self.src.pa.gpu
        z_order_nbr_lengths(dst_gpu.x, dst_gpu.y, dst_gpu.z,
                dst_gpu.h, src_gpu.x, src_gpu.y, src_gpu.z, src_gpu.h,
                self.make_vec(self.xmin[0], self.xmin[1],
                    self.xmin[2]), self.src.get_number_of_particles(),
                self.pid_keys[self.src_index].array,
                self.pids[self.dst_index].array,
                self.pids[self.src_index].array,
                self.max_cid[self.src_index], self.cids[self.dst_index].array,
                self.cid_to_idx[self.src_index].array,
                self.overflow_cid_to_idx.array, self.dst_to_src.array,
                nbr_lengths, self.radius_scale2, self.cell_size)

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        z_order_nbrs = self.helper.get_kernel(
                "z_order_nbrs", sorted=self._sorted,
                dst_src=self.dst_src)

        dst_gpu = self.dst.pa.gpu
        src_gpu = self.src.pa.gpu
        z_order_nbrs(dst_gpu.x, dst_gpu.y, dst_gpu.z,
                dst_gpu.h, src_gpu.x, src_gpu.y, src_gpu.z, src_gpu.h,
                self.make_vec(self.xmin[0], self.xmin[1],
                    self.xmin[2]),
                self.src.get_number_of_particles(),
                self.pid_keys[self.src_index].array,
                self.pids[self.dst_index].array,
                self.pids[self.src_index].array,
                self.max_cid[self.src_index], self.cids[self.dst_index].array,
                self.cid_to_idx[self.src_index].array,
                self.overflow_cid_to_idx.array, self.dst_to_src.array,
                start_indices, nbrs, self.radius_scale2, self.cell_size)

