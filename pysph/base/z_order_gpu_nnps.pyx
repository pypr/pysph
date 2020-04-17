#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libc.math cimport log
from libcpp.vector cimport vector
from libcpp.map cimport map

from cython.operator cimport dereference as deref, preincrement as inc

# Cython for compiler directives
cimport cython

import pyopencl as cl
import pyopencl.array
import pyopencl.algorithm
from pyopencl.scan import GenericScanKernel
from pyopencl.elementwise import ElementwiseKernel

import numpy as np
cimport numpy as np
from mako.template import Template

from pysph.base.gpu_nnps_helper import GPUNNPSHelper
from compyle.array import Array
import compyle.array as array
from compyle.opencl import get_context, get_config, profile_kernel
from compyle.api import Elementwise

from pysph.base.z_order_gpu_nnps_kernels import (ZOrderNbrsKernel,
                                                 ZOrderLengthKernel)

from pysph.base.gpu_helper_kernels import get_elwise, get_scan
from pysph.base.z_order_gpu_nnps_kernels import *

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
            bint cache=True, bint sort_gids=False, backend=None):
        GPUNNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids, backend
        )

        self.radius_scale2 = radius_scale*radius_scale

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

            self.pids.append(Array(np.uint32, n=num_particles,
                             backend=self.backend))
            self.pid_keys.append(Array(np.uint64, n=num_particles,
                                 backend=self.backend))
            self.cids.append(Array(np.uint32, n=num_particles,
                             backend=self.backend))
            self.cid_to_idx.append(Array(np.int32,
                                   backend=self.backend))

        self.curr_cid = array.ones(1, dtype=np.uint32,
                                   backend=self.backend)
        self.max_cid_src = array.zeros(1, dtype=np.int32,
                                       backend=self.backend)

        self.dst_to_src = Array(np.uint32, backend=self.backend)
        self.overflow_cid_to_idx = Array(np.int32, backend=self.backend)

        self.z_order_nbrs = [None] * 2
        self.z_order_nbr_lengths = [None] * 2

        self.domain.update()
        self.update()

    def get_spatially_ordered_indices(self, int pa_index):
        def update_pids():
            pids_new = array.arange(0, num_particles, 1, dtype=np.uint32,
                                    backend=self.backend)
            self.pids[pa_index].set_data(pids_new)

        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        num_particles = pa_wrapper.get_number_of_particles()
        self.sorted = True
        return self.pids[pa_index], update_pids

    cpdef _bin(self, int pa_index):
        self.sorted = False
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        fill_pids_knl = get_elwise(fill_pids, self.backend)

        pa_gpu = pa_wrapper.pa.gpu
        fill_pids_knl(pa_gpu.x, pa_gpu.y, pa_gpu.z,
                self.cell_size,
                self.xmin[0], self.xmin[1], self.xmin[2],
                self.pid_keys[pa_index], self.pids[pa_index])

        cdef double max_length = fmax(fmax((self.xmax[0] - self.xmin[0]),
            (self.xmax[1] - self.xmin[1])), (self.xmax[2] - self.xmin[2]))

        cdef int max_num_cells = (<int> ceil(max_length/self.hmin))

        cdef int max_num_bits = 3*(<int> ceil(log2(max_num_cells)))

        sorted_keys, sorted_indices = array.sort_by_keys(
            [self.pid_keys[pa_index], self.pids[pa_index]],
            key_bits=max_num_bits,
            backend=self.backend
        )
        self.pids[pa_index].set_data(sorted_indices)
        self.pid_keys[pa_index].set_data(sorted_keys)

        self.curr_cid.fill(1)

        fill_unique_cids_knl = get_scan(inp_fill_unique_cids, out_fill_unique_cids,
                                        np.int32, self.backend)

        fill_unique_cids_knl(keys=self.pid_keys[pa_index],
                             cids=self.cids[pa_index])

        curr_cids = self.cids[pa_index]

        cdef unsigned int num_cids = 1 + <unsigned int> curr_cids[-1]
        self.cid_to_idx[pa_index].resize(27 * num_cids)
        self.cid_to_idx[pa_index].fill(-1)

        self.max_cid[pa_index] = num_cids

        map_cid_to_idx_knl= get_elwise(map_cid_to_idx, self.backend)

        map_cid_to_idx_knl(
            pa_gpu.x, pa_gpu.y, pa_gpu.z,
            pa_wrapper.get_number_of_particles(), self.cell_size,
            self.xmin[0], self.xmin[1], self.xmin[2],
            self.pids[pa_index], self.pid_keys[pa_index],
            self.cids[pa_index], self.cid_to_idx[pa_index]
        )

    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles
        self.max_cid = []
        self.sorted = False

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

            map_dst_to_src_knl = get_elwise(map_dst_to_src, self.backend)

            self.max_cid_src.fill(self.max_cid[src_index])

            map_dst_to_src_knl(self.dst_to_src, self.cids[dst_index],
                    self.cid_to_idx[dst_index],
                    self.pid_keys[dst_index],
                    self.pid_keys[src_index], self.cids[src_index],
                    self.src.get_number_of_particles(), self.max_cid_src)

            overflow_size = <unsigned int>(self.max_cid_src.get()) - \
                    self.max_cid[src_index]

            self.overflow_cid_to_idx.resize(max(1, 27 * overflow_size))
            self.overflow_cid_to_idx.fill(-1)

            fill_overflow_map_knl = get_elwise(fill_overflow_map, self.backend)

            dst_gpu = self.dst.pa.gpu
            fill_overflow_map_knl(self.dst_to_src,
                    self.cid_to_idx[dst_index], dst_gpu.x, dst_gpu.y,
                    dst_gpu.z, self.src.get_number_of_particles(),
                    self.cell_size, self.xmin[0], self.xmin[1], self.xmin[2],
                    self.pid_keys[src_index], self.pids[dst_index],
                    self.overflow_cid_to_idx,
                    np.array(self.max_cid[src_index], dtype=np.uint32))


    cdef void find_neighbor_lengths(self, nbr_lengths):
        if not self.z_order_nbr_lengths[self.dst_src]:
            krnl_source = ZOrderLengthKernel(
                "z_order_nbr_lengths", dst_src=self.dst_src
            )

            self.z_order_nbr_lengths[self.dst_src] = Elementwise(
                krnl_source.function, backend=self.backend
            )

        knl = self.z_order_nbr_lengths[self.dst_src]

        dst_gpu = self.dst.pa.gpu
        src_gpu = self.src.pa.gpu
        knl(dst_gpu.x, dst_gpu.y, dst_gpu.z,
                dst_gpu.h, src_gpu.x, src_gpu.y, src_gpu.z,
                src_gpu.h,
                self.xmin[0], self.xmin[1], self.xmin[2],
                self.src.get_number_of_particles(),
                self.pid_keys[self.src_index],
                self.pids[self.dst_index],
                self.pids[self.src_index],
                self.max_cid[self.src_index], self.cids[self.dst_index],
                self.cid_to_idx[self.src_index],
                self.overflow_cid_to_idx, self.dst_to_src,
                self.radius_scale2, self.cell_size, nbr_lengths)



    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        if not self.z_order_nbrs[self.dst_src]:
            krnl_source = ZOrderNbrsKernel(
                "z_order_nbrs", dst_src=self.dst_src
            )

            self.z_order_nbrs[self.dst_src] = Elementwise(
                krnl_source.function, backend=self.backend
            )

        knl = self.z_order_nbrs[self.dst_src]

        dst_gpu = self.dst.pa.gpu
        src_gpu = self.src.pa.gpu
        knl(dst_gpu.x, dst_gpu.y, dst_gpu.z,
                dst_gpu.h, src_gpu.x, src_gpu.y, src_gpu.z,
                src_gpu.h,
                self.xmin[0], self.xmin[1], self.xmin[2],
                self.src.get_number_of_particles(),
                self.pid_keys[self.src_index],
                self.pids[self.dst_index],
                self.pids[self.src_index],
                self.max_cid[self.src_index], self.cids[self.dst_index],
                self.cid_to_idx[self.src_index],
                self.overflow_cid_to_idx, self.dst_to_src,
                self.radius_scale2, self.cell_size, start_indices, nbrs)
