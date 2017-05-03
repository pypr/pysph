#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair

from cython.operator cimport dereference as deref, preincrement as inc

from nnps_base cimport *
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

DEF EPS = 1e-13

IF UNAME_SYSNAME == "Windows":
    cdef inline double fmin(double x, double y) nogil:
        return x if x < y else y
    cdef inline double fmax(double x, double y) nogil:
        return x if x > y else y

    @cython.cdivision(True)
    cdef inline double log2(double n) nogil:
        return log(n)/log(2)

#############################################################################
cdef class StratifiedSFCNNPS(NNPS):

    """Finds nearest neighbors using Space-filling curves with stratified grids"""

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int num_levels = 1):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.radius_scale2 = radius_scale*radius_scale
        self.interval_size = 0

        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False, int num_levels = 1):

        cdef int narrays = len(particles)

        if fixed_h:
            self.num_levels = 1
        else:
            self.num_levels = num_levels

        self.pids = <u_int_vector_t***> malloc(narrays*sizeof(u_int_vector_t**))
        self.pid_indices = <key_to_idx_t***> malloc(narrays*sizeof(key_to_idx_t**))
        self.cell_sizes = <double**> malloc(narrays*sizeof(double*))

        cdef u_int_vector_t** current_pids
        cdef key_to_idx_t** current_indices
        cdef int i, j
        for i from 0<=i<narrays:
            self.pids[i] = <u_int_vector_t**> \
                    malloc(self.num_levels*sizeof(u_int_vector_t*))
            self.pid_indices[i] = <key_to_idx_t**> \
                    malloc(self.num_levels*sizeof(key_to_idx_t*))
            self.cell_sizes[i] = <double*> malloc(self.num_levels*sizeof(double))

            current_pids = self.pids[i]
            current_indices = self.pid_indices[i]
            for j from 0<=j<self.num_levels:
                current_pids[j] = NULL
                current_indices[j] = NULL

        self.current_pids = NULL
        self.current_indices = NULL
        self.current_cells = NULL

    def __dealloc__(self):
        cdef u_int_vector_t** current_pids
        cdef key_to_idx_t** current_indices
        cdef int i, j
        for i from 0<=i<self.narrays:
            current_pids = self.pids[i]
            current_indices = self.pid_indices[i]
            for j from 0<=j<self.num_levels:
                if current_pids[j] != NULL:
                    del current_pids[j]
                if current_indices[j] != NULL:
                    del current_indices[j]
            free(current_pids)
            free(current_indices)
            free(self.cell_sizes[i])
        free(self.pids)
        free(self.pid_indices)
        free(self.cell_sizes)

    #### Public protocol ################################################

    cpdef int count_particles(self, int interval):
        """Count number of particles in at a level"""
        return self.current_pids[interval].size()

    cpdef double get_binning_size(self, int interval):
        """Get bin size at a level"""
        return self.radius_scale*self.current_cells[interval]

    cpdef set_context(self, int src_index, int dst_index):
        """Set context for nearest neighbor searches.

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        """
        NNPS.set_context(self, src_index, dst_index)
        self.current_pids = self.pids[src_index]
        self.current_indices = self.pid_indices[src_index]
        self.current_cells = self.cell_sizes[src_index]

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        indices.reset()
        cdef int num_particles

        cdef u_int_vector_t** current_pids = self.pids[pa_index]
        cdef u_int_vector_t* pids_level

        cdef int i, j
        for i from 0<=i<self.num_levels:
            pids_level = current_pids[i]
            num_particles = pids_level.size()
            for j from 0<=j<num_particles:
                indices.c_append(<long>deref(pids_level)[j])

    @cython.cdivision(True)
    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil:
        """Low level, high-performance non-gil method to find neighbors.
        This requires that `set_context()` be called beforehand.  This method
        does not reset the neighbors array before it appends the
        neighbors to it.

        """
        cdef double* dst_x_ptr = self.dst.x.data
        cdef double* dst_y_ptr = self.dst.y.data
        cdef double* dst_z_ptr = self.dst.z.data
        cdef double* dst_h_ptr = self.dst.h.data

        cdef double* src_x_ptr = self.src.x.data
        cdef double* src_y_ptr = self.src.y.data
        cdef double* src_z_ptr = self.src.z.data
        cdef double* src_h_ptr = self.src.h.data

        cdef double x = dst_x_ptr[d_idx]
        cdef double y = dst_y_ptr[d_idx]
        cdef double z = dst_z_ptr[d_idx]
        cdef double h = dst_h_ptr[d_idx]

        cdef unsigned int* s_gid = self.src.gid.data
        cdef int orig_length = nbrs.length

        cdef int c_x, c_y, c_z
        cdef double* xmin = self.xmin.data
        cdef unsigned int i, j, k, n, idx

        cdef pair[u_int, u_int] candidate
        cdef int candidate_length

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2*h*h
        cdef double hj2 = 0

        cdef double h_max
        cdef int H, mask_len, num_boxes

        cdef int* x_boxes
        cdef int* y_boxes
        cdef int* z_boxes

        cdef map[u_int, pair[u_int, u_int]].iterator it
        cdef u_int_vector_t* pids_level
        cdef key_to_idx_t* indices_level

        for i from 0<=i<self.num_levels:
            pids_level = self.current_pids[i]
            indices_level = self.current_indices[i]

            h_max = fmax(self.radius_scale*h, self.radius_scale*self.current_cells[i])
            H = <int> ceil(h_max/(self.radius_scale*self.current_cells[i]))

            mask_len = (2*H+1)*(2*H+1)*(2*H+1)

            x_boxes = <int*> malloc(mask_len*sizeof(int))
            y_boxes = <int*> malloc(mask_len*sizeof(int))
            z_boxes = <int*> malloc(mask_len*sizeof(int))

            find_cell_id_raw(
                    x - xmin[0],
                    y - xmin[1],
                    z - xmin[2],
                    self.radius_scale*self.current_cells[i],
                    &c_x, &c_y, &c_z
                    )

            num_boxes = self._neighbor_boxes(c_x, c_y, c_z,
                    x_boxes, y_boxes, z_boxes, H)

            for j from 0<=j<num_boxes:
                it = indices_level.find(get_key(x_boxes[j], y_boxes[j],
                    z_boxes[j]))
                if it == indices_level.end():
                    continue
                candidate = deref(it).second
                n = candidate.first
                candidate_length = candidate.second

                for k from 0<=k<candidate_length:
                    idx = deref(pids_level)[n+k]

                    hj2 = self.radius_scale2*src_h_ptr[idx]*src_h_ptr[idx]

                    xij2 = norm2(
                        src_x_ptr[idx] - x,
                        src_y_ptr[idx] - y,
                        src_z_ptr[idx] - z
                        )

                    if (xij2 < hi2) or (xij2 < hj2):
                        nbrs.c_append(idx)

            free(x_boxes)
            free(y_boxes)
            free(z_boxes)

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )


    #### Private protocol ################################################

    @cython.cdivision(True)
    cdef inline int _get_level(self, double h) nogil:
        return <int> floor((self.radius_scale*h - self.hmin)/self.interval_size)

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z, int H) nogil:
        cdef int length = 0
        cdef int p, q, r
        for p from -H<=p<H+1:
            for q from -H<=q<H+1:
                for r from -H<=r<H+1:
                    if i+r>=0 and j+q>=0 and k+p>=0:
                        x[length] = i+r
                        y[length] = j+q
                        z[length] = k+p
                        length += 1
        return length

    @cython.cdivision(True)
    cpdef _refresh(self):
        self.interval_size = ((self.cell_size - self.hmin)/self.num_levels) + EPS

        cdef u_int_vector_t** current_pids
        cdef key_to_idx_t** current_indices
        cdef double* current_cells

        cdef int i, j
        for i from 0<=i<self.narrays:
            current_pids = self.pids[i]
            current_indices = self.pid_indices[i]
            current_cells = self.cell_sizes[i]
            for j from 0<=j<self.num_levels:
                if current_pids[j] != NULL:
                    del current_pids[j]
                if current_indices[j] != NULL:
                    del current_indices[j]
                current_pids[j] = new u_int_vector_t()
                current_indices[j] = new key_to_idx_t()
                current_cells[j] = 0
        self.current_pids = self.pids[self.src_index]
        self.current_indices = self.pid_indices[self.src_index]
        self.current_cells = self.cell_sizes[self.src_index]

    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper,
            int pa_index, UIntArray indices, u_int_vector_t** current_pids,
            key_to_idx_t** current_indices, double* current_cells):
        cdef double* x_ptr = pa_wrapper.x.data
        cdef double* y_ptr = pa_wrapper.y.data
        cdef double* z_ptr = pa_wrapper.z.data
        cdef double* h_ptr = pa_wrapper.h.data

        cdef double* xmin = self.xmin.data

        cdef int id_x, id_y, id_z
        cdef int c_x, c_y, c_z

        cdef u_int_vector_t* pids_level
        cdef key_to_idx_t* indices_level

        cdef int i, j, n, level
        for i from 0<=i<indices.length:
            n = indices.data[i]
            level = self._get_level(h_ptr[n])
            pids_level = current_pids[level]
            current_cells[level] = fmax(h_ptr[n], current_cells[level])
            pids_level.push_back(n)

        cdef CompareSortWrapper sort_wrapper
        cdef double cell_size

        cdef pair[u_int, pair[u_int, u_int]] temp
        cdef pair[u_int, u_int] cell

        cdef u_int length

        for level from 0<=level<self.num_levels:
            length = 0
            pids_level = current_pids[level]
            if pids_level.size() == 0:
                continue
            indices_level = current_indices[level]
            cell_size = self.radius_scale*current_cells[level]
            sort_wrapper = CompareSortWrapper(x_ptr, y_ptr, z_ptr, xmin, cell_size,
                    &(pids_level.front()), pids_level.size())
            sort_wrapper.compare_sort()

            j = deref(pids_level)[0]

            find_cell_id_raw(
                    x_ptr[j] - xmin[0],
                    y_ptr[j] - xmin[1],
                    z_ptr[j] - xmin[2],
                    cell_size,
                    &c_x, &c_y, &c_z
                    )

            temp.first = get_key(c_x, c_y, c_z)
            cell.first = 0

            for i from 0<i<pids_level.size():
                j = deref(pids_level)[i]
                find_cell_id_raw(
                        x_ptr[j] - xmin[0],
                        y_ptr[j] - xmin[1],
                        z_ptr[j] - xmin[2],
                        cell_size,
                        &id_x, &id_y, &id_z
                        )

                length += 1

                if(id_x != c_x or id_y != c_y or id_z != c_z):
                    cell.second = length
                    temp.second = cell
                    indices_level.insert(temp)

                    temp.first = get_key(id_x, id_y, id_z)
                    cell.first = i

                    length = 0

                    c_x = id_x
                    c_y = id_y
                    c_z = id_z

            cell.second = length + 1
            temp.second = cell
            indices_level.insert(temp)

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        cdef double* current_cells = self.cell_sizes[pa_index]
        cdef u_int_vector_t** current_pids = self.pids[pa_index]
        cdef key_to_idx_t** current_indices = self.pid_indices[pa_index]

        self.fill_array(pa_wrapper, pa_index, indices, current_pids,
                current_indices, current_cells)

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

        self.helper = GPUNNPSHelper(self.ctx, "stratified_sfc_nnps.mako", use_double)
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
                pa_wrapper.copy_to_gpu(self.queue, np.float64)
            else:
                pa_wrapper.copy_to_gpu(self.queue, np.float32)

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


