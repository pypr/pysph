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

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)
    void sort[Iter](Iter first, Iter last)

#############################################################################

cdef class ZOrderNNPS(NNPS):

    """Find nearest neighbors using Z-Order space filling curve"""

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.radius_scale2 = radius_scale*radius_scale
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles

        for i from 0<=i<self.narrays:
            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids[i] = <u_int*> malloc(num_particles*sizeof(u_int))
            self.pid_indices[i] = new key_to_idx_t()

        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False):
        cdef int narrays = len(particles)

        self.pids = <u_int**> malloc(narrays*sizeof(u_int*))
        self.pid_indices = <key_to_idx_t**> malloc(narrays*sizeof(key_to_idx_t*))

        self.current_pids = NULL
        self.current_indices = NULL

    def __dealloc__(self):
        cdef int i
        for i from 0<=i<self.narrays:
            free(self.pids[i])
            del self.pid_indices[i]
        free(self.pids)
        free(self.pid_indices)

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

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]

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
        cdef int i, j

        find_cell_id_raw(
                x - xmin[0],
                y - xmin[1],
                z - xmin[2],
                self.cell_size,
                &c_x, &c_y, &c_z
                )

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2*h*h
        cdef double hj2 = 0

        cdef map[u_int, pair[u_int, u_int]].iterator it

        cdef int x_boxes[27]
        cdef int y_boxes[27]
        cdef int z_boxes[27]
        cdef int num_boxes = self._neighbor_boxes(c_x, c_y, c_z,
                x_boxes, y_boxes, z_boxes)

        cdef pair[u_int, u_int] candidate

        cdef u_int n, idx
        for i from 0<=i<num_boxes:
            it = self.current_indices.find(get_key(x_boxes[i], y_boxes[i],
                z_boxes[i]))
            if it == self.current_indices.end():
                continue
            candidate = deref(it).second
            n = candidate.first
            candidate_length = candidate.second

            for j from 0<=j<candidate_length:
                idx = self.current_pids[n+j]

                hj2 = self.radius_scale2*src_h_ptr[idx]*src_h_ptr[idx]

                xij2 = norm2(
                    src_x_ptr[idx] - x,
                    src_y_ptr[idx] - y,
                    src_z_ptr[idx] - z
                    )

                if (xij2 < hi2) or (xij2 < hj2):
                    nbrs.c_append(idx)

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Find nearest neighbors for particle id 'd_idx' without cache

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        d_idx: size_t
            Index of the query point in the destination particle array

        nbrs: UIntArray
            Array to be populated by nearest neighbors of 'd_idx'

        """
        self.set_context(src_index, dst_index)

        if prealloc:
            nbrs.length = 0
        else:
            nbrs.c_reset()

        self.find_nearest_neighbors(d_idx, nbrs)

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        indices.reset()
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        cdef int num_particles = pa_wrapper.get_number_of_particles()

        cdef u_int* current_pids = self.pids[pa_index]

        cdef int j
        for j from 0<=j<num_particles:
            indices.c_append(<long>current_pids[j])

    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper, int pa_index,
            UIntArray indices, u_int* current_pids, key_to_idx_t* current_indices):
        cdef double* x_ptr = pa_wrapper.x.data
        cdef double* y_ptr = pa_wrapper.y.data
        cdef double* z_ptr = pa_wrapper.z.data

        cdef double* xmin = self.xmin.data

        cdef int id_x, id_y, id_z
        cdef int c_x, c_y, c_z

        cdef int i, n
        for i from 0<=i<indices.length:
            current_pids[i] = i

        cdef CompareSortWrapper sort_wrapper = \
                CompareSortWrapper(x_ptr, y_ptr, z_ptr, xmin, self.cell_size,
                        current_pids, indices.length)

        sort_wrapper.compare_sort()

        cdef pair[u_int, pair[u_int, u_int]] temp
        cdef pair[u_int, u_int] cell

        cdef int j
        j = current_pids[0]

        find_cell_id_raw(
                x_ptr[j] - xmin[0],
                y_ptr[j] - xmin[1],
                z_ptr[j] - xmin[2],
                self.cell_size,
                &c_x, &c_y, &c_z
                )

        temp.first = get_key(c_x, c_y, c_z)
        cell.first = 0

        cdef u_int length = 0

        for i from 0<i<indices.length:
            j = current_pids[i]
            find_cell_id_raw(
                    x_ptr[j] - xmin[0],
                    y_ptr[j] - xmin[1],
                    z_ptr[j] - xmin[2],
                    self.cell_size,
                    &id_x, &id_y, &id_z
                    )

            length += 1

            if(id_x != c_x or id_y != c_y or id_z != c_z):
                cell.second = length
                temp.second = cell
                current_indices.insert(temp)

                temp.first = get_key(id_x, id_y, id_z)
                cell.first = i

                length = 0

                c_x = id_x
                c_y = id_y
                c_z = id_z

        cell.second = length + 1
        temp.second = cell
        current_indices.insert(temp)

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z) nogil:
        cdef int length = 0
        cdef int p, q, r
        for p from -1<=p<2:
            for q from -1<=q<2:
                for r from -1<=r<2:
                    if i+r>=0 and j+q>=0 and k+p>=0:
                        x[length] = i+r
                        y[length] = j+q
                        z[length] = k+p
                        length += 1
        return length

    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper

        cdef int i, num_particles

        cdef double* xmax = self.xmax.data
        cdef double* xmin = self.xmin.data

        for i from 0<=i<self.narrays:
            free(self.pids[i])
            del self.pid_indices[i]

            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids[i] = <u_int*> malloc(num_particles*sizeof(u_int))
            self.pid_indices[i] = new key_to_idx_t()

        self.current_pids = self.pids[self.src_index]
        self.current_indices = self.pid_indices[self.src_index]

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        cdef int num_particles = pa_wrapper.get_number_of_particles()

        cdef u_int* current_pids = self.pids[pa_index]
        cdef key_to_idx_t* current_indices = self.pid_indices[pa_index]

        self.fill_array(pa_wrapper, pa_index, indices, current_pids, current_indices)


cdef class ZOrderGPUNNPS(GPUNNPS):
    def __init__(self, int dim, list particles, double radius_scale=2.0,
            int ghost_layers=1, domain=None, bint fixed_h=False,
            bint cache=True, bint sort_gids=False, bint use_double=True,
            ctx=None):
        GPUNNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids, ctx
        )

        self.radius_scale2 = radius_scale*radius_scale

        self.use_double = use_double

        cdef str prototypes =   """
                                inline unsigned long interleave(unsigned long p, \
                                        unsigned long q, unsigned long r);

                                inline int neighbor_boxes(int c_x, int c_y, int c_z, \
                                        unsigned long* nbr_boxes, unsigned long max_key);
                                """

        cdef str bit_interleaving = """
                                    inline unsigned long interleave(unsigned long p, \
                                            unsigned long q, unsigned long r)
                                    {
                                        p = (p | (p << 32)) & 0x1f00000000ffff;
                                        p = (p | (p << 16)) & 0x1f0000ff0000ff;
                                        p = (p | (p <<  8)) & 0x100f00f00f00f00f;
                                        p = (p | (p <<  4)) & 0x10c30c30c30c30c3;
                                        p = (p | (p <<  2)) & 0x1249249249249249;

                                        q = (q | (q << 32)) & 0x1f00000000ffff;
                                        q = (q | (q << 16)) & 0x1f0000ff0000ff;
                                        q = (q | (q <<  8)) & 0x100f00f00f00f00f;
                                        q = (q | (q <<  4)) & 0x10c30c30c30c30c3;
                                        q = (q | (q <<  2)) & 0x1249249249249249;

                                        r = (r | (r << 32)) & 0x1f00000000ffff;
                                        r = (r | (r << 16)) & 0x1f0000ff0000ff;
                                        r = (r | (r <<  8)) & 0x100f00f00f00f00f;
                                        r = (r | (r <<  4)) & 0x10c30c30c30c30c3;
                                        r = (r | (r <<  2)) & 0x1249249249249249;

                                        return (p | (q << 1) | (r << 2));
                                    }
                                    """

        cdef str find_cell_id = """
                                #define FIND_CELL_ID(x, y, z, h, c_x, c_y, c_z) \
                                c_x = floor((x)/h); c_y = floor((y)/h); c_z = floor((z)/h)
                                """

        cdef str find_idx = """
                            inline int find_idx(__global unsigned long* keys, \
                                    int num_particles, unsigned long key)
                            {
                                int first = 0;
                                int last = num_particles - 1;
                                int middle = (first + last) / 2;

                                while(first <= last)
                                {
                                    if(keys[middle] < key)
                                        first = middle + 1;
                                    else if(keys[middle] > key)
                                        last = middle - 1;
                                    else if(keys[middle] == key)
                                    {
                                        if(middle == 0)
                                            return 0;
                                        if(keys[middle - 1] != key)
                                            return middle;
                                        else
                                            last = middle - 1;
                                    }
                                    middle = (first + last) / 2;
                                }

                                return -1;
                            }
                            """

        cdef str norm2 =    """
                            #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
                            """

        self.preamble = "\n".join((norm2, find_cell_id, prototypes,
                bit_interleaving, find_idx))

        self.src_index = -1
        self.dst_index = -1
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        cdef np.ndarray current_pids = (self.pids[pa_index].get()).astype(np.int64)
        indices.resize(current_pids.size)
        indices.set_data(current_pids)

    cpdef _bin(self, int pa_index):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        arguments = """
                    %(data_t)s* x, %(data_t)s* y, %(data_t)s* z, %(data_t)s cell_size,
                    %(data_t)s xmin, %(data_t)s ymin, %(data_t)s zmin,
                    unsigned long* keys, unsigned int* pids
                    """ % {"data_t" : ("double" if self.use_double else "float")}

        pids_src =  """
                    unsigned long c_x, c_y, c_z;
                    FIND_CELL_ID(
                        x[i] - xmin,
                        y[i] - ymin,
                        z[i] - zmin,
                        cell_size, c_x, c_y, c_z
                        );
                    unsigned long key;
                    key = interleave(c_x, c_y, c_z);
                    keys[i] = key;
                    pids[i] = i;
                    """

        fill_pids = ElementwiseKernel(self.ctx,
                arguments, pids_src, "fill_pids", preamble=self.preamble)

        fill_pids(pa_wrapper.gpu_x, pa_wrapper.gpu_y, pa_wrapper.gpu_z,
                self.cell_size, self.xmin[0], self.xmin[1], self.xmin[2],
                self.pid_keys[pa_index], self.pids[pa_index])

        radix_sort = cl.algorithm.RadixSort(self.ctx,
                "unsigned int* pids, unsigned long* keys",
                scan_kernel=GenericScanKernel, key_expr="keys[i]",
                sort_arg_names=["pids", "keys"])

        (sorted_indices, sorted_keys), evnt = radix_sort(self.pids[pa_index],
                self.pid_keys[pa_index], key_bits=64)
        self.pids[pa_index] = sorted_indices
        self.pid_keys[pa_index] = sorted_keys

    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles
        self.pids = []
        self.pid_keys = []

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

        self.current_keys = self.pid_keys[src_index]
        self.current_pids = self.pids[src_index]

    cdef void find_neighbor_lengths(self, nbr_lengths):
        arguments = \
                """const %(data_t)s* d_x, const %(data_t)s* d_y, const %(data_t)s* d_z,
                const %(data_t)s* d_h, const %(data_t)s* s_x, const %(data_t)s* s_y,
                const %(data_t)s* s_z, const %(data_t)s* s_h,
                %(data_t)s xmin, %(data_t)s ymin, %(data_t)s zmin,
                unsigned int num_particles, const unsigned long* keys,
                const unsigned int* pids, unsigned int* nbr_lengths,
                %(data_t)s radius_scale2, %(data_t)s cell_size
                """ % {"data_t" : ("double" if self.use_double else "float")}


        src =   """
                %(data_t)s q_x = d_x[i];
                %(data_t)s q_y = d_y[i];
                %(data_t)s q_z = d_z[i];
                %(data_t)s q_h = d_h[i];

                int c_x, c_y, c_z;

                FIND_CELL_ID(
                    q_x - xmin,
                    q_y - ymin,
                    q_z - zmin,
                    cell_size, c_x, c_y, c_z
                    );

                int idx, j, k, m;
                %(data_t)s dist;
                %(data_t)s h_i = radius_scale2*q_h*q_h;
                %(data_t)s h_j;

                unsigned long key;
                unsigned int pid;

                for(j=-1; j<2; j++)
                {
                    for(k=-1; k<2; k++)
                    {
                        for(m=-1; m<2; m++)
                        {
                            if(c_x+m >= 0 && c_y+k >= 0 && c_z+j >=0)
                            {
                                key = interleave(c_x+m, c_y+k, c_z+j);
                                idx = find_idx(keys, num_particles, key);
                                if(idx == -1)
                                    continue;

                                // FIXED: keys[idx] can go out of bounds
                                while(keys[idx] == key)
                                {
                                    pid = pids[idx];
                                    h_j = radius_scale2*s_h[pid]*s_h[pid];
                                    dist = NORM2(q_x - s_x[pid], q_y - s_y[pid], \
                                            q_z - s_z[pid]);
                                    if(dist < h_i || dist < h_j)
                                        nbr_lengths[i] += 1;
                                    idx++;

                                    if(idx == num_particles)
                                        break;
                                }
                            }
                        }
                    }
                }
                """ % {"data_t" : ("double" if self.use_double else "float")}


        z_order_nbr_lengths = ElementwiseKernel(self.ctx,
                arguments, src, "z_order_nbr_lengths", preamble=self.preamble)

        z_order_nbr_lengths(self.dst.gpu_x, self.dst.gpu_y, self.dst.gpu_z,
                self.dst.gpu_h, self.src.gpu_x, self.src.gpu_y, self.src.gpu_z,
                self.src.gpu_h, self.xmin[0], self.xmin[1], self.xmin[2],
                self.src.get_number_of_particles(), self.current_keys, self.current_pids,
                nbr_lengths, self.radius_scale2, self.cell_size)

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        arguments = \
                """const %(data_t)s* d_x, const %(data_t)s* d_y, const %(data_t)s* d_z,
                const %(data_t)s* d_h, const %(data_t)s* s_x, const %(data_t)s* s_y,
                const %(data_t)s* s_z, const %(data_t)s* s_h,
                %(data_t)s xmin, %(data_t)s ymin, %(data_t)s zmin,
                unsigned int num_particles, const unsigned long* keys,
                const unsigned int* pids, const unsigned int* start_indices,
                unsigned int* nbrs, %(data_t)s radius_scale2, %(data_t)s cell_size
                """ % {"data_t" : ("double" if self.use_double else "float")}

        src =   """
                %(data_t)s q_x = d_x[i];
                %(data_t)s q_y = d_y[i];
                %(data_t)s q_z = d_z[i];
                %(data_t)s q_h = d_h[i];

                int c_x, c_y, c_z;

                FIND_CELL_ID(
                    q_x - xmin,
                    q_y - ymin,
                    q_z - zmin,
                    cell_size, c_x, c_y, c_z
                    );

                int idx, j, k, m;
                %(data_t)s dist;
                %(data_t)s h_i = radius_scale2*q_h*q_h;
                %(data_t)s h_j;

                unsigned long key;
                unsigned int pid;

                unsigned long start_idx = (unsigned long) start_indices[i];
                unsigned long curr_idx = 0;

                for(j=-1; j<2; j++)
                {
                    for(k=-1; k<2; k++)
                    {
                        for(m=-1; m<2; m++)
                        {
                            if(c_x+m >= 0 && c_y+k >= 0 && c_z+j >=0)
                            {
                                key = interleave(c_x+m, c_y+k, c_z+j);
                                idx = find_idx(keys, num_particles, key);
                                if(idx == -1)
                                    continue;

                                while(keys[idx] == key)
                                {
                                    pid = pids[idx];
                                    h_j = radius_scale2*s_h[pid]*s_h[pid];
                                    dist = NORM2(q_x - s_x[pid], q_y - s_y[pid], \
                                            q_z - s_z[pid]);
                                    if(dist < h_i || dist < h_j)
                                    {
                                        nbrs[start_idx + curr_idx] = pid;
                                        curr_idx++;
                                    }
                                    idx++;

                                    if(idx == num_particles)
                                        break;
                                }
                            }
                        }
                    }
                }

               """ % {"data_t" : ("double" if self.use_double else "float")}


        z_order_nbrs = ElementwiseKernel(self.ctx,
                arguments, src, "z_order_nbrs", preamble=self.preamble)

        z_order_nbrs(self.dst.gpu_x, self.dst.gpu_y, self.dst.gpu_z,
                self.dst.gpu_h, self.src.gpu_x, self.src.gpu_y, self.src.gpu_z,
                self.src.gpu_h, self.xmin[0], self.xmin[1], self.xmin[2],
                self.src.get_number_of_particles(), self.current_keys, self.current_pids,
                start_indices, nbrs, self.radius_scale2, self.cell_size)


