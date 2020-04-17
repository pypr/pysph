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
from pyopencl.elementwise import ElementwiseKernel

from pysph.base.nnps_base cimport *
from pysph.base.device_helper import DeviceHelper
from compyle.config import get_config
from compyle.array import get_backend, Array
from compyle.parallel import Elementwise, Scan
from compyle.types import annotate
from compyle.opencl import (get_context, get_queue,
                            set_context, set_queue)
import compyle.array as array

# Particle Tag information
from cyarray.carray cimport BaseArray, aligned_malloc, aligned_free
from utils import ParticleTAGS

from nnps_base cimport *

from pysph.base.gpu_helper_kernels import (exclusive_input, exclusive_output,
                                           get_scan)


cdef class GPUNeighborCache:
    def __init__(self, GPUNNPS nnps, int dst_index, int src_index,
            backend=None):
        self.backend = get_backend(backend)
        self._dst_index = dst_index
        self._src_index = src_index
        self._nnps = nnps
        self._particles = nnps.particles
        self._narrays = nnps.narrays
        cdef long n_p = self._particles[dst_index].get_number_of_particles()

        self._get_start_indices = None
        self._cached = False
        self._copied_to_cpu = False

        self._nbr_lengths_gpu = Array(np.uint32, n=n_p,
                                      backend=self.backend)

        self._neighbors_gpu = Array(np.uint32,
                                    backend=self.backend)

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
        # FIXME:
        # - Store sum kernel
        # - don't allocate neighbors_gpu each time.
        # - Don't allocate _nbr_lengths and start_idx.

        total_size_gpu = array.sum(self._nbr_lengths_gpu)

        cdef unsigned long total_size = <unsigned long>(total_size_gpu)

        # Allocate _neighbors_cpu and neighbors_gpu
        self._neighbors_gpu.resize(total_size)

        self._start_idx_gpu = self._nbr_lengths_gpu.copy()

        # Do prefix sum on self._neighbor_lengths for the self._start_idx
        if self._get_start_indices is None:
            self._get_start_indices = get_scan(exclusive_input, exclusive_output,
                    dtype=np.uint32, backend=self.backend)

        self._get_start_indices(ary=self._start_idx_gpu)

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
        # FIXME: Don't allocate here unless needed.
        self._cached = False
        self._copied_to_cpu = False
        cdef long n_p = self._particles[self._dst_index].get_number_of_particles()
        self._nbr_lengths_gpu.resize(n_p)

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
                 bint sort_gids=False, backend=None):
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

        backend : string
            Backend on which to build NNPS Module
        """
        NNPSBase.__init__(self, dim, particles, radius_scale, ghost_layers,
                domain, cache, sort_gids)

        self.backend = get_backend(backend)
        self.backend = 'opencl' if self.backend is 'cython' else self.backend
        self.use_double = get_config().use_double
        self.dtype = np.float64 if self.use_double else np.float32
        self.dtype_max = np.finfo(self.dtype).max
        self._last_domain_size = 0.0

        # Set the device helper if needed.
        for pa in particles:
            if pa.gpu is None:
                pa.set_device_helper(DeviceHelper(pa, backend=self.backend))

        # The cache.
        self.use_cache = cache
        _cache = []
        for d_idx in range(len(particles)):
            for s_idx in range(len(particles)):
                _cache.append(GPUNeighborCache(self, d_idx, s_idx,
                    backend=self.backend))
        self.cache = _cache
        self.use_double = get_config().use_double

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
        indices, callback = self.get_spatially_ordered_indices(pa_index)
        self.particles[pa_index].gpu.align(indices)
        callback()

    def set_use_cache(self, bint use_cache):
        self.use_cache = use_cache
        if use_cache:
            for cache in self.cache:
                cache.update()

    cdef void find_neighbor_lengths(self, nbr_lengths):
        raise NotImplementedError("NNPS :: find_neighbor_lengths called")

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        raise NotImplementedError("NNPS :: find_nearest_neighbors called")

    cpdef update(self):
        cdef int i, num_particles
        cdef ParticleArray pa

        cdef DomainManager domain = self.domain

        # use cell sizes computed by the domain.
        self.cell_size = domain.manager.cell_size
        self.hmin = domain.manager.hmin

        # compute bounds and refresh the data structure
        self._compute_bounds()
        self._refresh()

        # indices on which to bin. We bin all local particles
        for i in range(self.narrays):
            # bin the particles
            self._bin(pa_index=i)

        if self.use_cache:
            for cache in self.cache:
                cache.update()

    def update_domain(self):
        self.domain.update()

    cdef _compute_bounds(self):
        """Compute coordinate bounds for the particles"""
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef double domain_size
        xmax = -self.dtype_max
        ymax = -self.dtype_max
        zmax = -self.dtype_max

        xmin = self.dtype_max
        ymin = self.dtype_max
        zmin = self.dtype_max

        for pa_wrapper in pa_wrappers:
            x = pa_wrapper.pa.gpu.get_device_array('x')
            y = pa_wrapper.pa.gpu.get_device_array('y')
            z = pa_wrapper.pa.gpu.get_device_array('z')

            pa_wrapper.pa.gpu.update_minmax_cl(['x', 'y', 'z'])

            # find min and max of variables
            xmax = np.maximum(x.maximum, xmax)
            ymax = np.maximum(y.maximum, ymax)
            zmax = np.maximum(z.maximum, zmax)

            xmin = np.minimum(x.minimum, xmin)
            ymin = np.minimum(y.minimum, ymin)
            zmin = np.minimum(z.minimum, zmin)

        # Add a small offset to the limits.
        lx, ly, lz = xmax - xmin, ymax - ymin, zmax - zmin
        xmin -= lx*0.01; ymin -= ly*0.01; zmin -= lz*0.01
        xmax += lx*0.01; ymax += ly*0.01; zmax += lz*0.01

        domain_size = fmax(lx, ly)
        domain_size = fmax(domain_size, lz)
        if self._last_domain_size > 1e-16 and \
           domain_size > 2.0*self._last_domain_size:
            msg = (
                '*'*70 +
                '\nWARNING: Domain size has increased by a large amount.\n' +
                'Particles are probably diverging, please check your code!\n' +
                '*'*70
            )
            print(msg)
            self._last_domain_size = domain_size

        # If all of the dimensions have very small extent give it a unit size.
        _eps = 1e-12
        if (np.abs(xmax - xmin) < _eps) and (np.abs(ymax - ymin) < _eps) \
            and (np.abs(zmax - zmin) < _eps):
            xmin -= 0.5; xmax += 0.5
            ymin -= 0.5; ymax += 0.5
            zmin -= 0.5; zmax += 0.5

        # store the minimum and maximum of physical coordinates
        self.xmin = np.asarray([xmin, ymin, zmin])
        self.xmax = np.asarray([xmax, ymax, zmax])

    cpdef _bin(self, int pa_index):
        raise NotImplementedError("NNPS :: _bin called")

    cpdef _refresh(self):
        raise NotImplementedError("NNPS :: _refresh called")


cdef class BruteForceNNPS(GPUNNPS):
    def __init__(self, int dim, list particles, double radius_scale=2.0,
            int ghost_layers=1, domain=None, bint cache=True,
            bint sort_gids=False, backend='opencl'):
        GPUNNPS.__init__(self, dim, particles, radius_scale, ghost_layers,
                domain, cache, sort_gids, backend)

        self.radius_scale2 = radius_scale*radius_scale
        self.src_index = -1
        self.dst_index = -1
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

        cdef str norm2 = \
                """
                #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
                """

        self.preamble = norm2

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
        # IMPORTANT NOTE: pyopencl uses the length of the first argument
        # to determine the global work size
        arguments = \
                """%(data_t)s* d_x, %(data_t)s* d_y, %(data_t)s* d_z,
                %(data_t)s* d_h, %(data_t)s* s_x, %(data_t)s* s_y,
                %(data_t)s* s_z, %(data_t)s* s_h, unsigned int num_particles,
                unsigned int* nbr_lengths, %(data_t)s radius_scale2
                """ % {"data_t" : ("double" if self.use_double else "float")}

        src = """
                unsigned int j;
                unsigned int length = 0;
                %(data_t)s dist;
                %(data_t)s h_i = radius_scale2*d_h[i]*d_h[i];
                %(data_t)s h_j;
                for(j=0; j<num_particles; j++)
                {
                    h_j = radius_scale2*s_h[j]*s_h[j];
                    dist = NORM2(d_x[i] - s_x[j], d_y[i] - s_y[j], d_z[i] - s_z[j]);
                    if(dist < h_i || dist < h_j)
                        length += 1;
                }
                nbr_lengths[i] = length;
                """ % {"data_t" : ("double" if self.use_double else "float")}

        brute_force_nbr_lengths = ElementwiseKernel(
            get_context(), arguments, src, "brute_force_nbr_lengths",
            preamble=self.preamble
        )

        src_gpu = self.src.pa.gpu
        dst_gpu = self.dst.pa.gpu
        brute_force_nbr_lengths(dst_gpu.x.dev, dst_gpu.y.dev, dst_gpu.z.dev,
                dst_gpu.h.dev, src_gpu.x.dev, src_gpu.y.dev, src_gpu.z.dev,
                src_gpu.h.dev, self.src.get_number_of_particles(),
                nbr_lengths.dev, self.radius_scale2)

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):

        arguments = \
                """%(data_t)s* d_x, %(data_t)s* d_y, %(data_t)s* d_z,
                %(data_t)s* d_h, %(data_t)s* s_x, %(data_t)s* s_y,
                %(data_t)s* s_z, %(data_t)s* s_h, unsigned int num_particles,
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

        brute_force_nbrs = ElementwiseKernel(
            get_context(), arguments, src, "brute_force_nbrs",
            preamble=self.preamble
        )

        src_gpu = self.src.pa.gpu
        dst_gpu = self.dst.pa.gpu

        brute_force_nbrs(dst_gpu.x.dev, dst_gpu.y.dev, dst_gpu.z.dev,
                dst_gpu.h.dev, src_gpu.x.dev, src_gpu.y.dev, src_gpu.z.dev,
                src_gpu.h.dev, self.src.get_number_of_particles(),
                start_indices.dev, nbrs.dev, self.radius_scale2)

    cpdef _bin(self, int pa_index):
        pass

    cpdef _refresh(self):
        pass
