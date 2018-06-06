import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
from pytools import memoize
from pysph.base.opencl import get_context, get_queue

_cache = {}


@memoize
def get_copy_kernel(ctx, dtype1, dtype2):
    return ElementwiseKernel(
        ctx,
        """
        %(data_t1)s *x1, %(data_t1)s *y1, %(data_t1)s *z1, %(data_t1)s *h1,
        %(data_t2)s *x2, %(data_t2)s *y2, %(data_t2)s *z2, %(data_t2)s *h2
        """ % dict(
            data_t1=dtype1,
            data_t2=dtype2
        ),
        operation="""
        x2[i] = (%(data_t2)s)x1[i];
        y2[i] = (%(data_t2)s)y1[i];
        z2[i] = (%(data_t2)s)z1[i];
        h2[i] = (%(data_t2)s)h1[i];
        """ % dict(
            data_t2=dtype2
        )
    )


def cache_result(cache_key):
    global _cache

    def _decorator(f):
        def _cached_f(*args):
            key = (cache_key, *args)
            if key not in _cache:
                _cache[key] = f(*args)
            return _cache[key]

        return _cached_f

    return _decorator


c2d = {
    'half': np.float16,
    'float': np.float32,
    'double': np.float64
}


def ctype_to_dtype(ctype):
    return c2d[ctype]


class GPUParticleArrayWrapper(object):
    def __init__(self, pa_gpu, c_type_src, c_type, force_copy=True):
        self.c_type = c_type
        self.c_type_src = c_type_src
        if c_type != c_type_src or force_copy:
            self.is_copy = True
        else:
            self.is_copy = False

        self._initialize(pa_gpu)
        self.force_sync(pa_gpu)

    def _gpu_copy(self, pa_gpu):
        copy_kernel = get_copy_kernel(get_context(),
                                      self.c_type_src,
                                      self.c_type)
        copy_kernel(pa_gpu.x, pa_gpu.y, pa_gpu.z, pa_gpu.h,
                    self.x, self.y, self.z, self.h)

    def _initialize(self, pa_gpu):
        if not self.is_copy:
            self.x = pa_gpu.x
            self.y = pa_gpu.y
            self.z = pa_gpu.z
            self.h = pa_gpu.h
        else:
            self.x = cl.array.zeros(get_queue(), pa_gpu.x.shape,
                                    ctype_to_dtype(self.c_type))
            self.y = cl.array.zeros(get_queue(), pa_gpu.x.shape,
                                    ctype_to_dtype(self.c_type))
            self.z = cl.array.zeros(get_queue(), pa_gpu.x.shape,
                                    ctype_to_dtype(self.c_type))
            self.h = cl.array.zeros(get_queue(), pa_gpu.x.shape,
                                    ctype_to_dtype(self.c_type))
        self.prev_id = -1

    def _gpu_sync(self, pa_gpu):
        if self.x.shape != pa_gpu.x.shape:
            self._initialize(pa_gpu)
        self._gpu_copy(pa_gpu)

    def _ref_sync(self, pa_gpu):
        self.x = pa_gpu.x
        self.y = pa_gpu.y
        self.z = pa_gpu.z
        self.h = pa_gpu.h

    def _sync(self, pa_gpu):
        if self.is_copy:
            self._gpu_sync(pa_gpu)
        else:
            self._ref_sync(pa_gpu)
        self.prev_id = id(pa_gpu)

    def sync(self, pa_gpu):
        # No syncing required
        if id(pa_gpu.x) == self.prev_id:
            return
        self.force_sync(pa_gpu)

    def force_sync(self, pa_gpu):
        self._sync(pa_gpu)


class ParticleArrayWrapper(object):
    """A loose wrapper over Particle Array

    Objective is to transparently maintain a copy of
    the original particle array's position properties
    (x, y, z, h)
    """

    def __init__(self, pa, c_type_src, c_type='float'):
        self._pa = pa
        self._gpu = GPUParticleArrayWrapper(pa.gpu, c_type_src,
                                            c_type)

    def get_number_of_particles(self):
        return self._pa.get_number_of_particles()

    @property
    def gpu(self):
        return self._gpu

    def sync(self):
        self._gpu.sync(self._pa.gpu)

    def force_sync(self):
        self._gpu.force_sync(self._pa.gpu)
