"""Common integrators for 2D SPH"""

from pysph.base.carray cimport IntArray, DoubleArray, LongArray
from pysph.base.particle_array cimport ParticleArray

cdef class WCSPHRK2Integrator:
    ############################################################
    # Data attributes
    ############################################################
    cdef public list particles
    cdef public object evaluator

    # particle properties used for time stepping
    cdef DoubleArray x, x0, ax
    cdef DoubleArray y, y0, ay
    cdef DoubleArray z, z0, az
    cdef DoubleArray u, u0, au
    cdef DoubleArray v, v0, av
    cdef DoubleArray w, w0, aw
    cdef DoubleArray rho, rho0, arho

    ############################################################
    # Member functions
    ############################################################
    cpdef integrate(self, double dt)
