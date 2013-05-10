"""Common integrators for 2D SPH"""

from pysph.base.particle_array cimport ParticleArray
from pysph.base.nnps cimport NNPS

# PyZoltan
from pyzoltan.core.carray cimport IntArray, DoubleArray, LongArray

cdef class WCSPHRK2Integrator:
    ############################################################
    # Data attributes
    ############################################################
    cdef public list particles
    cdef public object evaluator
    cdef public NNPS nnps

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

    # set the values at the start of a time step
    cdef _set_initial_values(self)

    # reset accelerations before computing 
    cdef _reset_accelerations(self)
