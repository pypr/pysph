#cython: embedsignature=True
<%
from cython_generator import CythonGenerator
from kernels import CubicSpline, WendlandQuintic, Gaussian, QuinticSpline
generator = CythonGenerator(python_methods=True)
%>

from libc.math cimport *
import numpy as np

% for cls in (CubicSpline, WendlandQuintic, Gaussian, QuinticSpline):
<%
generator.parse(cls())
classname = cls.__name__
%>
${generator.get_code()}

cdef class ${classname}Wrapper:
    """Reasonably high-performance convenience wrapper for Kernels.
    """

    cdef public ${classname} kern
    cdef double[3] xij, grad
    cdef public double radius_scale
    cdef public double fac

    def __init__(self, kern):
        self.kern = kern
        self.radius_scale = kern.radius_scale
        self.fac = kern.fac

    cpdef double kernel(self, double xi, double yi, double zi, double xj, double yj, double zj, double h):
        cdef double* xij = self.xij
        xij[0] = xi-xj
        xij[1] = yi-yj
        xij[2] = zi-zj
        cdef double rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1] +xij[2]*xij[2])
        return self.kern.kernel(xij, rij, h)

    cpdef gradient(self, double xi, double yi, double zi, double xj, double yj, double zj, double h):
        cdef double* xij = self.xij
        xij[0] = xi-xj
        xij[1] = yi-yj
        xij[2] = zi-zj
        cdef double rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1] +xij[2]*xij[2])
        cdef double* grad = self.grad
        self.kern.gradient(xij, rij, h, grad)
        return grad[0], grad[1], grad[2]

% endfor

