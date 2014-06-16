from libc.math cimport *
import numpy as np


###########################################################################
cdef class CubicSpline:
    cdef public long dim
    cdef public double radius_scale
    cdef public double fac
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    cdef inline double get_deltap(self):
        return 2./3

    cpdef double py_get_deltap(self):
        return self.get_deltap()

    cdef inline void gradient(self, double* xij, double rij, double h, double* grad):
        cdef double tmp
        cdef double q
        cdef double h1
        cdef double val
        cdef double fac
        h1 = 1./h
        q = rij*h1

        fac = self.fac * h1
        if self.dim > 1:
            fac *= h1
        if self.dim > 2:
            fac *= h1

        # compute the gradient.
        if (rij > 1e-12):
            if (q >= 2.0):
                val = 0.0
            elif ( q >= 1.0 ):
                val = -0.75 * (2-q)*(2-q) * h1/rij
            else:
                val = -3.0*q * (1 - 0.75*q) * h1/rij
        else:
            val = 0.0

        tmp = val * fac
        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    cpdef py_gradient(self, double[:] xij, double rij, double h, double[:] grad):
        self.gradient(&xij[0], rij, h, &grad[0])

    cdef inline double kernel(self, double* xij, double rij, double h):
        cdef double q
        cdef double fac
        cdef double h1
        cdef double val
        h1 = 1./h
        q = rij*h1

        fac = self.fac * h1
        if self.dim > 1:
            fac *= h1
        if self.dim > 2:
            fac *= h1

        if ( q >= 2.0 ):
            val = 0.0

        elif ( q >= 1.0 ):
            val = 0.25 * ( 2-q ) * ( 2-q ) * ( 2-q )

        else:
            val = 1 - 1.5 * q * q * (1 - 0.5 * q)

        return val * fac

    cpdef double py_kernel(self, double[:] xij, double rij, double h):
        return self.kernel(&xij[0], rij, h)


cdef class CubicSplineWrapper:
    """Reasonably high-performance convenience wrapper for Kernels.
    """

    cdef public CubicSpline kern
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

###########################################################################
cdef class WendlandQuintic:
    cdef public long dim
    cdef public double radius_scale
    cdef public double fac
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    cdef inline double get_deltap(self):
        return 0.5

    cpdef double py_get_deltap(self):
        return self.get_deltap()

    cdef inline void gradient(self, double* xij, double rij, double h, double* grad):
        cdef double tmp
        cdef double q
        cdef double h1
        cdef double val
        cdef double fac
        h1 = 1./h
        q = rij*h1
        
        fac = self.fac * h1 * h1
        if self.dim > 2:
            fac *= h1

        # compute the gradient
        if (rij > 1e-12):
            if (q >= 2.0):
                val = 0.0
            else:
                val = -5 * q * (1-0.5*q) * (1-0.5*q) * (1-0.5*q) * h1/rij

        else:
            val = 0.0

        tmp = val * fac
        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    cpdef py_gradient(self, double[:] xij, double rij, double h, double[:] grad):
        self.gradient(&xij[0], rij, h, &grad[0])

    cdef inline double kernel(self, double* xij, double rij, double h):
        cdef double q
        cdef double fac
        cdef double h1
        cdef double val
        h1 = 1.0/h
        q = rij*h1
        
        fac = self.fac * h1 * h1
        if self.dim > 2:
            fac *= h1

        if ( q >= 2.0 ):
            val = 0.0

        else:
            val = (1-0.5*q) * (1-0.5*q) * (1-0.5*q) * (1-0.5*q) * (2*q + 1)

        return val * fac

    cpdef double py_kernel(self, double[:] xij, double rij, double h):
        return self.kernel(&xij[0], rij, h)


cdef class WendlandQuinticWrapper:
    """Reasonably high-performance convenience wrapper for Kernels.
    """

    cdef public WendlandQuintic kern
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

###########################################################################
cdef class Gaussian:
    cdef public long dim
    cdef public double radius_scale
    cdef public double fac
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    cdef inline double get_deltap(self):
        return 2.2360679774997898

    cpdef double py_get_deltap(self):
        return self.get_deltap()

    cdef inline void gradient(self, double* xij, double rij, double h, double* grad):
        cdef double tmp
        cdef double q
        cdef double h1
        cdef double val
        cdef double fac
        h1 = 1./h
        q = rij*h1

        fac = self.fac * h1
        if self.dim > 1:
            fac *= h1
        if self.dim > 2:
            fac *= h1

        # compute the gradient
        val = 0.0
        if (q <= 3.0):
            if (rij > 1e-12):
                val = -2 * q * exp(-q*q) * h1/rij

        tmp = val * fac
        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    cpdef py_gradient(self, double[:] xij, double rij, double h, double[:] grad):
        self.gradient(&xij[0], rij, h, &grad[0])

    cdef inline double kernel(self, double* xij, double rij, double h):
        cdef double q
        cdef double fac
        cdef double h1
        cdef double val
        h1 = 1./h
        q = rij*h1

        fac = self.fac * h1
        if self.dim > 1:
            fac *= h1
        if self.dim > 2:
            fac *= h1

        val = 0.0
        if ( q <= 3.0 ):
            val = exp(-q*q) * fac

        return val

    cpdef double py_kernel(self, double[:] xij, double rij, double h):
        return self.kernel(&xij[0], rij, h)


cdef class GaussianWrapper:
    """Reasonably high-performance convenience wrapper for Kernels.
    """

    cdef public Gaussian kern
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

###########################################################################
cdef class QuinticSpline:
    cdef public long dim
    cdef public double radius_scale
    cdef public double fac
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    cdef inline double get_deltap(self):
        return 1.0

    cpdef double py_get_deltap(self):
        return self.get_deltap()

    cdef inline void gradient(self, double* xij, double rij, double h, double* grad):
        cdef double tmp
        cdef double q
        cdef double h1
        cdef double val
        cdef double fac
        h1 = 1./h
        q = rij*h1

        fac = self.fac * h1 * h1

        # compute the gradient
        if (rij > 1e-12):
            if ( q > 3.0 ):
                val = 0.0

            elif ( q > 2.0 ):
                val = -5.0 * (3.0 - q)**4 * h1/rij

            elif ( q > 1.0 ):
                val = (-5.0 * (3.0 - q)**4 + 30.0 * (2.0 - q)**4) * h1/rij

            else:
                val = (-5.0 * (3.0 - q)**4 + 30.0 * (2.0 - q)**4 - 75.0 * (1.0 - q)**4) * h1/rij

        else:
            val = 0.0

        tmp = val * fac
        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    cpdef py_gradient(self, double[:] xij, double rij, double h, double[:] grad):
        self.gradient(&xij[0], rij, h, &grad[0])

    cdef inline double kernel(self, double* xij, double rij, double h):
        cdef double q
        cdef double fac
        cdef double h1
        cdef double val
        h1 = 1./h
        q = rij*h1

        fac = self.fac * h1 * h1

        if ( q > 3.0 ):
            val = 0.0

        elif ( q > 2.0 ):
            val = (3.0-q)**5

        elif ( q > 1.0 ):
            val = (3.0-q)**5 - 6.0*(2.0-q)**5

        else:
            val = (3.0-q)**5 - 6*(2.0-q)**5 + 15.0*(1.0-q)**5

        return val * fac

    cpdef double py_kernel(self, double[:] xij, double rij, double h):
        return self.kernel(&xij[0], rij, h)


cdef class QuinticSplineWrapper:
    """Reasonably high-performance convenience wrapper for Kernels.
    """

    cdef public QuinticSpline kern
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

