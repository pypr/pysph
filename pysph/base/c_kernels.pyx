#cython: embedsignature=True


from libc.math cimport *
import numpy as np



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
        cdef double val
        cdef double h1
        cdef double q
        cdef double fac
        cdef double tmp2
        h1 = 1./h
        q = rij*h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        # compute the gradient.
        tmp2 = 2. - q
        if (rij > 1e-12):
            if (q > 2.0):
                val = 0.0
            elif ( q > 1.0 ):
                val = -0.75 * tmp2*tmp2 * h1/rij
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

    cdef inline double gradient_h(self, double* xij, double rij, double h):
        cdef double h1
        cdef double q
        cdef double w
        cdef double fac
        cdef double dw
        cdef double tmp2
        h1 = 1./h; q = rij * h1
        
        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        # kernel and gradient evaluated at q
        tmp2 = 2. - q
        if ( q > 2.0 ):
            w = 0.0
            dw = 0.0
            
        elif ( q > 1.0 ):
            w = 0.25 * tmp2 * tmp2 * tmp2
            dw = -0.75 * tmp2 * tmp2
        else:
            w = 1 - 1.5 * q * q * (1 - 0.5 * q)
            dw = -3.0*q * (1 - 0.75*q)

        return -fac * h1 * ( dw*q + w*self.dim )

    cpdef double py_gradient_h(self, double[:] xij, double rij, double h):
        return self.gradient_h(&xij[0], rij, h)

    cdef inline double kernel(self, double* xij, double rij, double h):
        cdef double q
        cdef double fac
        cdef double h1
        cdef double val
        cdef double tmp2
        h1 = 1./h
        q = rij*h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1
        
        tmp2 = 2. - q
        if ( q > 2.0 ):
            val = 0.0

        elif ( q > 1.0 ):
            val = 0.25 * tmp2 * tmp2 * tmp2
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
        
        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        # compute the gradient
        val = 0.0
        tmp = 1.0 - 0.5*q
        if ( q < 2.0 ):
            if (rij > 1e-12):
                val = -5.0 * q * tmp * tmp * tmp * h1/rij

        tmp = val * fac
        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    cpdef py_gradient(self, double[:] xij, double rij, double h, double[:] grad):
        self.gradient(&xij[0], rij, h, &grad[0])

    cdef inline double gradient_h(self, double* xij, double rij, double h):
        cdef double tmp
        cdef double h1
        cdef double q
        cdef double w
        cdef double fac
        cdef double dw
        h1 = 1./h; q = rij*h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        # compute the kernel and gradient at q
        w = 0.0; dw = 0.0
        tmp = 1.0 - 0.5*q
        if ( q < 2.0 ):
            w = tmp * tmp * tmp * tmp * (2.0*q + 1.0)
            dw = -5.0 * q * tmp * tmp * tmp

        return -fac * h1 * ( dw*q + w*self.dim )

    cpdef double py_gradient_h(self, double[:] xij, double rij, double h):
        return self.gradient_h(&xij[0], rij, h)

    cdef inline double kernel(self, double* xij, double rij, double h):
        cdef double q
        cdef double fac
        cdef double h1
        cdef double val
        cdef double tmp
        h1 = 1.0/h
        q = rij*h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        val = 0.0

        tmp = 1. - 0.5 * q
        if ( q < 2.0 ):
            val = tmp * tmp * tmp * tmp * (2.0*q + 1.0)

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



cdef class Gaussian:
    cdef public long dim
    cdef public double radius_scale
    cdef public double fac
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    cdef inline double get_deltap(self):
        # The inflection point is at q=1/sqrt(2)
        # the deltap values for some standard kernels
        # have been tabulated in sec 3.2 of
        # http://cfd.mace.manchester.ac.uk/sph/SPH_PhDs/2008/crespo_thesis.pdf
        return 0.70710678118654746

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

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        # compute the gradient
        val = 0.0
        if (q < 3.0):
            if (rij > 1e-12):
                val = -2.0* q * exp(-q*q) * h1/rij

        tmp = val * fac
        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    cpdef py_gradient(self, double[:] xij, double rij, double h, double[:] grad):
        self.gradient(&xij[0], rij, h, &grad[0])

    cdef inline double gradient_h(self, double* xij, double rij, double h):
        cdef double q
        cdef double fac
        cdef double h1
        cdef double w
        cdef double dw
        h1 = 1./h; q= rij*h1
        
        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        # kernel and gradient evaluated at q
        w = 0.0; dw = 0.0
        if ( q < 3.0 ):
            w = exp(-q*q)
            dw = -2.0 * q * w
            
        return -fac * h1 * ( dw*q + w*self.dim )

    cpdef double py_gradient_h(self, double[:] xij, double rij, double h):
        return self.gradient_h(&xij[0], rij, h)

    cdef inline double kernel(self, double* xij, double rij, double h):
        cdef double q
        cdef double fac
        cdef double h1
        cdef double val
        h1 = 1./h
        q = rij*h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1
        
        val = 0.0
        if ( q < 3.0 ):
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



cdef class QuinticSpline:
    cdef public long dim
    cdef public double radius_scale
    cdef public double fac
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    cdef inline double get_deltap(self):
        # The inflection points for the polynomial are obtained as
        # http://www.wolframalpha.com/input/?i=%28%283-x%29%5E5+-+6*%282-x%29%5E5+%2B+15*%281-x%29%5E5%29%27%27
        # the only permissible value is taken
        return 0.759298480738450

    cpdef double py_get_deltap(self):
        return self.get_deltap()

    cdef inline void gradient(self, double* xij, double rij, double h, double* grad):
        cdef double tmp
        cdef double val
        cdef double h1
        cdef double q
        cdef double fac
        cdef double tmp1
        cdef double tmp3
        cdef double tmp2
        h1 = 1./h
        q = rij*h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        tmp3 = 3. - q
        tmp2 = 2. - q
        tmp1 = 1. - q
        
        # compute the gradient
        if (rij > 1e-12):
            if ( q > 3.0 ):
                val = 0.0

            elif ( q > 2.0 ):
                val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
                val *= h1/rij

            elif ( q > 1.0 ):
                val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
                val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
                val *= h1/rij
            else:
                val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
                val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
                val -= 75.0 * tmp1 * tmp1 * tmp1 * tmp1
                val *= h1/rij
        else:
            val = 0.0

        tmp = val * fac
        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    cpdef py_gradient(self, double[:] xij, double rij, double h, double[:] grad):
        self.gradient(&xij[0], rij, h, &grad[0])

    cdef inline double gradient_h(self, double* xij, double rij, double h):
        cdef double h1
        cdef double q
        cdef double w
        cdef double fac
        cdef double dw
        cdef double tmp1
        cdef double tmp3
        cdef double tmp2
        h1 = 1./h; q = rij*h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        tmp3 = 3. - q
        tmp2 = 2. - q
        tmp1 = 1. - q
        
        # compute the kernel & gradient at q
        if ( q > 3.0 ):
            w = 0.0
            dw = 0.0

        elif ( q > 2.0 ):
            w = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
            dw = -5.0 * tmp3 * tmp3 * tmp3 * tmp3

        elif ( q > 1.0 ):
            w = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
            w -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2

            dw = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
            dw += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
        else:
            w = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
            w -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2
            w += 15. * tmp1 * tmp1 * tmp1 * tmp1 * tmp1

            dw = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
            dw += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
            dw -= 75.0 * tmp1 * tmp1 * tmp1 * tmp1

        return -fac * h1 * ( dw*q + w*self.dim )

    cpdef double py_gradient_h(self, double[:] xij, double rij, double h):
        return self.gradient_h(&xij[0], rij, h)

    cdef inline double kernel(self, double* xij, double rij, double h):
        cdef double val
        cdef double h1
        cdef double q
        cdef double fac
        cdef double tmp1
        cdef double tmp3
        cdef double tmp2
        h1 = 1./h
        q = rij*h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        tmp3 = 3. - q
        tmp2 = 2. - q
        tmp1 = 1. - q
        if ( q > 3.0 ):
            val = 0.0

        elif ( q > 2.0 ):
            val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3

        elif ( q > 1.0 ):
            val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
            val -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2

        else:
            val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
            val -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2
            val += 15. * tmp1 * tmp1 * tmp1 * tmp1 * tmp1

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


