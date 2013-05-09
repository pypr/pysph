cimport numpy
import numpy

# local imports
from point cimport cPoint_new, cPoint_distance, cPoint_sub, Point

cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double fabs(double)
    double sin(double)
    double cos(double)
    double pow(double x, double y)
    int floor(double)
    double fmod(double, double)

cdef:
    double PI = numpy.pi
    double PIb1 = 1.0/PI
    double SQRT_1_PI = 1.0/sqrt(PI)
    double infty = numpy.inf    

cdef inline double h_dim(double h, int dim):
    cdef double h1 = 1.0/h
    if dim == 1:
        return h1
    elif dim == 2:
        return h1*h1
    else:
        return h1*h1*h1

class AveragingType:
    NoAveraging = 0
    HAveraging = 1
    KAveraging = 2
    SHAveraging = 3    

cdef class Kernel:
    cdef double function(self, cPoint xi, cPoint xj, double h):
        cdef double rij = cPoint_distance(xi, xj)
        cdef double fac = h_dim(h, self.dim) * self.fac
        cdef double q = rij/h

        cdef double ret = fac * self._function(q)
        return ret

    cdef gradient(self, cPoint xi, cPoint xj, double h, cPoint grad):
        cdef double rij = cPoint_distance(xi, xj)
        cdef double fac = h_dim(h, self.dim) * self.fac
        cdef double q = rij/h
        cdef double val = 0.0

        grad.x = xi.x - xj.x
        grad.y = xi.y - xj.y
        grad.z = xi.z - xj.z

        if ( rij > 1e-8 ):
            val = fac * self._gradient(q)/(h*rij)
        else:
            val = 0.0

        grad.x = grad.x * val
        grad.y = grad.y * val
        grad.z = grad.z * val
        
    cdef double gradient_h(self, cPoint xi, cPoint xj, double h):
        cdef double rij = cPoint_distance(xi, xj)
        cdef double fac = h_dim(h, self.dim) * self.fac
        cdef double q = rij/h

        cdef double w = self._function(q)
        cdef double dw = self._gradient(q)
        cdef int dim = self.dim

        return -fac*dw*q/h - fac*w*dim/h

    cdef double _function(self, double q):
        raise NotImplementedError("Kernel::_function")

    cdef double _gradient(self, double q):
        raise NotImplementedError("Kernel::_gradient")

    #####################################################
    # Python wrappers
    #####################################################
    def py_function(self, Point xi, Point xj, double h):
        return self.function(xi.data, xj.data, h)

    def py_gradient(self, Point xi, Point xj, double h):
        ret = Point()
        self.gradient(xi.data, xj.data, h, ret.data)
        return ret

cdef class CubicSpline(Kernel):
    def __init__(self, int dim):
        self.dim = dim

        if dim == 1:
            self.fac = 2.0/3.0
        elif dim == 2:
            self.fac = 10*PIb1/7.0
        else:
            self.fac = PIb1
        
        self.radius = 2.0

    cdef double _function(self, double q):
        if q >= 2.0:
            return 0.0
        elif q >= 1.0:
            return 0.25 * (2-q)**3
        else:
            return 1 - 1.5*q*q * (1 - 0.5*q)
        
    cdef double _gradient(self, double q):
        if q >= 2.0:
            return 0
        elif q >= 1.0:
            return -0.75 * (2-q)**2
        else:
            return -3*q * (1 - 0.75*q)

cdef class Gaussian(Kernel):
    def __init__(self, int dim):
        self.dim = dim

        if dim == 1:
            self.fac = SQRT_1_PI
        elif dim == 2:
            self.fac = SQRT_1_PI**2
        else:
            self.fac = SQRT_1_PI**3
        
        self.radius = 3.1

    cdef double _function(self, double q):
        if ( q >= 3.0 ):
            return 0
        else:
            return exp(-q*q)

    cdef double _gradient(self, double q):
        return -2*q*self._function(q)
