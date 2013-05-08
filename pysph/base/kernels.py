from textwrap import dedent

###############################################################################
# `CubicSpline` class.
###############################################################################
class CubicSpline(object):
    def __init__(self, dim=1):
        self.dim = dim
    def cython_code(self):
        code = dedent('''\
        from libc.math cimport fabs, sqrt, M_1_PI
            
        cdef inline double CubicSplineKernel(double xij, double yij, double zij, double h):
            cdef double rij = sqrt( xij*xij + yij*yij + zij*zij )

            cdef double h1 = 1./h
            cdef double q = rij*h1
            cdef double val = 0.0
            cdef double fac

            if DIM == 3:
                fac = M_1_PI * h1 * h1 * h1

            elif DIM == 2:
                fac = 10*M_1_PI/7.0 * h1 * h1

            else:
                fac = 2./3 * h1

            if ( q >= 2.0 ):
                val = 0.0

            elif ( q >= 1.0 ):
                val = 0.25 * ( 2-q ) * ( 2-q ) * ( 2-q )

            else:
                val = 1 - 1.5 * q * q * (1 - 0.5 * q)

            return val * fac


        cdef inline CubicSplineGradient(double xij, double yij, double zij, double h,
                                        double* grad):
            cdef double rij = sqrt( xij*xij + yij*yij + zij*zij )

            cdef double h1 = 1./h
            cdef double q = rij*h1
            cdef double val = 0.0, fac, tmp

            if DIM == 3:
                fac = M_1_PI * h1 * h1 * h1

            elif DIM == 2:
                fac = 10*M_1_PI/7.0 * h1 * h1

            else:
                fac = 2./3 * h1

            # compute the gradient
            if (rij > 1e-8):
                if (q >= 2.0):
                    val = 0.0
                elif ( q >= 1.0 ):
                    val = -0.75 * (2-q)*(2-q) * h1/rij
                else:
                    val = -3.0*q * (1 - 0.75*q) * h1/rij
            else:
                val = 0.0

            tmp = val * fac
            grad[0] = tmp * xij
            grad[1] = tmp * yij
            grad[2] = tmp * zij
            
        '''.replace("DIM", str(self.dim)))
        return dict(helper=code)


