from textwrap import dedent
from math import pi, sqrt, exp

M_1_PI = 1.0/pi
M_2_SQRTPI = 2.0/sqrt(pi)

def get_correction(kernel, h0):
    rij = kernel.deltap * h0
    return kernel.kernel(rij=rij, h=h0)

###############################################################################
# `CubicSpline` class.
###############################################################################
class CubicSpline(object):
    def __init__(self, dim=1):
        self.radius_scale = 2.0
        self.dim = dim

    def get_deltap(self):
        return 2./3

    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):
        h1 = 1./h
        q = rij*h1

        if self.dim == 3:
            fac = M_1_PI * h1 * h1 * h1

        elif self.dim == 2:
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

    def gradient(self, xij=[0., 0, 0], rij=1.0, h=1.0, grad=[0, 0, 0]):
        h1 = 1./h
        q = rij*h1

        if self.dim == 3:
            fac = M_1_PI * h1 * h1 * h1

        elif self.dim == 2:
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
        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    def cython_code(self):
        code = dedent('''
        from libc.math cimport M_1_PI
        ''')
        return dict(helper=code)


class WendlandQuintic(object):
    def __init__(self, dim=2):
        self.radius_scale = 2.0
        if dim == 1:
            raise ValueError("WendlandQuintic: Dim %d not supported"%dim)
        self.dim = dim

    def get_deltap(self):
        return 0.5

    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):
        h1 = 1.0/h
        q = rij*h1

        if self.dim == 3:
            fac = M_1_PI * h1 * h1 * h1 * 21.0/16.0

        elif self.dim == 2:
            fac = 7.0*M_1_PI/4.0 * h1 * h1

        else:
            fac = 0.0

        if ( q >= 2.0 ):
            val = 0.0

        else:
            val = (1-0.5*q) * (1-0.5*q) * (1-0.5*q) * (1-0.5*q) * (2*q + 1)

        return val * fac


    def gradient(self, xij=[0., 0, 0], rij=1.0, h=1.0, grad=[0, 0, 0]):
        h1 = 1./h
        q = rij*h1

        if self.dim == 3:
            fac = M_1_PI * h1 * h1 * h1 * 21.0/16.0

        elif self.dim == 2:
            fac = 7.0*M_1_PI/4.0 * h1 * h1

        else:
            fac = 0.0

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

    def cython_code(self):
        code = dedent('''
        from libc.math cimport M_1_PI
        ''')
        return dict(helper=code)

class Gaussian(object):
    def __init__(self, dim=2):
        self.radius_scale = 3.0
        self.dim = dim

    def get_deltap(self):
        return sqrt(0.5)

    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):
        h1 = 1./h
        q = rij*h1

        fac = (0.5 * M_2_SQRTPI * h1)**self.dim

        if ( q >= 3.0 ):
            val = 0.0

        else:
            val = exp(-q*q)

        return val * fac

    def gradient(self, xij=[0., 0, 0], rij=1.0, h=1.0, grad=[0., 0, 0]):
        h1 = 1./h
        q = rij*h1

        fac = (0.5 * M_2_SQRTPI * h1)**self.dim

        # compute the gradient
        if (rij > 1e-12):
            if (q >= 3.0):
                val = 0.0
            else:
                val = -2 * q * exp(-q*q) * h1/rij

        else:
            val = 0.0

        tmp = val * fac
        grad[0] = tmp * xij[0]
        grad[1] = tmp * xij[1]
        grad[2] = tmp * xij[2]

    def cython_code(self):
        code = dedent('''
        from libc.math cimport exp, M_2_SQRTPI
        ''')
        return dict(helper=code)

class QuinticSpline(object):
    def __init__(self, dim=2):
        self.radius_scale = 3.0
        if dim != 2:
            raise NotImplementedError('Quintic spline currently only supports 2D kernels.')
        self.dim = dim

    # this is incorrect for the moment and needs to be calculated
    def get_deltap(self):
        return 1.0

    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):
        h1 = 1./h
        q = rij*h1

        if self.dim == 2:
            fac = M_1_PI * 7./478.0 * h1 * h1

        else:
            fac = 0.0

        if ( q > 3.0 ):
            val = 0.0

        elif ( q > 2.0 ):
            val = (3.0-q)**5

        elif ( q > 1.0 ):
            val = (3.0-q)**5 - 6.0*(2.0-q)**5

        else:
            val = (3.0-q)**5 - 6*(2.0-q)**5 + 15.0*(1.0-q)**5

        return val * fac

    def gradient(self, xij=[0., 0, 0], rij=1.0, h=1.0, grad=[0., 0, 0]):
        h1 = 1./h
        q = rij*h1

        if self.dim == 2:
            fac = M_1_PI * 7./478.0 * h1 * h1

        else:
            fac = 0.0

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

    def cython_code(self):
        code = dedent('''
        from libc.math cimport M_1_PI
        ''')
        return dict(helper=code)
