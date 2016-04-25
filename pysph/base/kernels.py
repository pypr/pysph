"""Definition of some SPH kernel functions
"""

from math import pi, sqrt, exp

M_1_PI = 1.0/pi
M_2_SQRTPI = 2.0/sqrt(pi)

def get_correction(kernel, h0):
    rij = kernel.deltap * h0
    return kernel.kernel(rij=rij, h=h0)

def get_compiled_kernel(kernel):
    """Given a kernel, return a high performance wrapper kernel.
    """
    from pysph.base import c_kernels
    cls = getattr(c_kernels, kernel.__class__.__name__)
    wrapper = getattr(c_kernels, kernel.__class__.__name__ + 'Wrapper')
    kern = cls(**kernel.__dict__)
    return wrapper(kern)

###############################################################################
# `CubicSpline` class.
###############################################################################
class CubicSpline(object):
    r"""Cubic Spline Kernel: [Monaghan1992]_

    .. math::
             W(q) = \ &\sigma_3\left[ 1 - \frac{3}{2}q^2\left( 1 - \frac{q}{2} \right) \right], \ & \textrm{for} \ 0 \leq q \leq 1,\\
                  = \ &\frac{\sigma_3}{4}(2-q)^3, & \textrm{for}\ 1 < q \leq 2,\\
                  = \ &0, & \textrm{for}\ q>2, \\

    where :math:`\sigma_3` is a dimensional normalizing factor for the
    cubic spline function given by:

    .. math::
             \sigma_3  = \ & \frac{2}{3h^1}, & \textrm{for dim=1}, \\
             \sigma_3  = \ & \frac{10}{7\pi h^2}, \ & \textrm{for dim=2}, \\
             \sigma_3  = \ & \frac{1}{\pi h^3}, & \textrm{for dim=3}. \\

    References
    ----------
    .. [Monaghan1992] `J. Monaghan, Smoothed Particle Hydrodynamics, "Annual
        Review of Astronomy and Astrophysics", 30 (1992), pp. 543-574.
        <http://adsabs.harvard.edu/abs/1992ARA&A..30..543M>`_
    """
    def __init__(self, dim=1):
        self.radius_scale = 2.0
        self.dim = dim

        if dim == 3:
            self.fac = M_1_PI
        elif dim == 2:
            self.fac = 10*M_1_PI/7.0
        else:
            self.fac = 2.0/3.0

    def get_deltap(self):
        return 2./3

    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):
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

    def gradient(self, xij=[0., 0, 0], rij=1.0, h=1.0, grad=[0, 0, 0]):
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

    def gradient_h(self, xij=[0., 0, 0], rij=1.0, h=1.0):
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

class WendlandQuintic(object):
    def __init__(self, dim=2):
        self.radius_scale = 2.0
        if dim == 1:
            raise ValueError("WendlandQuintic: Dim %d not supported"%dim)
        self.dim = dim

        if dim == 3:
            self.fac = M_1_PI * 21.0/16.0
        if dim == 2:
            self.fac = 7.0 * M_1_PI/4.0

    def get_deltap(self):
        return 0.5

    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):
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

    def gradient(self, xij=[0., 0, 0], rij=1.0, h=1.0, grad=[0, 0, 0]):
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

    def gradient_h(self, xij=[0., 0, 0], rij=1.0, h=1.0):
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

class Gaussian(object):
    r"""Gaussian Kernel: [Liu2010]_

    .. math::
             W(q) = \ &\sigma_g e^{-q^2}, \ & \textrm{for} \ 0\leq q \leq 3,\\
                  = \ & 0, & \textrm{for} \ q>3,\\

    where :math:`\sigma_g` is a dimensional normalizing factor for the gaussian function
    given by:

    .. math::
             \sigma_g  = \ & \frac{1}{\pi^{1/2} h^1}, \ & \textrm{for dim=1}, \\
             \sigma_g  = \ & \frac{1}{\pi h^2}, \ & \textrm{for dim=2}, \\
             \sigma_g  = \ & \frac{1}{\pi^{3/2} h^3}, & \textrm{for dim=3}. \\

    References
    ----------
    .. [Liu2010] `M. Liu, & G. Liu, Smoothed particle hydrodynamics (SPH):
        an overview and recent developments, "Archives of computational
        methods in engineering", 17.1 (2010), pp. 25-76.
        <http://link.springer.com/article/10.1007/s11831-010-9040-7>`_
    """
    def __init__(self, dim=2):
        self.radius_scale = 3.0
        self.dim = dim

        self.fac = 0.5*M_2_SQRTPI
        if dim > 1:
            self.fac *= 0.5*M_2_SQRTPI
        if dim > 2:
            self.fac *= 0.5*M_2_SQRTPI

    def get_deltap(self):
        # The inflection point is at q=1/sqrt(2)
        # the deltap values for some standard kernels
        # have been tabulated in sec 3.2 of
        # http://cfd.mace.manchester.ac.uk/sph/SPH_PhDs/2008/crespo_thesis.pdf
        return 0.70710678118654746

    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):
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

    def gradient(self, xij=[0., 0, 0], rij=1.0, h=1.0, grad=[0., 0, 0]):
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

    def gradient_h(self, xij=[0., 0., 0.], rij=1.0, h=1.0):
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

class QuinticSpline(object):
    r"""Quintic Spline SPH kernel: [Liu2010]_

    .. math::
             W(q) = \ &\sigma_5\left[ (3-q)^5 - 6(2-q)^5 + 15(1-q)^5 \right], \ & \textrm{for} \ 0\leq q \leq 1,\\
                  = \ &\sigma_5\left[ (3-q)^5 - 6(2-q)^5 \right], & \textrm{for} \ 1 <  q \leq 2,\\
                  = \ &\sigma_5 \ (3-q)^5 , & \textrm{for} \ 2 < q \leq 3,\\
                  = \ & 0, & \textrm{for} \ q>3,\\

    where :math:`\sigma_5` is a dimensional normalizing factor for the
    quintic spline function given by:

    .. math::
             \sigma_5  = \ & \frac{120}{h^1}, & \textrm{for dim=1}, \\
             \sigma_5  = \ & \frac{7}{478\pi h^2}, \ & \textrm{for dim=2}, \\
             \sigma_5  = \ & \frac{3}{359\pi h^3}, & \textrm{for dim=3}. \\

    Raises
    ------
    NotImplementedError
        Quintic spline currently only supports 2D kernels.
    """
    def __init__(self, dim=2):
        self.radius_scale = 3.0
        if dim != 2:
            raise NotImplementedError('Quintic spline currently only supports 2D kernels.')
        self.dim = dim

        self.fac = M_1_PI * 7.0/478.0

    def get_deltap(self):
        # The inflection points for the polynomial are obtained as
        # http://www.wolframalpha.com/input/?i=%28%283-x%29%5E5+-+6*%282-x%29%5E5+%2B+15*%281-x%29%5E5%29%27%27
        # the only permissible value is taken
        return 0.759298480738450

    def kernel(self, xij=[0., 0, 0], rij=1.0, h=1.0):
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

    def gradient(self, xij=[0., 0, 0], rij=1.0, h=1.0, grad=[0., 0, 0]):
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

    def gradient_h(self, xij=[0., 0, 0], rij=1.0, h=1.0):
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
