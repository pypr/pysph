"""Helper module to easily create uniform distributions of particles"""

import numpy

def uniform_distribution_hcp2D(dx, xmin, xmax, ymin, ymax, adjust=False):
    """Hexagonal closed packing arrangement in 2D"""
    dy = 0.5 * numpy.sqrt(3.0) * dx
    dxb2 = 0.5 * dx
    dyb2 = 0.5 * dy
    
    # since we will be shifting each alternate row by dxb2, we use
    # xstart as dx/4
    xstart = xmin + 0.25 * dx

    ystart = ymin+dyb2

    # adjust ymax so that particles can fill a periodic region
    if adjust:
        _y = numpy.arange(ystart, ymax, dy)
        ymax = _y[-1] + 1.5*dy
    
    # create the points
    x, y = numpy.mgrid[xstart:xmax:dx,
                       ystart:ymax:dy]

    # each alternate row is shifted by dxb2
    x[:,::2] += dxb2
    x = x.ravel(); y = y.ravel()

    print 'HCP packing domain: xmin, xmax, ymin, ymax =  ', xmin, xmax, ymin, ymax
    print 'HCP packing particles: xmin, xmax, ymin, ymax = ', x.min(), x.max(), y.min(), y.max()
    print 'Particle spacings: dx, dy = ', dx, dy
    print 'Offset: xmin, xmax = ', x.min()-xmin, xmax-x.max()
    print 'Offset: ymin, ymax = ', y.min()-ymin, ymax-y.max()

    return x, y, dx, dy, xmin, xmax, ymin, ymax

def uniform_distribution_cubic2D(dx, xmin, xmax, ymin, ymax, nrows=None):
    """Cubic lattice arrangement in 2D"""
    dy = dx
    dxb2 = 0.5 * dx
    dyb2 = 0.5 * dy
    
    if nrows is not None:
        ymax = nrows * dy

    xstart = xmin + dxb2
    ystart = ymin + dyb2
    x, y = numpy.mgrid[xstart:xmax:dx,
                       ystart:ymax:dy]

    x = x.ravel(); y = y.ravel()

    print 'Cubic packing domain: xmin, xmax, ymin, ymax =  ', xmin, xmax, ymin, ymax
    print 'Cubic packing particles: xmin, xmax, ymin, ymax = ', x.min(), x.max(), y.min(), y.max()
    print 'Particle spacings: dx, dy = ', dx, dy
    print 'Offset: xmin, xmax = ', x.min()-xmin, xmax-x.max()
    print 'Offset: ymin, ymax = ', y.min()-ymin, ymax-y.max()

    return x, y, dx, dy, xmin, xmax, ymin, ymax
