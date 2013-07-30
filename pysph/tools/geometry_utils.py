""" Helper functions to generate commonly used geometries.

PySPH used an axis convention as follows:

Y
|
|
|
|
|
|      /Z
|     /
|    /
|   /
|  /
| /
|/_________________X



"""

import numpy

def create_2D_tank(x1,y1,x2,y2,dx):
    """ Generate an open rectangular tank.

    Parameters:
    -----------

    x1,y1,x2,y2 : Coordinates defining the rectangle in 2D

    dx : The spacing to use

    """

    yl = numpy.arange(y1, y2+dx/2, dx)
    xl = numpy.ones_like(yl) * x1
    nl = len(xl)

    yr = numpy.arange(y1,y2+dx/2, dx)
    xr = numpy.ones_like(yr) * x2
    nr = len(xr)

    xb = numpy.arange(x1+dx, x2-dx+dx/2, dx)
    yb = numpy.ones_like(xb) * y1
    nb = len(xb)

    n = nb + nl + nr

    x = numpy.empty( shape=(n,) )
    y = numpy.empty( shape=(n,) )

    idx = 0
    x[idx:nl] = xl; y[idx:nl] = yl

    idx += nl
    x[idx:idx+nb] = xb; y[idx:idx+nb] = yb

    idx += nb
    x[idx:idx+nr] = xr; y[idx:idx+nr] = yr

    return x, y

def create_3D_tank(x1, y1, z1, x2, y2, z2, dx):
    """ Generate an open rectangular tank.

    Parameters:
    -----------

    x1,y1,x2,y2,x3,y3 : Coordinates defining the rectangle in 2D

    dx : The spacing to use

    """

    points = []
    # create the base X-Y plane
    x, y = numpy.mgrid[x1:x2+dx/2:dx, y1:y2+dx/2:dx]
    x = x.ravel(); y = y.ravel()
    z = numpy.ones_like(x) * z1

    for i in range(len(x)):
        points.append( (x[i], y[i], z[i]) )

    # create the front X-Z plane
    x, z = numpy.mgrid[x1:x2+dx/2:dx, z1:z2+dx/2:dx]
    x = x.ravel(); z = z.ravel()
    y = numpy.ones_like(x) * y1

    for i in range(len(x)):
        points.append( (x[i], y[i], z[i]) )

    # create the Y-Z plane
    y, z = numpy.mgrid[y1:y2+dx/2:dx, z1:z2+dx/2:dx]
    y = y.ravel(); z = z.ravel()
    x = numpy.ones_like(y) * x1

    for i in range(len(x)):
        points.append( (x[i], y[i], z[i]) )

    # create the second X-Z plane
    x, z = numpy.mgrid[x1:x2+dx/2:dx, z1:z2+dx/2:dx]
    x = x.ravel(); z = z.ravel()
    y = numpy.ones_like(x) * y2

    for i in range(len(x)):
        points.append( (x[i], y[i], z[i]) )

    # create the second Y-Z plane
    y, z = numpy.mgrid[y1:y2+dx/2:dx, z1:z2+dx/2:dx]
    y = y.ravel(); z = z.ravel()
    x = numpy.ones_like(y) * x2

    for i in range(len(x)):
        points.append( (x[i], y[i], z[i]) )

    points = set(points)

    x = numpy.array( [i[0] for i in points] )
    y = numpy.array( [i[1] for i in points] )
    z = numpy.array( [i[2] for i in points] )

    return x, y, z

def create_2D_filled_region(x1, y1, x2, y2, dx):
    x,y = numpy.mgrid[x1:x2+dx/2:dx, y1:y2+dx/2:dx]
    x = x.ravel(); y = y.ravel()

    return x, y

def create_3D_filled_region(x1, y1, z1, x2, y2, z2, dx):
    x,y,z = numpy.mgrid[x1:x2+dx/2:dx, y1:y2+dx/2:dx, z1:z2+dx/2:dx]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    return x, y, z


