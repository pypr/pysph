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
from pysph.base.utils import get_particle_array_wcsph

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

    x = numpy.concatenate( [xl, xb, xr] )
    y = numpy.concatenate( [yl, yb, yr] )

    return x, y

def create_2D_filled_region(x1, y1, x2, y2, dx):
    x,y = numpy.mgrid[x1:x2+dx/2:dx, y1:y2+dx/2:dx]
    x = x.ravel(); y = y.ravel()

    return x, y

def create_obstacle(x1, x2, height, dx):
    eps = 1e-6

    # left inside wall
    yli = numpy.arange( dx/2.0, height + eps, dx )
    xli = numpy.ones_like(yli) * x1

    # left outside wall
    ylo = numpy.arange( dx, height + dx/2.0 + eps, dx )
    xlo = numpy.ones_like( ylo ) * x1 - dx/2.0

    # # right inside wal
    # yri = numpy.arange( dx/2.0, height + eps, dx )
    # xri = numpy.ones_like(yri) * x2

    # # right outter wall
    # yro = numpy.arange( dx, height + dx/2.0 + eps, dx )
    # xro = numpy.ones_like(yro) * x2 + dx/2.0

    # concatenate the arrays
    x = numpy.concatenate( (xli, xlo) )
    y = numpy.concatenate( (yli, ylo) )

    return x, y

class DamBreak2DGeometry(object):
    def __init__(self, container_width=4.0, container_height=3.0,
                 fluid_column_width=1.0, fluid_column_height=2.0,
                 dx=0.03, dy=0.03, nboundary_layers=4,
                 ro=1000.0, co=1.0, with_obstacle=False):
        self.container_width = container_width
        self.container_height = container_height
        self.fluid_column_height = fluid_column_height
        self.fluid_column_width = fluid_column_width

        self.dx = dx
        self.dy = dy

        self.nsolid = 0
        self.nfluid = 0

        self.ro=ro
        self.co=co

        self.with_obstacle = with_obstacle

    def get_wall(self, nboundary_layers=4):
        container_width = self.container_width
        container_height = self.container_height

        dx, dy = self.dx, self.dy

        _x = []; _y = []
        for i in range(nboundary_layers):
            xb, yb = create_2D_tank(
                x1=-0.5*i*dx, y1=-0.5*i*dy,
                x2=container_width+0.5*i*dx, y2=container_height, dx=dx)

            _x.append(xb); _y.append(yb)

        x = numpy.concatenate(_x); y = numpy.concatenate(_y)
        self.nsolid = len(x)
        
        return x, y

    def get_fluid(self, noffset=2):
        fluid_column_width = self.fluid_column_width
        fluid_column_height = self.fluid_column_height

        dx, dy = self.dx, self.dy

        _x = []; _y = []
        for i in range(noffset):
            ii = i+1
            xf, yf = create_2D_filled_region(
                x1=0.5*ii*dx, y1=0.5*ii*dx,
                x2=fluid_column_width+0.5*i*dx, y2=fluid_column_height,
                dx=dx)

            _x.append(xf); _y.append(yf)

        x = numpy.concatenate(_x); y = numpy.concatenate(_y)
        self.nfluid = len(x)
        
        return x, y

    def create_particles(self, nboundary_layers=2, nfluid_offset=2,
                         hdx=1.5):
        xf, yf = self.get_fluid(nfluid_offset)
        fluid = get_particle_array_wcsph(x=xf, y=yf)
        nfluid = self.nfluid            

        xb, yb = self.get_wall(nboundary_layers)
        boundary = get_particle_array_wcsph(x=xb, y=yb)

        dx, dy, ro = self.dx, self.dy, self.ro

        # smoothing length, mass and density
        fluid.h[:] = numpy.ones_like(xf) * hdx * dx
        fluid.m[:] = dx * dy * 0.5 * ro
        fluid.rho[:] = ro
        fluid.rho0[:] = ro

        boundary.h[:] = numpy.ones_like(xb) * hdx * dx
        boundary.m[:] = dx * dy * 0.5 * ro
        boundary.rho[:] = ro
        boundary.rho0[:] = ro

        # create the particles list
        particles = [fluid, boundary]

        if self.with_obstacle:
            xo, yo = create_obstacle( x1=2.5, x2=2.5+dx, height=0.25, dx=dx )
            obstacle = get_particle_array_wcsph(x=xo, y=yo)

            obstacle.h[:] = numpy.ones_like(xo) * hdx * dx
            obstacle.m[:] = dx * dy * 0.5 * ro
            obstacle.rho[:] = ro
            obstacle.rho0[:] = ro

            particles.append( obstacle )

        return particles
