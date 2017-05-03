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
from numpy import concatenate, where, array
from pysph.base.utils import get_particle_array_wcsph, get_particle_array_iisph

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
                 ro=1000.0, co=1.0, with_obstacle=False,
                 beta=1.0, nfluid_offset=2, hdx=1.5, iisph=False,
                 wall_hex_pack=True):

        self.container_width = container_width
        self.container_height = container_height
        self.fluid_column_height = fluid_column_height
        self.fluid_column_width = fluid_column_width

        self.nboundary_layers = nboundary_layers
        self.nfluid_offset = nfluid_offset
        self.beta = beta
        self.hdx = hdx
        self.dx = dx
        self.dy = dy
        self.iisph = iisph
        self.wall_hex_pack = wall_hex_pack

        self.nsolid = 0
        self.nfluid = 0

        self.ro = ro
        self.co = co

        self.with_obstacle = with_obstacle

    def get_wall(self, nboundary_layers=4):
        container_width = self.container_width
        container_height = self.container_height

        dx, dy = self.dx/self.beta, self.dy/self.beta
        factor = 0.5 if self.wall_hex_pack else 1.0
        _x = []; _y = []
        for i in range(nboundary_layers):
            xb, yb = create_2D_tank(
                x1=-factor*i*dx, y1=-factor*i*dy,
                x2=container_width+factor*i*dx, y2=container_height, dx=dx)

            _x.append(xb); _y.append(yb)

        x = numpy.concatenate(_x); y = numpy.concatenate(_y)
        self.nsolid = len(x)

        return x, y

    def get_fluid(self, noffset=2):
        fluid_column_width = self.fluid_column_width
        fluid_column_height = self.fluid_column_height

        dx, dy = self.dx, self.dy
        factor = 0.5

        _x = []; _y = []
        for i in range(noffset):
            ii = i+1
            xf, yf = create_2D_filled_region(
                x1 = dx-factor*i*dx, y1 = dx-factor*i*dx,
                #x1=0.5*ii*dx, y1=0.5*ii*dx,
                x2=fluid_column_width+factor*i*dx, y2=fluid_column_height,
                dx=dx)

            _x.append(xf); _y.append(yf)

        x = numpy.concatenate(_x); y = numpy.concatenate(_y)
        self.nfluid = len(x)

        return x, y

    def create_particles(self, nboundary_layers=2, nfluid_offset=2,
                         hdx=1.5, **kwargs):
        nfluid = self.nfluid
        xf, yf = self.get_fluid(nfluid_offset)
        if self.iisph:
            fluid = get_particle_array_iisph(name='fluid', x=xf, y=yf)
        else:
            fluid = get_particle_array_wcsph(name='fluid', x=xf, y=yf)

        fluid.gid[:] = list(range(fluid.get_number_of_particles()))

        np = nfluid

        xb, yb = self.get_wall(nboundary_layers)
        if self.iisph:
            boundary = get_particle_array_iisph(name='boundary', x=xb, y=yb)
        else:
            boundary = get_particle_array_wcsph(name='boundary', x=xb, y=yb)

        np += boundary.get_number_of_particles()

        dx, dy, ro = self.dx, self.dy, self.ro

        # smoothing length, mass and density
        fluid.h[:] = numpy.ones_like(xf) * hdx * dx
        if nfluid_offset == 2:
            fluid.m[:] = dx * dy * ro * 0.5
        else:
            fluid.m[:] = dx * dy * ro
        fluid.rho[:] = ro
        if not self.iisph:
            fluid.rho0[:] = ro

        boundary.h[:] = numpy.ones_like(xb) * hdx * dx
        if nboundary_layers == 2:
            boundary.m[:] = dx * dy * ro * 0.5
        else:
            boundary.m[:] = dx * dy * ro
        boundary.rho[:] = ro
        if not self.iisph:
            boundary.rho0[:] = ro

        # create the particles list
        particles = [fluid, boundary]

        if self.with_obstacle:
            xo, yo = create_obstacle( x1=2.5, x2=2.5+dx, height=0.25, dx=dx )
            gido = numpy.array( list(range(xo.size)), dtype=numpy.uint32 )

            obstacle = get_particle_array_wcsph(name='obstacle',x=xo, y=yo)

            obstacle.h[:] = numpy.ones_like(xo) * hdx * dx
            obstacle.m[:] = dx * dy * 0.5 * ro
            obstacle.rho[:] = ro
            if not self.iisph:
                obstacle.rho0[:] = ro

            # add the obstacle to the boundary particles
            boundary.append_parray( obstacle )

            np += obstacle.get_number_of_particles()

        # set the gid for the boundary particles
        boundary.gid[:] = list(range( boundary.get_number_of_particles()))

        # boundary particles can do with a reduced list of properties
        # to be saved to disk since they are fixed
        boundary.set_output_arrays( ['x', 'y', 'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid'] )
        if self.iisph:
            boundary.add_output_arrays(['V'])

        print("2D dam break with %d fluid, %d boundary particles"%(
            fluid.get_number_of_particles(),
            boundary.get_number_of_particles()))

        return particles

from pyzoltan.core.carray import LongArray
class DamBreak3DGeometry(object):
    def __init__(
        self, container_height=1.0, container_width=1.0, container_length=3.22,
        fluid_column_height=0.55, fluid_column_width=1.0, fluid_column_length=1.228,
        obstacle_center_x=2.5, obstacle_center_y=0,
        obstacle_length=0.16, obstacle_height=0.161, obstacle_width=0.4,
        nboundary_layers=5, with_obstacle=True, dx=0.02, hdx=1.2, rho0=1000.0):

        # save the geometry details
        self.container_width = container_width
        self.container_length = container_length
        self.container_height = container_height

        self.fluid_column_length=fluid_column_length
        self.fluid_column_width=fluid_column_width
        self.fluid_column_height=fluid_column_height

        self.obstacle_center_x = obstacle_center_x
        self.obstacle_center_y = obstacle_center_y

        self.obstacle_width=obstacle_width
        self.obstacle_length=obstacle_length
        self.obstacle_height=obstacle_height

        self.nboundary_layers=nboundary_layers
        self.dx=dx

        self.hdx = hdx
        self.rho0 = rho0
        self.with_obstacle = with_obstacle

    def get_max_speed(self, g=9.81):
        return numpy.sqrt( 2 * g * self.fluid_column_height )

    def create_particles(self, **kwargs):
        fluid_column_height=self.fluid_column_height
        fluid_column_width=self.fluid_column_width
        fluid_column_length=self.fluid_column_length

        container_height = self.container_height
        container_length = self.container_length
        container_width = self.container_width

        obstacle_height = self.obstacle_height
        obstacle_length = self.obstacle_length
        obstacle_width = self.obstacle_width

        obstacle_center_x = self.obstacle_center_x
        obstacle_center_y = self.obstacle_center_y

        nboundary_layers = self.nboundary_layers
        dx = self.dx

        # get the domain limits
        ghostlims = nboundary_layers * dx

        xmin, xmax = 0.0 -ghostlims, container_length + ghostlims
        zmin, zmax = 0.0 - ghostlims, container_height + ghostlims

        cw2 = 0.5 * container_width
        ymin, ymax = -cw2 - ghostlims, cw2 + ghostlims

        # create all particles
        eps = 0.1 * dx
        xx, yy, zz = numpy.mgrid[xmin:xmax+eps:dx,
                                 ymin:ymax+eps:dx,
                                 zmin:zmax+eps:dx]

        x = xx.ravel(); y = yy.ravel(); z = zz.ravel()

        # create a dummy particle array from which we'll sort
        pa = get_particle_array_wcsph(name='block', x=x, y=y, z=z)

        # get the individual arrays
        indices = []
        findices = []
        oindices = []

        obw2 = 0.5 * obstacle_width
        obl2 = 0.5 * obstacle_length
        obh = obstacle_height
        ocx = obstacle_center_x
        ocy = obstacle_center_y

        for i in range(x.size):
            xi = x[i]; yi = y[i]; zi = z[i]

            # fluid
            if ( (0 < xi <= fluid_column_length) and \
                     (-cw2 < yi < cw2) and \
                     (0 < zi <= fluid_column_height) ):

                findices.append(i)

            # obstacle
            if ( (ocx-obl2 <= xi <= ocx+obl2) and \
                     (ocy-obw2 <= yi <= ocy+obw2) and \
                     (0 < zi <= obh) ):

                oindices.append(i)

        # extract the individual arrays
        fa = LongArray(len(findices)); fa.set_data(numpy.array(findices))
        fluid = pa.extract_particles(fa)
        fluid.set_name('fluid')

        if self.with_obstacle:
            oa = LongArray(len(oindices)); oa.set_data(numpy.array(oindices))
            obstacle = pa.extract_particles(oa)
            obstacle.set_name('obstacle')

        indices = concatenate( (where( y <= -cw2 )[0],
                                where( y >= cw2 )[0],
                                where( x >= container_length )[0],
                                where( x <= 0 )[0],
                                where( z <= 0 )[0]) )

        # remove duplicates
        indices = array(list(set(indices)))

        wa = LongArray(indices.size); wa.set_data(indices)
        boundary = pa.extract_particles(wa)
        boundary.set_name('boundary')

        # create the particles
        if self.with_obstacle:
            particles = [fluid, boundary, obstacle]
        else:
            particles = [fluid, boundary]

        # set up particle properties
        h0 = self.hdx * dx

        volume = dx**3
        m0 = self.rho0 * volume

        for pa in particles:
            pa.m[:] = m0
            pa.h[:] = h0

            pa.rho[:] = self.rho0

        nf = fluid.num_real_particles
        nb = boundary.num_real_particles

        if self.with_obstacle:
            no = obstacle.num_real_particles
            print("3D dam break with %d fluid, %d boundary, %d obstacle particles"%(nf, nb, no))
        else:
            print("3D dam break with %d fluid, %d boundary particles"%(nf, nb))


        # load balancing props for the arrays
        #fluid.set_lb_props(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'gid',
        #                    'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0'])
        fluid.set_lb_props( list(fluid.properties.keys()) )

        #boundary.set_lb_props(['x', 'y', 'z', 'rho', 'h', 'm', 'gid', 'rho0'])
        #obstacle.set_lb_props(['x', 'y', 'z', 'rho', 'h', 'm', 'gid', 'rho0'])
        boundary.set_lb_props( list(boundary.properties.keys()) )

        # boundary and obstacle particles can do with a reduced list of properties
        # to be saved to disk since they are fixed
        boundary.set_output_arrays( ['x', 'y', 'z', 'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid'] )

        if self.with_obstacle:
            obstacle.set_lb_props( list(obstacle.properties.keys()) )
            obstacle.set_output_arrays( ['x', 'y', 'z', 'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid'] )

        return particles
