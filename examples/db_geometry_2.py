import numpy
from numpy import concatenate, where, array

from pysph.base.utils import get_particle_array

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
    
    xu =  xb = numpy.arange(x1+dx, x2-dx+dx/2, dx)
    yu = numpy.ones_like(xb) * y2

    x = numpy.concatenate( [xl, xb, xr, xu] )
    y = numpy.concatenate( [yl, yb, yr, yu] )

    return x, y

def create_2D_filled_region(x1, y1, x2, y2, dx):
    print "%d %d %d %d" %(x1,y1,x2,y2)
    eps = 1e-6
    x,y = numpy.mgrid[x1:x2+eps:dx, y1:y2+eps:dx]
    x = x.ravel(); y = y.ravel()

    return x, y

def create_obstacle(x1, x2, y1,y2, dx):
    eps = 1e-6

#    # left inside wall
#    yli = numpy.arange( dx/2.0, height + eps, dx )
#    xli = numpy.ones_like(yli) * x1
#
#    # left outside wall
#    ylo = numpy.arange( dx, height + dx/2.0 + eps, dx )
#    xlo = numpy.ones_like( ylo ) * x1 - dx/2.0
#
#    # # right inside wal
#    # yri = numpy.arange( dx/2.0, height + eps, dx )
#    # xri = numpy.ones_like(yri) * x2
#
#    # # right outter wall
#    # yro = numpy.arange( dx, height + dx/2.0 + eps, dx )
#    # xro = numpy.ones_like(yro) * x2 + dx/2.0
#
#    # concatenate the arrays
#    x = numpy.concatenate( (xli, xlo) )
#    y = numpy.concatenate( (yli, ylo) )
    
    ####################################################
        # left inside wall
    yli = numpy.arange( y1 + dx/2.0, y2 + eps, dx )
    xli = numpy.ones_like(yli) * x1

    # left outside wall
    ylo = numpy.arange( y1 + dx, y2 + dx/2.0 + eps, dx )
    xlo = numpy.ones_like( ylo ) * x1 - dx/2.0

     # right inside wall
    yri = numpy.arange( y1 + dx/2.0, y2 + eps, dx )
    xri = numpy.ones_like(yri) * x2

     # right outter wall
    yro = numpy.arange( y1 + dx, y2 + dx/2.0 + eps, dx )
    xro = numpy.ones_like(yro) * x2 + dx/2.0
     # upper inside wall
    xui = numpy.arange(x1+dx/2.0,x2+eps,dx)
    yui = numpy.ones_like(xui) * (y2)
    # upper outside wall
    xuo = numpy.arange(x1,x2+eps,dx)
    yuo = numpy.ones_like(xuo) * (y2+dx/2.0)
    # down inside wall
    xdo = numpy.arange(x1+dx/2.0,x2+eps,dx)
    ydo = numpy.ones_like(xuo) * (y1+dx/2.0)
    # down outside wall
    xdo = numpy.arange(x1,x2+eps,dx)
    ydo = numpy.ones_like(xuo) * (y1)
    # concatenate the arrays
    x = numpy.concatenate( (xli, xlo,xri,xro,xui,xuo,xdo) )
    y = numpy.concatenate( (yli, ylo,yri,yro,yui,yuo,ydo) )

    return x, y

class DamBreak2DGeometry(object):
    def __init__(self, container_width=4.0, container_height=3.0,
                 fluid_column_width=1.0, fluid_column_height=2.0,
                 dx=0.03, dy=0.03, nboundary_layers=4,
                 ro=1000.0, co=1.0, with_obstacle=False,
                 beta=1.0, nfluid_offset=2, hdx=1.5):

        self.container_width = container_width
        self.container_height = container_height
        self.fluid_column_height = fluid_column_height
        self.fluid_column_width = fluid_column_width

        self.nboundary_layers=nboundary_layers
        self.nfluid_offset=nfluid_offset
        self.beta = beta
        self.hdx = hdx
        self.dx = dx
        self.dy = dy

        self.nsolid = 0
        self.nfluid = 0

        self.ro=ro
        self.rho0 = ro
        self.co=co

        self.with_obstacle = with_obstacle

    def get_wall(self, nboundary_layers=20):
        container_width = self.container_width
        container_height = self.container_height

        dx, dy = self.dx, self.dy

        _x = []; _y = []
        for i in range(6):
            xb, yb = create_2D_tank(
                x1=-0.5*i*dx, y1=-0.5*i*dx,
                x2=container_width+0.5*i*dx, y2=container_height+0.5*i*dx, dx=dx)

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
                x1 = dx-0.5*i*dx, y1 = dx-0.5*i*dx,
                #x1=0.5*ii*dx, y1=0.5*ii*dx,
                x2=1.0-dx+0.5*i*dx, y2=fluid_column_height-dx+0.5*i*dx,
                dx=dx)
            _x.append(xf); _y.append(yf)
   
        for i in range(noffset):
            ii = i+1
            xf1, yf1 = create_2D_filled_region(
                x1 = 1.0+0.5*i*dx, y1 = 3.0+dx-0.5*i*dx,
                #x1=0.5*ii*dx, y1=0.5*ii*dx,
                x2=10.0-dx+0.5*i*dx, y2=fluid_column_height-dx+0.5*i*dx,
                dx=dx)
            _x.append(xf1); _y.append(yf1)
        for i in range(noffset):
            ii = i+1
            xf2, yf2 = create_2D_filled_region(
                x1 = 2.0+dx-0.5*i*dx, y1 = dx-0.5*i*dx,
                #x1=0.5*ii*dx, y1=0.5*ii*dx,
                x2=10.0-dx+0.5*i*dx, y2=3.0-0.5*i*dx,
                dx=dx)
            _x.append(xf2); _y.append(yf2)
    
        for i in range(noffset):
            ii = i+1
            xf3, yf3 = create_2D_filled_region(
                x1 = 1+0.5*i*dx, y1 = dx-0.5*i*dx,
                #x1=0.5*ii*dx, y1=0.5*ii*dx,
                x2=2.0-0.5*i*dx, y2=2.0-dx+0.5*i*dx,
                dx=dx)
            _x.append(xf3); _y.append(yf3)
   #     print "done"
#        _x.append(xf); _y.append(yf)
#        _x.append(xf1); _y.append(yf1)
#        _x.append(xf2); _y.append(yf2)
#        _x.append(xf3); _y.append(yf3)

        x = numpy.concatenate(_x); y = numpy.concatenate(_y)
        self.nfluid = len(x)

        return x, y

    def create_particles(self, nboundary_layers=2, nfluid_offset=2,
                         hdx=1.5, **kwargs):
        nfluid = self.nfluid
        xf, yf = self.get_fluid(nfluid_offset)
        fluid = get_particle_array(name='fluid', x=xf, y=yf)

        fluid.gid[:] = range(fluid.get_number_of_particles())

        np = nfluid

        xb, yb = self.get_wall(1)
        boundary = get_particle_array(name='boundary', x=xb, y=yb)

        np += boundary.get_number_of_particles()

        dx, dy, ro = self.dx, self.dy, self.ro

        # smoothing length, mass and density
        fluid.h[:] = numpy.ones_like(xf) * hdx * dx


        # create the particles list
        #particles = [fluid, boundary]

        if self.with_obstacle:
            xx = []; yy=[]
            ##########
            for i in range(10):
                xb, yb = create_2D_tank(
                    x1 = 1.0+0.5*i*dx, y1 = 2.0+0.5*i*dx,
                    #x1=0.5*ii*dx, y1=0.5*ii*dx,
                    x2=2.0-0.5*i*dx, y2=3.0-0.5*i*dx,
                    dx=dx)
                xx.append(xb);yy.append(yb)
          
        #############
            xo = numpy.concatenate(xx); yo = numpy.concatenate(yy)
            #xo, yo = create_obstacle( x1=1.0, x2=2.0+dx,y1 = 2.0,y2 = 3.0, dx=dx )
            gido = numpy.array( range(xo.size), dtype=numpy.uint32 )
            
            obstacle = get_particle_array(name='obstacle',x=xo, y=yo)
            
          
            # add the obstacle to the boundary particles
            #boundary.append_parray( obstacle )

            np += obstacle.get_number_of_particles()

        # set the gid for the boundary particles
        particles = [fluid, boundary, obstacle]
        fluid.gid[:]=  range( fluid.get_number_of_particles() )
        boundary.gid[:] = range( boundary.get_number_of_particles() )
        obstacle.gid[:] = range( obstacle.get_number_of_particles() )
        # boundary particles can do with a reduced list of properties
        # to be saved to disk since they are fixed
        boundary.set_output_arrays( ['x', 'y', 'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid'] )
        
        #############
        

        print "2D dam break with %d fluid, %d obstacle particles, %d boundary particles"%(
            fluid.get_number_of_particles(),
            obstacle.get_number_of_particles(),
            boundary.get_number_of_particles()),
  
#        
  
      ###########################333
      # particle volume for fluid and solid
        fluid.add_property('V')
        obstacle.add_property('V' )
        boundary.add_property('V' )

    # Shepard filtered velocities for the fluid
        for name in ['uf', 'vf', 'wf']:
            fluid.add_property(name)

    # advection velocities and accelerations
        for name in ('uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 'au', 'av', 'aw'):
            fluid.add_property(name)

     # kernel summation correction for the solid
        obstacle.add_property('wij')
        boundary.add_property('wij')

    # imposed or prescribed boundary velocity for the solid
        boundary.add_property('u0'); boundary.u0[:] = 0.
        boundary.add_property('v0'); boundary.v0[:] = 0.
        boundary.add_property('w0'); boundary.w0[:] = 0.
        obstacle.add_property('u0'); obstacle.u0[:] = 0.
        obstacle.add_property('v0'); obstacle.v0[:] = 0.
        obstacle.add_property('w0'); obstacle.w0[:] = 0.

    # imposed accelerations on the obstacle
        obstacle.add_property('ax')
        obstacle.add_property('ay')
        obstacle.add_property('az')
        boundary.add_property('ax')
        boundary.add_property('ay')
        boundary.add_property('az')

    # magnitude of velocity
        fluid.add_property('vmag')

    # setup the particle properties
        volume = dx * dx
    
    # mass is set to get the reference density of rho0
        fluid.m[:] = volume * self.rho0
        boundary.m[:] = volume * self.rho0
        obstacle.m[:] = volume * self.rho0
        
    # volume is set as dx^2
        fluid.V[:] = 1./volume
        boundary.V[:] = 1./volume
        obstacle.V[:] = 1./volume

    # smoothing lengths
        fluid.h[:] = hdx * dx
        obstacle.h[:] = hdx * dx
        boundary.h[:] = hdx * dx
        obstacle.set_output_arrays( ['x', 'y', 'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid','u0'] )
        obstacle.set_output_arrays( ['x', 'y', 'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid','u0'] )

    # imposed horizontal velocity on the lid
    # solid.u0[:] = 0.0
    # solid.v0[:] = 0.0
   ##############################
       
        

            
        
                

        return particles
