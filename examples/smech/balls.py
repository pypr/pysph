"""Colliding Elastic Balls"""

import numpy

# SPH equations
from pysph.sph.equation import Group
from pysph.sph.basic import IsothermalEOS, ContinuityEquation, MonaghanArtificialViscosity,\
     XSPHCorrection, VelocityGradient2D
from pysph.sph.smech.basic import MomentumEquationWithStress2D, HookesDeviatoricStressRate2D,\
    MonaghanArtificialStress

from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, WendlandQuintic
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import Integrator, SmechStep

def get_K(G, nu):
    ''' Get the bulk modulus from shear modulus and Poisson ratio '''
    return 2.0*G*(1+nu)/(3*(1-2*nu))

# constants
E = 1e7
nu = 0.3975
G = E/(2.0*(1+nu))
K = get_K(G, nu)
ro = 1.0
co = numpy.sqrt(K/ro)

deltap = 0.001
fac=1e-10

def create_particles(two_arr=False, **kwargs):
    #x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -0.105:0.105+1e-4:dx]
    dx = 0.001 # 1mm
    ri = 0.03 # 3cm inner radius
    ro = 0.04 # 4cm outer radius
    spacing = 0.041 # spacing = 2*5cm
    
    x,y = numpy.mgrid[-ro:ro:dx, -ro:ro:dx]
    x = x.ravel()
    y = y.ravel()
    
    d = (x*x+y*y)
    keep = numpy.flatnonzero((ri*ri<=d) * (d<ro*ro))
    x = x[keep]
    y = y[keep]

    print 'num_particles', len(x)*2
    
    if not two_arr:
        x = numpy.concatenate([x-spacing,x+spacing])
        y = numpy.concatenate([y,y])

    #print bdry, numpy.flatnonzero(bdry)
    m = numpy.ones_like(x)*dx*dx
    h = numpy.ones_like(x)*1.4*dx
    rho = numpy.ones_like(x)
    z = numpy.zeros_like(x)

    p = 0.5*1.0*100*100*(1 - (x**2 + y**2))

    cs = numpy.ones_like(x) * 10000.0

    # u is set later
    v = z
    u_f = 0.059

    p *= 0
    h *= 1
    
    # create the particle array
    pa = get_particle_array(
        name="solid", x=x+spacing, y=y, m=m, rho=rho, h=h,
        p=p, cs=cs, u=z, v=v)

    pa.cs[:] = co
    pa.u = pa.cs*u_f*(2*(x<0)-1)

    # add requisite properties
    
    # velocity gradient properties
    pa.add_property( {'name':'v00'} )
    pa.add_property( {'name':'v01'} )
    pa.add_property( {'name':'v10'} )
    pa.add_property( {'name':'v11'} )

    # artificial stress properties
    pa.add_property( {'name':'r00'} )
    pa.add_property( {'name':'r01'} )
    pa.add_property( {'name':'r02'} )
    pa.add_property( {'name':'r11'} )
    pa.add_property( {'name':'r12'} )
    pa.add_property( {'name':'r22'} )

    # deviatoric stress components
    pa.add_property( {'name':'s00'} )
    pa.add_property( {'name':'s01'} )
    pa.add_property( {'name':'s02'} )
    pa.add_property( {'name':'s11'} )
    pa.add_property( {'name':'s12'} )
    pa.add_property( {'name':'s22'} )
    pa.add_property( {'name':'s10'} )

    # deviatoric stress accelerations
    pa.add_property( {'name':'as00'} )
    pa.add_property( {'name':'as01'} )
    pa.add_property( {'name':'as02'} )
    pa.add_property( {'name':'as11'} )
    pa.add_property( {'name':'as12'} )
    pa.add_property( {'name':'as22'} )
    pa.add_property( {'name':'as10'} )

    # deviatoric stress initial values
    pa.add_property( {'name':'s000'} )
    pa.add_property( {'name':'s010'} )
    pa.add_property( {'name':'s020'} )
    pa.add_property( {'name':'s110'} )
    pa.add_property( {'name':'s120'} )
    pa.add_property( {'name':'s220'} )

    # standard acceleration variables
    pa.add_property( {'name':'arho'} )
    pa.add_property( {'name':'au'} )
    pa.add_property( {'name':'av'} )
    pa.add_property( {'name':'aw'} )
    pa.add_property( {'name':'ax'} )
    pa.add_property( {'name':'ay'} )
    pa.add_property( {'name':'az'} )

    # initial values
    pa.add_property( {'name':'rho0'} )
    pa.add_property( {'name':'u0'} )
    pa.add_property( {'name':'v0'} )
    pa.add_property( {'name':'w0'} )
    pa.add_property( {'name':'x0'} )
    pa.add_property( {'name':'y0'} )
    pa.add_property( {'name':'z0'} )

    return [pa,]

# create the Application
app = Application()

# kernel
kernel = CubicSpline(dim=2)

# integrator
integrator = Integrator(solid=SmechStep())

# Create a solver
solver = Solver(kernel=kernel, dim=2, integrator=integrator)

# default parameters
dt = 1e-8
tf = 1e-2
solver.set_time_step(dt)
solver.set_final_time(tf)

# add the equations
equations = [

    # Properties computed set from the current state
    Group(
        equations=[
            # p
            IsothermalEOS(dest='solid', sources=None, rho0=ro, c0=co),

            # vi,j : requires properties v00, v01, v10, v11
            VelocityGradient2D(dest='solid', sources=['solid',]),
            
            # rij : requires properties r00, r01, r02, r11, r12, r22,
            #                           s00, s01, s02, s11, s12, s22            
            MonaghanArtificialStress(
               dest='solid', sources=None, eps=0.3),
            ],
        ),
    
    # Acceleration variables are now computed
    Group(
        equations=[

            # arho
            ContinuityEquation(dest='solid', sources=['solid',]),
            
            # au, av
            MomentumEquationWithStress2D(
                dest='solid', sources=['solid',], n=4),

            # au, av
            MonaghanArtificialViscosity(
                dest='solid', sources=['solid',], alpha=1.0, beta=1.0),
            
            # a_s00, a_s01, a_s11
            HookesDeviatoricStressRate2D(
                dest='solid', sources=None, shear_mod=G),

            # ax, ay, az
            XSPHCorrection(
                dest='solid', sources=['solid',], eps=0.5),

            ]

        ) # End Acceleration Group

    ] # End Group list

# Setup the application and solver.  This also generates the particles.
app.setup(
    solver=solver, equations=equations, particle_factory=create_particles,
    name='fluid')

# run
app.run()
