"""Taylor bar example with SPH"""
import numpy

from pysph.sph.equation import Group

from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, WendlandQuintic
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import Integrator, SolidMechStep

# basic sph equations
from pysph.sph.basic_equations import ContinuityEquation, \
    MonaghanArtificialViscosity, XSPHCorrection, VelocityGradient2D

# baic stress equations
from pysph.sph.solid_mech.basic import HookesDeviatoricStressRate2D,\
    MomentumEquationWithStress2D, EnergyEquationWithStress2D

# plasticity model and eos
from pysph.sph.solid_mech.hvi import VonMisesPlasticity2D, MieGruneisenEOS

# boundary force
from pysph.sph.boundary_equations import MonaghanBoundaryForce

# Numerical Parameters and constants
dx = dy = 0.000384848
hdx = 2.0
h = hdx * dx
r0 = 7850
m0 = dx * dy * r0
v_s = 200
ss = 4699
C = 3630
S = 1800
gamma = 1.81
alpha = 0.5
beta = 0.5
eta = 0.01
eps = 0.5
bar_width=0.0076
G = 8*1e10
Yo = 6*1e8
ro2= 2750
plate_start = -2.0*bar_width
plate_end = 2.0*bar_width


def get_plate_particles():
    x = numpy.arange(plate_start, plate_end+dx, dx)
    y = numpy.zeros_like(x)

    # normals and tangents
    tx = numpy.ones_like(x)
    ty = numpy.zeros_like(x)
    tz = numpy.zeros_like(x)

    ny = numpy.ones_like(x)
    nx = numpy.zeros_like(x)
    nz = numpy.zeros_like(x)

    pa = get_particle_array(name='plate', x=x, y=y, tx=tx, ty=ty, tz=tz,
                            nx=nx, ny=ny, nz=nz)
    pa.m[:] = m0

    return pa

def get_bar_particles():
    xarr = numpy.arange(-bar_width/2.0, bar_width/2.0 + dx, dx)
    yarr = numpy.arange(4*dx, 0.0254 + 4*dx, dx)
    
    x,y = numpy.meshgrid( xarr, yarr )
    x, y = x.ravel(), y.ravel()                    

    print 'Number of bar particles: ', len(x)

    hf = numpy.ones_like(x) * h
    mf = numpy.ones_like(x) * dx * dy * r0
    rhof = numpy.ones_like(x) * r0
    csf = numpy.ones_like(x) * ss
    z = numpy.zeros_like(x)
    pa = get_particle_array(name="bar",
                            x=x, y=y, h=hf, m=mf, rho=rhof, cs=csf,
                            e=z)
    # negative fluid particles
    pa.v[:]=-200
    return pa

def create_particles(**kwargs):
    bar = get_bar_particles()
    plate = get_plate_particles()

    # add requisite properties
    
    # velocity gradient for the bar
    bar.add_property( {'name':'v00'} )
    bar.add_property( {'name':'v01'} )
    bar.add_property( {'name':'v10'} )
    bar.add_property( {'name':'v11'} )

    # deviatoric stress components
    bar.add_property( {'name':'s00'} )
    bar.add_property( {'name':'s01'} )
    bar.add_property( {'name':'s02'} )
    bar.add_property( {'name':'s11'} )
    bar.add_property( {'name':'s12'} )
    bar.add_property( {'name':'s22'} )

    # deviatoric stress accelerations
    bar.add_property( {'name':'as00'} )
    bar.add_property( {'name':'as01'} )
    bar.add_property( {'name':'as02'} )
    bar.add_property( {'name':'as11'} )
    bar.add_property( {'name':'as12'} )
    bar.add_property( {'name':'as22'} )

    # deviatoric stress initial values
    bar.add_property( {'name':'s000'} )
    bar.add_property( {'name':'s010'} )
    bar.add_property( {'name':'s020'} )
    bar.add_property( {'name':'s110'} )
    bar.add_property( {'name':'s120'} )
    bar.add_property( {'name':'s220'} )
    
    bar.add_property( {'name':'e0'} )

    # artificial stress properties
    bar.add_property( {'name':'r00'} )
    bar.add_property( {'name':'r01'} )
    bar.add_property( {'name':'r11'} )

    # standard acceleration variables
    bar.add_property( {'name':'arho'} )
    bar.add_property( {'name':'au'} )
    bar.add_property( {'name':'av'} )
    bar.add_property( {'name':'aw'} )
    bar.add_property( {'name':'ax'} )
    bar.add_property( {'name':'ay'} )
    bar.add_property( {'name':'az'} )
    bar.add_property( {'name':'ae'} )

    # initial values
    bar.add_property( {'name':'rho0'} )
    bar.add_property( {'name':'u0'} )
    bar.add_property( {'name':'v0'} )
    bar.add_property( {'name':'w0'} )
    bar.add_property( {'name':'x0'} )
    bar.add_property( {'name':'y0'} )
    bar.add_property( {'name':'z0'} )
    bar.add_property( {'name':'e0'} )

    return [bar, plate]

# create the Application
app = Application()

# kernel
kernel = WendlandQuintic(dim=2)
wdeltap = kernel.kernel(rij=dx, h=hdx*dx)

# integrator
integrator = Integrator(bar=SolidMechStep())

# Create a solver
solver = Solver(kernel=kernel, dim=2, integrator=integrator)

# default parameters
dt = 1e-9
tf = 2.5e-5
solver.set_time_step(dt)
solver.set_final_time(tf)

# add the equations
equations = [

    # Properties computed set from the current state
    Group(
        equations=[
            # p
            MieGruneisenEOS(dest='bar', sources=None, r0=r0, c0=C, S=S),
            
            # vi,j : requires properties v00, v01, v10, v11
            VelocityGradient2D(dest='bar', sources=['bar',]),
            
            # rij : requires properties s00, s01, s11
            VonMisesPlasticity2D(flow_stress=Yo, dest='bar',
                                 sources=None),
            ],
        ),
    
    # Acceleration variables are now computed
    Group(
        equations=[

            # arho
            ContinuityEquation(dest='bar', sources=['bar']),
            
            # au, av
            MomentumEquationWithStress2D(
                dest='bar', sources=['bar'], n=4, wdeltap=wdeltap),

            # au, av
            MonaghanArtificialViscosity(
                dest='bar', sources=['bar'], alpha=0.5, beta=0.5),

            # au av
            MonaghanBoundaryForce(
                dest='bar', sources=['plate'], deltap=dx),

            # ae
            EnergyEquationWithStress2D(dest='bar', sources=['bar'],
                                       alpha=0.5, beta=0.5, eta=0.01),
                
            # a_s00, a_s01, a_s11
            HookesDeviatoricStressRate2D(
                dest='bar', sources=None, shear_mod=G),

            # ax, ay, az
            XSPHCorrection(
                dest='bar', sources=['bar',], eps=0.5),

            ]

        ) # End Acceleration Group

    ] # End Group list

# Setup the application and solver.  This also generates the particles.
app.setup(
    solver=solver, equations=equations, particle_factory=create_particles)

# run
app.run()
