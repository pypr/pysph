"""Taylor bar example with SPH. (5 minutes)
"""
import numpy

from pysph.sph.equation import Group

from pysph.base.utils import get_particle_array
from pysph.base.kernels import WendlandQuintic
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import SolidMechStep

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
    cs = numpy.ones_like(x)*ss
    pa = get_particle_array(name='plate', x=x, y=y, tx=tx, ty=ty, tz=tz,
                            nx=nx, ny=ny, nz=nz, cs=cs)
    pa.m[:] = m0

    return pa

def get_bar_particles():
    xarr = numpy.arange(-bar_width/2.0, bar_width/2.0 + dx, dx)
    yarr = numpy.arange(4*dx, 0.0254 + 4*dx, dx)

    x,y = numpy.meshgrid( xarr, yarr )
    x, y = x.ravel(), y.ravel()

    print('Number of bar particles: ', len(x))

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

class TaylorBar(Application):
    def create_particles(self):
        bar = get_bar_particles()
        plate = get_plate_particles()

        # add requisite properties

        # velocity gradient for the bar
        for name in ('v00', 'v01', 'v10', 'v11'):
            bar.add_property(name)

        # deviatoric stress components
        for name in ('s00', 's01', 's02', 's11', 's12', 's22'):
            bar.add_property(name)

        # deviatoric stress accelerations
        for name in ('as00', 'as01', 'as02', 'as11', 'as12', 'as22'):
            bar.add_property(name)

        # deviatoric stress initial values
        for name in ('s000', 's010', 's020', 's110', 's120', 's220'):
            bar.add_property(name)

        bar.add_property('e0')

        # artificial stress properties
        for name in ('r00', 'r01', 'r11'):
            bar.add_property(name)

        # standard acceleration variables and initial values.
        for name in ('arho', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ae',
                     'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'e0'):
            bar.add_property(name)

        return [bar, plate]

    def create_solver(self):
        kernel = WendlandQuintic(dim=2)
        self.wdeltap = kernel.kernel(rij=dx, h=hdx*dx)

        integrator = PECIntegrator(bar=SolidMechStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator)
        dt = 1e-9
        tf = 2.5e-5
        solver.set_time_step(dt)
        solver.set_final_time(tf)
        return solver

    def create_equations(self):
        equations = [

            # Properties computed set from the current state
            Group(
                equations=[
                    # p
                    MieGruneisenEOS(dest='bar', sources=None, gamma=gamma,
                                    r0=r0, c0=C, S=S),

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
                        dest='bar', sources=['bar'], n=4, wdeltap=self.wdeltap),

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
        return equations


if __name__ == '__main__':
    app = TaylorBar()
    app.run()
