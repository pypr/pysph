"""Colliding Elastic Balls. (10 minutes)
"""

import numpy

# SPH equations
from pysph.sph.equation import Group
from pysph.sph.basic_equations import IsothermalEOS, ContinuityEquation, MonaghanArtificialViscosity,\
     XSPHCorrection, VelocityGradient2D
from pysph.sph.solid_mech.basic import MomentumEquationWithStress2D, HookesDeviatoricStressRate2D,\
    MonaghanArtificialStress

from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import SolidMechStep

def get_K(G, nu):
    ''' Get the bulk modulus from shear modulus and Poisson ratio '''
    return 2.0*G*(1+nu)/(3*(1-2*nu))

def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


# constants
E = 1e7
nu = 0.3975
G = E/(2.0*(1+nu))
K = get_K(G, nu)
rho0 = 1.0
c0 = numpy.sqrt(K/rho0)

dx = 0.0005
hdx = 1.5

# geometry
ri = 0.03
ro = 0.04

class Rings(Application):
    def create_particles(self):
        #x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -0.105:0.105+1e-4:dx]
        spacing = 0.041 # spacing = 2*5cm

        x,y = numpy.mgrid[-ro:ro:dx, -ro:ro:dx]
        x = x.ravel()
        y = y.ravel()

        d = (x*x+y*y)
        keep = numpy.flatnonzero((ri*ri<=d) * (d<ro*ro))
        x = x[keep]
        y = y[keep]

        x = numpy.concatenate([x-spacing,x+spacing])
        y = numpy.concatenate([y,y])

        print('Ellastic Collision with %d particles'%(x.size))
        print("Shear modulus G = %g, Young's modulus = %g, Poisson's ratio =%g"%(G,E,nu))

        #print bdry, numpy.flatnonzero(bdry)
        m = numpy.ones_like(x)*dx*dx
        h = numpy.ones_like(x)*hdx*dx
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

        pa.cs[:] = c0
        pa.u = pa.cs*u_f*(2*(x<0)-1)

        # add requisite properties

        # sound speed etc.
        add_properties(pa, 'cs', 'e' )

        # velocity gradient properties
        add_properties(pa, 'v00', 'v01', 'v10', 'v11')

        # artificial stress properties
        add_properties(pa, 'r00', 'r01', 'r02', 'r11', 'r12', 'r22')

        # deviatoric stress components
        add_properties(pa, 's00', 's01', 's02', 's11', 's12', 's22')

        # deviatoric stress accelerations
        add_properties(pa, 'as00', 'as01', 'as02', 'as11', 'as12', 'as22')

        # deviatoric stress initial values
        add_properties(pa, 's000', 's010', 's020', 's110', 's120', 's220')

        # standard acceleration variables
        add_properties(pa, 'arho', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ae')

        # initial values
        add_properties(pa, 'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'e0')

        # load balancing properties
        pa.set_lb_props( list(pa.properties.keys()) )

        return [pa,]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        self.wdeltap = kernel.kernel(rij=dx, h=hdx*dx)

        integrator = PECIntegrator(solid=SolidMechStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator)

        dt = 1e-8
        tf = 5e-5
        solver.set_time_step(dt)
        solver.set_final_time(tf)
        solver.set_print_freq(500)
        return solver

    def create_equations(self):
        equations = [

            # Properties computed set from the current state
            Group(
                equations=[
                    # p
                    IsothermalEOS(dest='solid', sources=None,
                                  rho0=rho0, c0=c0, p0=0.0),

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
                        dest='solid', sources=['solid',], n=4, wdeltap=self.wdeltap),

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
        return equations


if __name__ == '__main__':
    app = Rings()
    app.run()
