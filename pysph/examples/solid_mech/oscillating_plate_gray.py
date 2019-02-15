import numpy as np
from math import cos, sin, cosh, sinh

# SPH equations
from pysph.sph.equation import Group
from pysph.sph.basic_equations import (IsothermalEOS, ContinuityEquation,
                                       MonaghanArtificialViscosity,
                                       XSPHCorrection, VelocityGradient2D)
from pysph.sph.solid_mech.basic import (MomentumEquationWithStress,
                                        HookesDeviatoricStressRate,
                                        MonaghanArtificialStress)

from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import SolidMechStep


def get_bulk_mod(G, nu):
    ''' Get the bulk modulus from shear modulus and Poisson ratio '''
    return 2.0 * G * (1 + nu) / (3 * (1 - 2 * nu))


def get_speed_of_sound(E, nu, rho0):
    return np.sqrt(E / (3 * (1. - 2 * nu) * rho0))


def add_properties(pa, *props):
    for prop in props:
        pa.add_property(name=prop)


class OscillatingPlate(Application):
    def initialize(self):
        self.L = 0.2
        self.H = 0.02
        # wave number K
        self.KL = 1.875
        self.K = 1.875 / self.L

        # edge velocity of the plate (m/s)
        self.Vf = 0.05
        self.dx_plate = 0.002
        self.h = 1.3 * self.dx_plate
        self.plate_rho0 = 1000.
        self.plate_E = 2. * 1e6
        self.plate_nu = 0.3975
        self.plate_G = self.plate_E / (2. * (1. + self.plate_nu))
        self.cs = get_speed_of_sound(self.plate_E, self.plate_nu,
                                     self.plate_rho0)

        # bulk modulus
        # self.plate_bulk_mod = get_bulk_mod(self.plate_G, self.plate_nu)
        self.plate_inside_wall_length = self.L / 4.
        self.wall_layers = 3

    def create_particles(self):
        # plate coordinates
        xp, yp = np.mgrid[-self.plate_inside_wall_length:self.L +
                          self.dx_plate / 2.:self.dx_plate, -self.H / 2.:
                          self.H / 2. + self.dx_plate / 2.:self.dx_plate]
        xp = xp.ravel()
        yp = yp.ravel()

        m = self.plate_rho0 * self.dx_plate**2.
        # speed of sound
        cs = get_speed_of_sound(self.plate_E, self.plate_nu, self.plate_rho0)
        ##################################
        # vertical velocity of the plate #
        ##################################
        # initialize with zero at the beginning
        v = np.zeros_like(xp)
        v = v.ravel()

        # set the vertical velocity for particles which are only
        # out of the wall
        K = self.K
        # L = self.L
        KL = self.KL
        M = sin(KL) + sinh(KL)
        N = cos(KL) + cosh(KL)
        Q = 2 * (cos(KL) * sinh(KL) - sin(KL) * cosh(KL))
        for i in range(len(v)):
            if xp[i] > 0.:
                # set the velocity
                tmp1 = (cos(K * xp[i]) - cosh(K * xp[i]))
                tmp2 = (sin(K * xp[i]) - sinh(K * xp[i]))
                v[i] = self.Vf * cs * (M * tmp1 - N * tmp2) / Q

        plate = get_particle_array(x=xp, y=yp, v=v, m=m, h=self.h,
                                   rho=self.plate_rho0, cs=cs, name="plate")
        # get the index of the particle which will be used to compute the
        # amplitude
        xp_max = max(xp)
        fltr = np.argwhere(xp == xp_max)
        fltr_idx = int(len(fltr) / 2.)
        amplitude_idx = fltr[fltr_idx][0]

        # add properties
        add_properties(plate, 'cs', 'e', 'v00', 'v01', 'v02', 'v10', 'v11',
                       'v12', 'v20', 'v21', 'v22', 'r00', 'r01', 'r02', 'r11',
                       'r12', 'r22', 's00', 's01', 's02', 's11', 's12', 's22',
                       'as00', 'as01', 'as02', 'as11', 'as12', 'as22', 's000',
                       's010', 's020', 's110', 's120', 's220', 'arho', 'au',
                       'av', 'aw', 'ax', 'ay', 'az', 'ae', 'rho0', 'u0', 'v0',
                       'w0', 'x0', 'y0', 'z0', 'e0')
        plate.add_constant("amplitude_idx", amplitude_idx)

        # get the minimum and maximum of the plate
        xp_min = xp.min()
        yp_min = yp.min()
        yp_max = yp.max()
        xw_upper, yw_upper = np.mgrid[
            -self.plate_inside_wall_length:self.dx_plate / 2.:self.dx_plate,
            yp_max + self.dx_plate:yp_max + self.dx_plate +
            (self.wall_layers - 1) * self.dx_plate + self.dx_plate / 2.:
            self.dx_plate]
        xw_upper = xw_upper.ravel()
        yw_upper = yw_upper.ravel()

        xw_lower, yw_lower = np.mgrid[
            -self.plate_inside_wall_length:self.dx_plate / 2.:self.dx_plate,
            yp_min - self.dx_plate:yp_min - self.dx_plate -
            (self.wall_layers - 1) * self.dx_plate - self.dx_plate / 2.:
            -self.dx_plate]
        xw_lower = xw_lower.ravel()
        yw_lower = yw_lower.ravel()

        xw_left_max = xp_min - self.dx_plate
        xw_left_min = xw_left_max - (
            self.wall_layers - 1) * self.dx_plate - self.dx_plate / 2.
        yw_left_max = yw_upper.max() + self.dx_plate / 2.
        yw_left_min = yw_lower.min()

        xw_left, yw_left = np.mgrid[xw_left_max:xw_left_min:-self.dx_plate,
                                    yw_left_min:yw_left_max:self.dx_plate]
        xw_left = xw_left.ravel()
        yw_left = yw_left.ravel()

        # wall coordinates
        xw, yw = np.concatenate((xw_lower, xw_upper, xw_left)), np.concatenate(
            (yw_lower, yw_upper, yw_left))

        # create the particle array
        wall = get_particle_array(x=xw, y=yw, m=m, h=self.h,
                                  rho=self.plate_rho0, name="wall")

        add_properties(wall, 'cs', 'e', 'v00', 'v01', 'v02', 'v10', 'v11',
                       'v12', 'v20', 'v21', 'v22', 'r00', 'r01', 'r02', 'r11',
                       'r12', 'r22', 's00', 's01', 's02', 's11', 's12', 's22',
                       'as00', 'as01', 'as02', 'as11', 'as12', 'as22', 's000',
                       's010', 's020', 's110', 's120', 's220', 'arho', 'au',
                       'av', 'aw', 'ax', 'ay', 'az', 'ae', 'rho0', 'u0', 'v0',
                       'w0', 'x0', 'y0', 'z0', 'e0')
        wall.set_lb_props(list(wall.properties.keys()))

        return [plate, wall]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        self.wdeltap = kernel.kernel(rij=self.dx_plate, h=self.h)

        integrator = EPECIntegrator(plate=SolidMechStep())

        dt = 1e-6
        tf = 1

        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False)
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    # p
                    IsothermalEOS(dest='plate', sources=None,
                                  rho0=self.plate_rho0, c0=self.cs, p0=0.0),

                    # vi,j : requires properties v00, v01, v10, v11
                    VelocityGradient2D(dest='plate', sources=['plate',
                                                              'wall']),

                    # rij : requires properties r00, r01, r02, r11, r12, r22,
                    #                           s00, s01, s02, s11, s12, s22
                    MonaghanArtificialStress(dest='plate', sources=None,
                                             eps=0.3),
                ], ),

            # Acceleration variables are now computed
            Group(equations=[

                # arho
                ContinuityEquation(dest='plate', sources=['plate', 'wall']),

                # au, av
                MomentumEquationWithStress(dest='plate', sources=[
                    'plate', 'wall'
                ], n=4, wdeltap=self.wdeltap),

                # au, av
                MonaghanArtificialViscosity(dest='plate', sources=[
                    'plate', 'wall'
                ], alpha=1.0, beta=1.0),

                # a_s00, a_s01, a_s11
                HookesDeviatoricStressRate(dest='plate', sources=None,
                                           shear_mod=self.plate_G),

                # ax, ay, az
                XSPHCorrection(dest='plate', sources=[
                    'plate',
                ], eps=0.5),
            ])
        ]
        return equations

    def post_process(self):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        t, amplitude = [], []
        for sd, array in iter_output(files, 'plate'):
            _t = sd['t']
            t.append(_t)
            amplitude.append(array.y[array.amplitude_idx[0]])

        import matplotlib
        import os
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.clf()
        plt.plot(t, amplitude)
        plt.xlabel('t')
        plt.ylabel('Amplitude')
        plt.legend()
        fig = os.path.join(self.output_dir, "amplitude.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = OscillatingPlate()
    app.run()
    app.post_process()
