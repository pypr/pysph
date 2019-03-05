import numpy as np
from math import cos, sin, cosh, sinh

# SPH equations
from pysph.sph.solid_mech.basic import (ElasticSolidsScheme,
                                        get_particle_array_elastic_dynamics)

from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application


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

        # bulk modulus
        self.plate_inside_wall_length = self.L / 4.
        self.wall_layers = 3

        self.alpha = 1.0
        self.beta = 1.0
        self.xsph_eps = 0.5
        self.artificial_stress_eps = 0.3

        self.tf = 1.0
        self.dt = 1e-5

    def create_particles(self):
        # plate coordinates
        xp, yp = np.mgrid[-self.plate_inside_wall_length:self.L +
                          self.dx_plate / 2.:self.dx_plate, -self.H /
                          2.:self.H / 2. + self.dx_plate / 2.:self.dx_plate]
        xp = xp.ravel()
        yp = yp.ravel()

        m = self.plate_rho0 * self.dx_plate**2.

        # get the index of the particle which will be used to compute the
        # amplitude
        xp_max = max(xp)
        fltr = np.argwhere(xp == xp_max)
        fltr_idx = int(len(fltr) / 2.)
        amplitude_idx = fltr[fltr_idx][0]

        kernel = CubicSpline(dim=2)
        self.wdeltap = kernel.kernel(rij=self.dx_plate, h=self.h)
        plate = get_particle_array_elastic_dynamics(
            x=xp, y=yp, m=m, h=self.h,
            rho=self.plate_rho0, name="plate", constants=dict(
                wdeltap=self.wdeltap, n=4, rho_ref=self.plate_rho0,
                E=self.plate_E, nu=self.plate_nu, amplitude_idx=amplitude_idx))

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
                v[i] = self.Vf * plate.cs[0] * (M * tmp1 - N * tmp2) / Q

        # set vertical velocity
        plate.v = v

        # #########################################
        # #### Create the wall particle array #####
        # #########################################
        # get the minimum and maximum of the plate
        xp_min = xp.min()
        yp_min = yp.min()
        yp_max = yp.max()
        xw_upper, yw_upper = np.mgrid[-self.plate_inside_wall_length:self.
                                      dx_plate / 2.:self.dx_plate, yp_max +
                                      self.dx_plate:yp_max + self.dx_plate +
                                      (self.wall_layers - 1) * self.dx_plate +
                                      self.dx_plate / 2.:self.dx_plate]
        xw_upper = xw_upper.ravel()
        yw_upper = yw_upper.ravel()

        xw_lower, yw_lower = np.mgrid[-self.plate_inside_wall_length:self.
                                      dx_plate / 2.:self.dx_plate, yp_min -
                                      self.dx_plate:yp_min - self.dx_plate -
                                      (self.wall_layers - 1) * self.dx_plate -
                                      self.dx_plate / 2.:-self.dx_plate]
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
        wall = get_particle_array_elastic_dynamics(
            x=xw, y=yw, m=m, h=self.h, rho=self.plate_rho0, name="wall",
            constants=dict(E=self.plate_E, nu=self.plate_nu))

        return [plate, wall]

    def create_scheme(self):
        s = ElasticSolidsScheme(elastic_solids=['plate'], solids=['wall'], dim=2)
        s.configure_solver(dt=self.dt, tf=self.tf)
        return s

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
