"""Two-dimensional dam break over a dry bed.  (30 minutes)

The case is described in "B.Buchner, Green Water on Ship -
Type Off shore Structures, Ph.D.thesis,TU Delft,
Delft University of Technology (2002), Appendix II"

URL: https://repository.tudelft.nl/islandora/object/uuid%3Af0c0bd67-d52a-4b79-8451-1279629a5b80

"""

import os
import numpy as np

from pysph.base.utils import get_particle_array
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.examples.dam_break_2d import DamBreak2D

fluid_column_height = 1.0
H = fluid_column_height
fluid_column_width = 2.0 * H
container_height = 3.0 * H
container_width = 5.366 * H
nboundary_layers = 2
nu = 0.0
dx = 0.03
g = 9.81
ro = 1000.0
vref = np.sqrt(2 * 9.81 * H)
co = 10.0 * vref
gamma = 7.0
alpha = 0.1
beta = 0.0
B = co * co * ro / gamma
p0 = 1000.0
hdx = 1.3
h = hdx * dx
m = dx**2 * ro


class DamBreak2DBuchner(DamBreak2D):
    def add_user_options(self, group):
        super(DamBreak2DBuchner, self).add_user_options(group)

        interp_methods = ['shepard', 'sph', 'order1']
        group.add_argument(
            '--interp-method', action="store", type=str, dest='interp_method',
            default='shepard', help="Specify the interpolation method.",
            choices=interp_methods
        )

    def consume_user_options(self):
        super(DamBreak2DBuchner, self).consume_user_options()
        self.interp_method = self.options.interp_method
        if self.options.scheme != "iisph":
            self.co = co
            self.scheme.configure(c0=co)

    def create_particles(self):
        if self.options.staggered_grid:
            nboundary_layers = 2
            nfluid_offset = 2
            wall_hex_pack = True
        else:
            nboundary_layers = 4
            nfluid_offset = 1
            wall_hex_pack = False

        xt, yt = get_2d_tank(
                             dx=self.dx,
                             length=container_width,
                             height=container_height,
                             base_center=[container_width/2, 0],
                             num_layers=nboundary_layers
                             )

        xf, yf = get_2d_block(
                              dx=self.dx,
                              length=fluid_column_width,
                              height=fluid_column_height,
                              center=[fluid_column_width/2, fluid_column_height/2]
                              )

        xf += self.dx
        yf += self.dx

        fluid = get_particle_array(name='fluid', x=xf, y=yf, h=h, m=m, rho=ro)
        boundary = get_particle_array(name='boundary', x=xt, y=yt, h=h, m=m,
                                      rho=ro)

        self.scheme.setup_properties([fluid, boundary])
        if self.options.scheme == 'iisph':
            # the default position tends to cause the particles to be pushed
            # away from the wall, so displacing it by a tiny amount helps.
            fluid.x += self.dx / 4

        self.kernel_corrections(fluid, boundary)

        return [fluid, boundary]

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        import matplotlib.pyplot as plt
        from pysph.examples import db_exp_data as dbd
        from pysph.tools.interpolator import Interpolator
        H = 1.0
        factor_y = 1/(ro*g*H)
        factor_x = np.sqrt(g/H)

        data_t, data_p0 = dbd.get_buchner_data()
        files = self.output_files

        t = []
        p0 = []
        for sd, arrays1, arrays2 in iter_output(files, "fluid", "boundary"):
            t.append(sd["t"]*factor_x)
            interp = Interpolator([arrays1, arrays2], x=[container_width],
                                  y=[H*0.2], method=self.interp_method)
            p0.append(interp.interpolate('p')*factor_y)

        plt.plot(t, p0, label="Computed")

        fname = os.path.join(self.output_dir, 'results.npz')
        t, p0 = list(map(np.asarray, (t, p0)))
        np.savez(fname, t=t, p0=p0)

        plt.scatter(data_t, data_p0, color=(0, 0, 0), label="Experiment (Buchner, 2002)")
        plt.legend()
        plt.ylabel(r"$\frac{P}{\rho gH}$")
        plt.xlabel(r"$t \sqrt{\frac{g}{H}}$")
        plt.savefig(os.path.join(self.output_dir, 'p_vs_t.png'))


if __name__ == '__main__':
    app = DamBreak2DBuchner()
    app.run()
    app.post_process(app.info_filename)
