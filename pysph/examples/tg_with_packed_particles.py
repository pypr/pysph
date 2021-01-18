from numpy import pi, sin, cos, exp

from pysph.examples.taylor_green import TaylorGreen, exact_solution
from pysph.base.utils import get_particle_array

# domain and constants
L = 1.0
U = 1.0
rho0 = 1.0
c0 = 10 * U
p0 = c0**2 * rho0


class TGPacked(TaylorGreen):
    def _get_packed_points(self):
        '''
        returns
        xs, ys, zs, xf, yf, zf
        '''
        from pysph.tools.geometry import get_packed_periodic_packed_particles
        folder = self.output_dir
        dx = self.dx
        return get_packed_periodic_packed_particles(
            self.add_user_options, folder, dx, L=L, B=L)

    def create_fluid(self):
        # create the particles
        xs, ys, zs, xf, yf, zf = self._get_packed_points()
        x, y = xf, yf
        if self.options.init is not None:
            fname = self.options.init
            from pysph.solver.utils import load
            data = load(fname)
            _f = data['arrays']['fluid']
            x, y = _f.x.copy(), _f.y.copy()

        # Initialize
        dx = self.dx
        m = self.volume * rho0
        h = self.hdx * dx
        re = self.options.re
        b = -8.0 * pi * pi / re
        u0, v0, p0 = exact_solution(U=U, b=b, t=0, x=x, y=y)
        color0 = cos(2 * pi * x) * cos(4 * pi * y)

        # create the arrays
        fluid = get_particle_array(
            name='fluid', x=x, y=y, m=m, h=h, u=u0, v=v0, rho=rho0, p=p0,
            color=color0)
        return fluid


if __name__ == '__main__':
    app = TGPacked()
    app.run()
    app.post_process(app.info_filename)
