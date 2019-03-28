'''The standard Sphysics dam break benchmark. (4 hours)

This is intended to be the same as the standard dam break from the
DualSPhysics code just for comparison.
'''
import numpy as np

from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.scheme import WCSPHScheme


def ravel(*args):
    return tuple(np.ravel(x) for x in args)


def rhstack(*args):
    '''Join given set of args, we that each element in args has the same shape.
    Each argument is first ravelled and then stacked.
    '''
    return tuple(np.hstack(ravel(*t)) for t in zip(*args))


class DamBreak(Application):
    def add_user_options(self, group):
        group.add_argument(
            '--dx', action='store', type=float, dest='dx',  default=0.0085,
            help='Particle spacing.'
        )
        hdx = np.sqrt(3)
        group.add_argument(
            '--hdx', action='store', type=float, dest='hdx',  default=hdx,
            help='Specify the hdx factor where h = hdx * dx.'
        )

    def consume_user_options(self):
        dx = self.options.dx
        self.dx = dx
        self.hdx = self.options.hdx

    def create_scheme(self):
        self.c0 = c0 = 10.0 * np.sqrt(2.0*9.81*0.3)
        self.hdx = hdx = np.sqrt(3)
        dx = 0.01
        h0 = hdx*dx
        alpha = 0.1
        beta = 0.0
        gamma = 7.0
        s = WCSPHScheme(
            ['fluid'], ['boundary'], dim=3, rho0=1000, c0=c0,
            h0=h0, hdx=hdx, gz=-9.81, alpha=alpha, beta=beta, gamma=gamma,
            hg_correction=True, tensile_correction=False
        )
        return s

    def configure_scheme(self):
        s = self.scheme
        h0 = self.dx * self.hdx
        s.configure(h0=h0, hdx=self.hdx)
        dt = 0.25*h0/(1.1 * self.c0)
        tf = 1.5
        s.configure_solver(
            tf=tf, dt=dt, adaptive_timestep=True, n_damp=50
        )

    def create_particles(self):
        dx = self.dx
        l, b, h = 1.6, 0.67, 0.4
        lw, hw = 0.4, 0.3

        # Big filled vessel with staggered points.
        x, y, z = np.mgrid[0:l+dx:dx, 0:b+dx:dx, 0:h:dx]

        # The post
        x3, y3, z3 = np.mgrid[0.9:1.02:dx, 0.25:0.37:dx, dx:0.45:dx]
        xmax, ymax, zmax = max(x3.flat), max(y3.flat), max(z3.flat)
        post_cond = ~((x3 > 0.9) & (x3 < xmax) & (y3 > 0.25) &
                      (y3 < ymax) & (z3 < zmax))
        p_post = x3[post_cond], y3[post_cond], z3[post_cond]

        # Masks to extract different parts from the vessel.
        wcond = ((x > 0) & (x < lw) & (y > 0) &
                 (y < b) & (z > 0) & (z < hw))
        box = ~((x > 0) & (x <= l) & (y > 0) & (y < b) & (z > 0) & (z <= h))

        p_box = x[box], y[box], z[box]
        xf, yf, zf = x[wcond], y[wcond], z[wcond]

        xs, ys, zs = rhstack(p_box, p_post)

        vol = dx**3
        m = vol*1000
        f = get_particle_array(
            name='fluid', x=xf, y=yf, z=zf, m=m, h=dx*self.hdx,
            rho=1000.0
        )
        b = get_particle_array(
            name='boundary', x=xs, y=ys, z=zs, m=m, h=dx*self.hdx,
            rho=1000.0
        )

        self.scheme.setup_properties([f, b])
        return [f, b]

    def customize_output(self):
        self._mayavi_config('''
        viewer.scalar = 'vmag'
        b = particle_arrays['boundary']
        b.plot.actor.mapper.scalar_visibility = False
        b.plot.actor.property.opacity = 0.15
        ''')


if __name__ == '__main__':
    app = DamBreak()
    app.run()
