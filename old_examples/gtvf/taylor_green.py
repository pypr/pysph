"""
Taylor-Green instability using GTVF.
"""
from pysph.sph.wc.gtvf import GTVFScheme, get_particle_array_gtvf
from pysph.examples.taylor_green import TaylorGreen
from pysph.base.kernels import WendlandQuintic


# domain and constants
L = 1.0
U = 1.0
rho0 = 1.0
c0 = 10 * U
p0 = rho0*c0*c0
hdx = 1.3


class TaylorGreenGTVF(TaylorGreen):
    def create_particles(self):
        [pa] = super(TaylorGreenGTVF, self).create_particles()
        fluid = get_particle_array_gtvf(
            name='fluid', x=pa.x, y=pa.y, m=pa.m, rho=pa.rho, h=pa.h,
            u=pa.u, v=pa.v, p=pa.p
        )
        return [fluid]

    def create_scheme(self):
        s = super(TaylorGreenGTVF, self).create_scheme()
        gtvf = GTVFScheme(fluids=['fluid'], solids=[], dim=2, rho0=rho0, c0=c0,
                          nu=None, h0=None, p0=p0, pref=None)
        s.schemes.update(gtvf=gtvf)
        return s

    def configure_scheme(self):
        kernel = WendlandQuintic(dim=2)
        scheme = self.scheme
        self.hdx = hdx
        h0 = self.hdx * self.dx
        dt_cfl = 0.25 * h0 / (c0 + U)
        dt_viscous = 0.125 * h0**2 / self.nu
        dt_force = 0.25 * 1.0
        self.dt = min(dt_cfl, dt_viscous, dt_force)
        scheme.configure(nu=self.nu, h0=h0, pref=p0)
        scheme.configure_solver(kernel=kernel, tf=self.tf, dt=self.dt)


if __name__ == '__main__':
    app = TaylorGreenGTVF()
    app.run()
    app.post_process(app.info_filename)
