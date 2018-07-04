"""Simple compression of a cylindrical charge between two molds

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            OOOOOOOOOOOOOOO
            OOOOOOOOOOOOOOO
            OOOOOOOOOOOOOOO
            OOOOOOOOOOOOOOO
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

"""
import os
import numpy as np

import matplotlib as mpl
mpl.use('pgf')
from matplotlib import pyplot as plt

from pysph.base.kernels import WendlandQuintic
from pysph.sph.integrator import EPECIntegrator
from pysph.base.utils import get_particle_array_wcsph
from pysph.solver.application import Application
from pysph.sph.scheme import WCSPHScheme
from pysph.sph.integrator_step import WCSPHStep
from pysph.solver.utils import iter_output

rho0 = 1000.0
u0 = -0.01
mu = 10
co = 10
hdx = 1.0
dx = 0.004

tf = 3.6

class Compression(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--N", action="store", type=float, dest="N", default=500,
            help="Number of particles per layer."
        )

    def consume_user_options(self):
        self.r = 0.050
        self.R = 0.145
        self.D = 0.040
        self.disp0 = 0.0
        self.disp = []
        self.time = []

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], ['top_wall', 'bottom_wall'], dim=3, rho0=rho0, c0=co,
            h0=dx*hdx, hdx=hdx, gamma=7.0, alpha=0.5, beta=0.0, nu=mu/rho0,
            tensile_correction=True)
        kernel = WendlandQuintic(dim=3)
        dt_cfl = 0.25 * (hdx*dx)/co
        dt_viscous = 0.125 * (hdx*dx)**2*rho0/mu
        print("CFL based time step: %.3E"%dt_cfl)
        print("Viscous time step:   %.3E"%dt_viscous)
        dt = min(dt_cfl,dt_viscous)
        s.configure_solver(
            kernel=kernel, integrator_cls=EPECIntegrator, tf=tf, dt=dt,
            adaptive_timestep=True, n_damp=50, pfreq=200)
        return s

    def create_particles(self):
        golden_ratio = (1+np.sqrt(5))/2
        h0 = hdx * dx
        m0 = rho0 * dx**3

        # FLUID
        n = np.arange(1,self.options.N,1)
        x = np.arange(0.5*dx,self.D-0.25*dx,dx)
        x,n = np.meshgrid(x,n)
        r = self.r*np.sqrt(n/self.options.N)
        theta = 2*np.pi/golden_ratio**2*n
        y = r * np.cos(theta+np.pi/(4*dx)*x)
        z = r * np.sin(theta+np.pi/(4*dx)*x)

        # set up particle properties
        fluid = get_particle_array_wcsph(name='fluid', x=x, y=y, z=z, m=m0,
                    h=h0, rho=rho0)

        # TOP WALL
        N = (self.R/self.r)**2*self.options.N
        n = np.arange(1,N,1)
        x = np.array([self.D+0.5*dx,self.D+1.5*dx,self.D+2.5*dx])
        x,n = np.meshgrid(x,n)
        r = self.R*np.sqrt(n/N)
        theta = 2*np.pi/golden_ratio**2*n
        y = r * np.cos(theta+np.pi/(4*dx)*x)
        z = r * np.sin(theta+np.pi/(4*dx)*x)

        # set up particle properties
        top_wall = get_particle_array_wcsph(name='top_wall', x=x, y=y,
                    z=z, m=m0, h=h0, rho=rho0)

        # BOTTOM WALL
        N = (self.R/self.r)**2*self.options.N
        n = np.arange(1,N,1)
        x = np.array([-2.5*dx,-1.5*dx, -0.5*dx])
        x,n = np.meshgrid(x,n)
        r = self.R*np.sqrt(n/N)
        theta = 2*np.pi/golden_ratio**2*n
        y = r * np.cos(theta+np.pi/(4*dx)*x)
        z = r * np.sin(theta+np.pi/(4*dx)*x)

        # set up particle properties
        bottom_wall = get_particle_array_wcsph(name='bottom_wall', x=x, y=y,
                        z=z, m=m0, h=h0, rho=rho0)

        self.scheme.setup_properties([fluid, top_wall, bottom_wall])
        top_wall.ax[:] = u0
        top_wall.u[:] = u0
        return [fluid, top_wall, bottom_wall]

    def post_process(self, info_file_or_dir):
        # plot pressure force
        time = []
        F = []

        for sd, array in iter_output(self.output_files, 'top_wall'):
            t = sd['t']
            time.append(t)
            p, x = array.get('p', 'x')
            xmin = min(x)
            force = 0.0
            for press, pos in zip(p,x):
                if pos < xmin + dx/4:
                    force += press*dx*dx
            F.append(force)

        plt.figure()
        plt.plot(time, F, '-k')
        plt.title("Compression Force")
        plt.xlabel('Time')
        plt.ylabel('Force')
        plt.tight_layout()
        pdf_fig = os.path.join(self.output_dir, "force.pdf")
        plt.savefig(pdf_fig)
        try:
            tex_fig = os.path.join(self.output_dir, "force.tex")
            from matplotlib2tikz import save as tikz_save
            tikz_save(tex_fig)
        except ImportError:
            print("Did not write tikz figure.")

if __name__ == '__main__':
    app = Compression()
    app.run()
    #app.post_process(app.info_filename)
