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

rho0 = 1000.0
mu = 10
Vmax = 0.2
co = 10*Vmax
hdx = 1.2
dx = 0.003

t0 = -2.5
tf = 5.5
tdamp = 0

class Compression(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--N", action="store", type=float, dest="N", default=300,
            help="Number of particles."
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
            h0=dx*hdx, hdx=1.5, gamma=7.0, alpha=0.5, beta=0.0, nu=mu/rho0,
            hg_correction=True, tensile_correction=True)
        kernel = WendlandQuintic(dim=3)
        dt_cfl = 0.25 * (hdx*dx)/( co + Vmax )
        dt_viscous = 0.125 * (hdx*dx)**2*rho0/mu
        print("CFL based time step: %.3E"%dt_cfl)
        print("Viscous time step:   %.3E"%dt_viscous)
        dt = min(dt_cfl,dt_viscous)
        s.configure_solver(
            kernel=kernel, integrator_cls=EPECIntegrator, tf=tf, dt=dt,
            adaptive_timestep=True, n_damp=50
        )
        return s

    def create_particles(self):
        golden_ratio = (1+np.sqrt(5))/2

        # FLUID
        n = np.arange(0,self.options.N,1)
        x = np.arange(0,self.D,2*dx)
        x,n = np.meshgrid(x,n)
        r = self.r*np.sqrt(n/self.options.N)
        theta = 2*np.pi/golden_ratio**2*n
        y = r * np.cos(theta+np.pi/(4*dx)*x)
        z = r * np.sin(theta+np.pi/(4*dx)*x)

        # set up particle properties
        h0 = hdx * dx
        volume = dx**3
        m0 = rho0 * volume
        fluid = get_particle_array_wcsph(name='fluid', x=x, y=y, z=z)
        fluid.m[:] = m0
        fluid.h[:] = h0
        fluid.rho[:] = rho0

        # TOP WALL
        N = (self.R/self.r)**2*self.options.N
        n = np.arange(0,N,1)
        x = np.arange(self.D,self.D+8*dx,2*dx)
        x,n = np.meshgrid(x,n)
        r = self.R*np.sqrt(n/N)
        theta = 2*np.pi/golden_ratio**2*n
        y = r * np.cos(theta+np.pi/(4*dx)*x)
        z = r * np.sin(theta+np.pi/(4*dx)*x)

        # set up particle properties
        h0 = hdx * dx
        volume = dx**3
        m0 = rho0 * volume
        top_wall = get_particle_array_wcsph(name='top_wall', x=x, y=y, z=z)
        top_wall.m[:] = m0
        top_wall.h[:] = h0
        top_wall.rho[:] = rho0
        #top_wall.u[:] = self.D/t0

        # BOTTOM WALL
        N = (self.R/self.r)**2*self.options.N
        n = np.arange(0,N,1)
        x = np.arange(-8*dx,0,2*dx)
        x,n = np.meshgrid(x,n)
        r = self.R*np.sqrt(n/N)
        theta = 2*np.pi/golden_ratio**2*n
        y = r * np.cos(theta+np.pi/(4*dx)*x)
        z = r * np.sin(theta+np.pi/(4*dx)*x)

        # set up particle properties
        h0 = hdx * dx
        volume = dx**3
        m0 = rho0 * volume
        bottom_wall = get_particle_array_wcsph(name='bottom_wall', x=x, y=y, z=z)
        bottom_wall.m[:] = m0
        bottom_wall.h[:] = h0
        bottom_wall.rho[:] = rho0

        self.scheme.setup_properties([fluid, top_wall, bottom_wall])
        return [fluid, top_wall, bottom_wall]

    def post_stage(self, current_time, dt, stage):
        # prescribed motion of the top mold
        if stage == 1:
            return

        top_pa = next((p for p in self.particles if p.name=='top_wall'),None)
        if current_time < tdamp:
            damping_factor = 0.5*(np.sin((-0.5+current_time/tdamp)*np.pi)+1.0)
        else:
            damping_factor = 1.0

        top_pa.u[:] = self.D/t0*np.exp(current_time/t0)*damping_factor
        top_pa.x[:] += top_pa.u[:]*dt
        self.disp.append(top_pa.x[0])
        self.time.append(current_time)

    def post_process(self, info_file_or_dir):
        dd = self.D*np.exp(np.array(self.time)/t0)
        plt.figure(figsize=(5,2))
        plt.plot(self.time[::10], self.disp[::10], '.k')
        plt.plot(self.time, dd, '--k')
        plt.title("Displacement of top mold")
        plt.xlabel('Time')
        plt.ylabel('Displacement')
        plt.tight_layout()
        pdf_fig = os.path.join(self.output_dir, "displacement.pdf")
        plt.savefig(pdf_fig)
        try:
            tex_fig = os.path.join(self.output_dir, "displacement.tex")
            from matplotlib2tikz import save as tikz_save
            tikz_save(tex_fig)
        except ImportError:
            print("Did not write tikz figure.")


if __name__ == '__main__':
    app = Compression()
    app.run()
    app.post_process(app.info_filename)
