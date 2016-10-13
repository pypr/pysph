"""Simulate the Robert's problem (1D) (40 seconds).
"""
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import ADKEScheme, GasDScheme, SchemeChooser
import numpy

# Numerical constants
dim = 1
gamma = 1.4
gamma1 = gamma - 1.0

# solution parameters
dt = 1e-4
tf = 0.1

# domain size and discretization parameters
xmin = -0.5; xmax = 0.5
nl = 1930; nr = 1000.0
dxl = 0.5/nl; dxr = 1/nr

kernel_factor = 2.0
h0 = kernel_factor*dxr


class Robert(ShockTubeSetup):
    def create_particles(self):
        return self.generate_particles(xmin = -0.5, xmax=1, dxl=dxl, dxr=dxr,
                m=dxr, pl=10.33, pr=1.0, h0=h0, bx=0.03, gamma1=gamma1,
                ul=-0.39, ur=-3.02)

    def create_scheme(self):
        s.configure_solver(dt=dt, tf=tf, adaptive_timestep=False, pfreq=50)
        return s

    def create_scheme(self):
        self.dt = dt
        self.tf = tf
        adke = ADKEScheme(
            fluids=['fluid'], solids=['boundary'], dim=dim, gamma=gamma, alpha=1,
            beta=2.0, k=1.0, eps=0.5, g1=0.5, g2=1.0)


        mpm = GasDScheme(
                fluids=['fluid'], solids=[], dim=dim, gamma=gamma,
                kernel_factor=kernel_factor, alpha1=1.0, alpha2=0.1,
                beta=2.0, update_alpha1=True, update_alpha2=True
                )
        s = SchemeChooser(default='adke', adke=adke, mpm=mpm)
        return s




if __name__ == '__main__':
    app = Robert()
    app.run()
    app.post_process()
