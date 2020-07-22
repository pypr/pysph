"""Couette flow with Lees-Edwards boundary conditions.

This is using the transport velocity formulation and Lees-Edwards Boundary
Conditions (300 seconds) to form a shearflow without walls. The shearflow is superposed
with a constant lateral velocity to test particle transition across the boundary.
"""

import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.scheme import TVFScheme

# domain and reference values
Re = 0.0125
d = 0.5
Lx = 2 * d
Ly = 0.4 * Lx
rho0 = 1.0
nu = 0.01

# compute shear rate
Vmax = nu * Re / (2 * d)
gamma = 2 * Vmax / Lx

# compute reference pressure
c0 = 10 * Vmax
p0 = c0 * c0 * rho0

# numerical setup
dx = 0.05
hdx = 1.0

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0 / (c0 + Vmax)
dt_viscous = 0.125 * h0 ** 2 / nu
dt_force = 1.0

# time integration
tf = 3200.0
dt = min(dt_cfl, dt_viscous, dt_force)


class CouetteFlow(Application):
    def create_domain(self):
        """Create a DomainManager with Lees-Edwards BCs.

        Both directions are set as periodic and the value gamma_yx ist set to the
        shear rate to shift particles crossing the x-boundary. As the BC has to keep
        track of time, the time step is passed as well.
        """
        return DomainManager(
            xmin=0,
            xmax=Lx,
            periodic_in_x=True,
            ymin=0,
            ymax=Ly,
            periodic_in_y=True,
            gamma_yx=gamma,
            n_layers=1,
            dt=dt,
        )

    def create_particles(self):
        _x = np.arange(dx / 2, Lx, dx)

        # create the fluid particles
        _y = np.arange(dx / 2, Ly, dx)

        x, y = np.meshgrid(_x, _y)
        fx = x.ravel()
        fy = y.ravel()

        fluid = get_particle_array(
            name="fluid", x=fx, y=fy, rho=rho0 * np.ones_like(fx)
        )

        print("Couette flow :: Re = %g, dt = %g" % (Re, dt))

        self.scheme.setup_properties([fluid])

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * rho0

        # volume is set as dx^2
        fluid.V[:] = 1.0 / volume

        # smoothing lengths
        fluid.h[:] = hdx * dx

        # initial speed
        fluid.v[:] = (fluid.x[:] - Lx / 2) * gamma
        fluid.u[:] = Vmax

        # return the particle list
        return [fluid]

    def create_scheme(self):
        s = TVFScheme(
            ["fluid"], [], dim=2, rho0=rho0, c0=c0, nu=nu, p0=p0, pb=p0, h0=dx * hdx
        )
        s.configure_solver(tf=tf, dt=dt, output_only_real=False)
        return s


if __name__ == "__main__":
    app = CouetteFlow()
    app.run()
