"""A wind-tunnel flow past an array of cylinders.  (20 minutes)

This example demonstrates how one can use the inlet and outlet feature of PySPH
to simulate flow inside wind tunnel. For simplicity the tunnel is periodic
along the y-axis, this is reasonable if the tunnel width is large. The fluid is
initialized with a unit velocity along the x-axis and impulsively started and
the cylinder is of unit radius. The inlet produces an incoming stream of
particles and the outlet consumes these particles. The TVF scheme is used by
default. Note that the create_equations method does some surgery of the
scheme-generated equations to adapt them for this problem.

"""
import numpy as np

from pysph.base.utils import get_particle_array
from pysph.base.nnps import DomainManager
from pysph.solver.application import Application
from pysph.sph.scheme import TVFScheme
from pysph.tools import geometry as G
from pysph.sph.simple_inlet_outlet import SimpleInlet, SimpleOutlet
from pysph.sph.integrator_step import InletOutletStep

# Geometric parameters
l_tunnel = 15.0
w_tunnel = 7.0
chord = 2.0  # diameter of circle
center = l_tunnel/3., w_tunnel/2.  # center of circle
n_inlet = 6  # Number of inlet layers
n_outlet = 6  # Number of outlet layers

# Fluid mechanical/numerical parameters
re = 1e3
dx = 0.1
hdx = 1.2
rho = 1000
umax = 1.0
tf = 5.0

# Computed parameters
c0 = 1.5*umax*10
p0 = rho*c0*c0
h0 = dx*hdx
nu = umax*chord/re
dt_cfl = 0.25 * h0/(c0 + umax)
dt_viscous = 0.125 * h0**2/nu
dt = 0.5*min(dt_cfl, dt_viscous)


class WindTunnel(Application):
    def create_domain(self):
        i_ghost = n_inlet*dx
        o_ghost = n_outlet*dx
        domain = DomainManager(
            xmin=-i_ghost, xmax=l_tunnel+o_ghost, ymin=0, ymax=w_tunnel,
            periodic_in_y=True
        )
        return domain

    def create_particles(self):
        x, y = np.mgrid[dx:l_tunnel+dx/4:dx, dx/2:w_tunnel-dx/4:dx]
        x, y = (np.ravel(t) for t in (x, y))
        one = np.ones_like(x)
        volume = dx*dx*one
        m = volume*rho
        fluid = get_particle_array(
            name='fluid', m=m, x=x, y=y, h=h0*one, V=1.0/volume,
            u=umax*one
        )

        xc, yc = center
        cond = (x - xc)**2 + (y - yc)**2 < 1.0
        one = np.ones_like(x[cond])
        volume = dx*dx*one
        solid = get_particle_array(
            name='solid', x=x[cond].ravel(), y=y[cond].ravel(), m=volume*rho,
            rho=one*rho, h=h0*one, V=1.0/volume
        )
        G.remove_overlap_particles(fluid, solid, dx, dim=2)

        x, y = np.mgrid[dx:n_outlet*dx:dx, dx/2:w_tunnel-dx/4:dx]
        x, y = (np.ravel(t) for t in (x, y))
        x += l_tunnel
        one = np.ones_like(x)
        volume = dx*dx*one
        m = volume*rho
        outlet = get_particle_array(
            name='outlet', x=x, y=y, m=m, h=h0*one, V=1.0/volume, u=umax*one
        )

        # Setup the inlet particle array with just the particles we need at the
        # exit plane which is replicated by the inlet.
        y = np.arange(dx/2, w_tunnel - dx/4.0, dx)
        x = np.zeros_like(y)
        one = np.ones_like(x)
        volume = one*dx*dx

        inlet = get_particle_array(
            name='inlet', x=x, y=y, m=volume*rho,
            h=h0*one, u=umax*one, rho=rho*one,
            V=1.0/volume
        )
        self.scheme.setup_properties([fluid, inlet, outlet, solid])
        for p in fluid.properties:
            if p not in outlet.properties:
                outlet.add_property(p)

        return [fluid, solid, inlet, outlet]

    def create_scheme(self):
        s = TVFScheme(
            ['fluid', 'inlet', 'outlet'], ['solid'],
            dim=2, rho0=rho, c0=c0, nu=nu, p0=p0, pb=p0, h0=dx*hdx, gx=0.0
        )
        extra_steppers = dict(
            inlet=InletOutletStep(), outlet=InletOutletStep()
        )
        s.configure_solver(
            extra_steppers=extra_steppers, tf=tf, dt=dt, n_damp=10, pfreq=100
        )
        return s

    def create_inlet_outlet(self, particle_arrays):
        f_pa = particle_arrays['fluid']
        i_pa = particle_arrays['inlet']
        o_pa = particle_arrays['outlet']

        xmin = -dx*n_inlet
        inlet = SimpleInlet(
            i_pa, f_pa, spacing=dx, n=n_inlet, axis='x', xmin=xmin, xmax=0.0,
            ymin=0.0, ymax=w_tunnel
        )
        xmax = l_tunnel + dx*n_outlet
        outlet = SimpleOutlet(
            o_pa, f_pa, xmin=l_tunnel, xmax=xmax, ymin=0.0, ymax=w_tunnel
        )
        return [inlet, outlet]

    def create_equations(self):
        eqs = self.scheme.get_equations()
        # print(eqs)  # Print the equations for your understanding.
        g0 = eqs[0]
        # Remove all the unnecessary summation density equations for the inlet
        # and outlet.
        del g0.equations[1:]
        g1 = eqs[1]
        # Remove the state equations for inlet and outlet.
        del g1.equations[1:]
        g3 = eqs[3]
        # Remove the momentum and other equations for inlet and outlet
        del g3.equations[4:]
        return eqs


if __name__ == '__main__':
    app = WindTunnel()
    app.run()
