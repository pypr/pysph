"""Incompressible flow past a periodic array of cylinders. (42 hours)


See Ellero and Adams, International Journal for Numerical Methods in
Engineering, 2011, vol 86, pp 1027-1040 for the detailed parameters for this
problem and also  Adami, Hu and Adams, JCP, 2013, vol 241, pp 292-307.

In particular, we note that we set c0 from Ellero and Adams as using the
value from Adami et al. will cause the solution to blow up.

If one sets c0=10*Umax and sets pb=300*p0, that will cause particles to void
at the rear of the cylinder.

"""

import os
import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application

from pysph.sph.scheme import TVFScheme


# domain and reference values
L = 0.12; Umax = 1.2e-4
a = 0.02; H = 4*a
fx = 2.5e-4

# c0 is set from Ellero and Adams.
# Note that setting this to 0.1*np.sqrt(a*fx) as per Adami Hu and Adams is
# incorrect and will actually cause a blow up of the solution.
c0 = 0.02
rho0 = 1000.0
p0 = c0*c0*rho0
pb = p0

# Reynolds number and kinematic viscosity
nu = 0.1/rho0; Re = a*Umax/nu

# Numerical setup
nx = 144; dx = L/nx
ghost_extent = 5 * 1.5 * dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0/( c0 + Umax )
dt_viscous = 0.125 * h0**2/nu
dt_force = 0.25 * np.sqrt(h0/abs(fx))

T = a/Umax

tf = 2.5*T
dt = min(dt_cfl, dt_viscous, dt_force)


class PeriodicCylinders(Application):

    def create_domain(self):
        # domain for periodicity
        domain = DomainManager(xmin=0, xmax=L, periodic_in_x=True)
        return domain

    def create_particles(self):
        # create all the particles
        _x = np.arange( dx/2, L, dx )
        _y = np.arange( -ghost_extent, H+ghost_extent, dx )
        x, y = np.meshgrid(_x, _y); x = x.ravel(); y = y.ravel()

        # sort out the fluid and the solid
        indices = []
        cx = 0.5 * L; cy = 0.5 * H
        for i in range(x.size):
            xi = x[i]; yi = y[i]
            if ( np.sqrt( (xi-cx)**2 + (yi-cy)**2 ) > a ):
                if ( (yi > 0) and (yi < H) ):
                    indices.append(i)

        # create the arrays
        solid = get_particle_array(name='solid', x=x, y=y)

        # remove the fluid particles from the solid
        fluid = solid.extract_particles(indices); fluid.set_name('fluid')
        solid.remove_particles(indices)

        print("Periodic cylinders :: Re = %g, nfluid = %d, nsolid=%d, dt = %g"%(
            Re, fluid.get_number_of_particles(),
            solid.get_number_of_particles(), dt))
        print("tf = %f"%tf)

        # add requisite properties to the arrays:
        self.scheme.setup_properties([fluid, solid])

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * rho0
        solid.m[:] = volume * rho0

        # initial particle density
        fluid.rho[:] = rho0
        solid.rho[:] = rho0

        # volume is set as dx^2. V is the number density form of the
        # particle volume and will be computed in the equations for the
        # fluid phase. The initial values are used for the solid phase
        fluid.V[:] = 1./volume
        solid.V[:] = 1./volume

        # particle smoothing lengths
        fluid.h[:] = hdx * dx
        solid.h[:] = hdx * dx

        # return the particle list
        return [fluid, solid]

    def create_scheme(self):
        s = TVFScheme(
            ['fluid'], ['solid'], dim=2, rho0=rho0, c0=c0, nu=nu,
            p0=p0, pb=p0, h0=dx*hdx, gx=fx
        )
        s.configure_solver(tf=tf, dt=dt, n_damp=100, pfreq=500)
        return s

    def post_process(self, info_fname):
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        t, cd = self._plot_cd_vs_t()
        res = os.path.join(self.output_dir, 'results.npz')
        np.savez(res, t=t, cd=cd)

    def _plot_cd_vs_t(self):
        from pysph.solver.utils import iter_output, load
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.sph.equation import Group
        from pysph.sph.wc.transport_velocity import (SetWallVelocity,
            MomentumEquationPressureGradient, SolidWallNoSlipBC,
            SolidWallPressureBC, VolumeSummation)

        data = load(self.output_files[0])
        solid = data['arrays']['solid']
        fluid = data['arrays']['fluid']
        x, y = solid.x.copy(), solid.y.copy()
        cx = 0.5 * L; cy = 0.5 * H
        inside = np.sqrt((x-cx)**2 + (y-cy)**2) <= a
        dest = solid.extract_particles(inside.nonzero()[0])
        # We use the same equations for this as the simulation, except that we
        # do not include the acceleration terms as these are externally
        # imposed.  The goal of these is to find the force of the fluid on the
        # cylinder, thus, gx=0.0 is used in the following.
        equations = [
            Group(
                equations=[
                    VolumeSummation(
                        dest='fluid', sources=['fluid', 'solid']
                    ),
                    VolumeSummation(
                        dest='solid', sources=['fluid', 'solid']
                    ),
                    ], real=False),

            Group(
                equations=[
                    SetWallVelocity(dest='solid', sources=['fluid']),
                    ], real=False),

            Group(
                equations=[
                    SolidWallPressureBC(dest='solid', sources=['fluid'],
                                        gx=0.0, b=1.0, rho0=rho0, p0=p0),
                    ], real=False),

            Group(
                equations=[
                    # Pressure gradient terms
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['solid'], gx=0.0, pb=pb),
                    SolidWallNoSlipBC(
                        dest='fluid', sources=['solid'], nu=nu),
                    ], real=True),
        ]

        sph_eval = SPHEvaluator(
            arrays=[dest, fluid], equations=equations, dim=2,
            kernel=QuinticSpline(dim=2)
        )

        t, cd = [], []
        for sd, fluid in iter_output(self.output_files, 'fluid'):
            fluid.remove_property('vmag2')
            t.append(sd['t'])
            sph_eval.update_particle_arrays([dest, fluid])
            sph_eval.evaluate()
            Fx = np.sum(-fluid.au*fluid.m)
            cd.append(Fx/(nu*rho0*Umax))

        t, cd = list(map(np.asarray, (t, cd)))

        # Now plot the results.
        import matplotlib
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        f = plt.figure()
        plt.plot(t, cd)
        plt.xlabel('$t$'); plt.ylabel(r'$C_D$')
        fig = os.path.join(self.output_dir, "cd_vs_t.png")
        plt.savefig(fig, dpi=300)
        plt.close()

        return t, cd

if __name__ == '__main__':
    app = PeriodicCylinders()
    app.run()
    app.post_process(app.info_filename)
