"""Evolution of a circular patch of incompressible fluid. (60 seconds)

See J. J. Monaghan "Simulating Free Surface Flows with SPH", JCP, 1994, 100, pp
399 - 406

An initially circular patch of fluid is subjected to a velocity profile that
causes it to deform into an ellipse. Incompressibility causes the initially
circular patch to deform into an ellipse such that the area is conserved. An
analytical solution for the locus of the patch is available (exact_solution)

This is a standard test for the formulations for the incompressible SPH
equations.

"""
from __future__ import print_function

import os
from numpy import array, ones_like, mgrid, sqrt
import numpy as np

# PySPH base and carray imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.scheme import WCSPHScheme


def _derivative(x, t):
    A, a = x
    Anew = A*A*(a**4 -1)/(a**4 + 1)
    anew = -a*A
    return array((Anew, anew))

def _scipy_integrate(y0, tf, dt):
    from scipy.integrate import odeint
    result = odeint(_derivative, y0, [0.0, tf])
    return result[-1]

def _numpy_integrate(y0, tf, dt):
    t = 0.0
    y = y0
    while t <= tf:
        t += dt
        y += dt*_derivative(y, t)
    return y

def exact_solution(tf=0.0075, dt=1e-6, n=101):
    """Exact solution for the locus of the circular patch.

    n is the number of points to find the result at.

    Returns the semi-minor axis, A, pressure, x, y.

    Where x, y are the points corresponding to the ellipse.
    """
    import numpy

    y0 = array([100.0, 1.0])

    try:
        from scipy.integrate import odeint
    except ImportError:
        Anew, anew = _numpy_integrate(y0, tf, dt)
    else:
        Anew, anew = _scipy_integrate(y0, tf, dt)

    dadt = _derivative([Anew, anew], tf)[0]
    po = 0.5*-anew**2 * (dadt - Anew**2)

    theta = numpy.linspace(0,2*numpy.pi, n)

    return anew, Anew, po, anew*numpy.cos(theta), 1/anew*numpy.sin(theta)


class EllipticalDrop(Application):
    def initialize(self):
        self.co = 1400.0
        self.ro = 1.0
        self.hdx = 1.3
        self.dx = 0.025
        self.alpha = 0.1

    def create_scheme(self):
        s = WCSPHScheme(
            ['fluid'], [], dim=2, rho0=self.ro, c0=self.co,
            h0=self.dx*self.hdx, hdx=self.hdx, gamma=7.0, alpha=0.1, beta=0.0
        )
        kernel = Gaussian(dim=2)
        dt = 5e-6; tf = 0.0076
        s.configure_solver(
            kernel=kernel, integrator_cls=EPECIntegrator, dt=dt, tf=tf,
            adaptive_timestep=True, cfl=0.3, n_damp=50,
            output_at_times=[0.0008, 0.0038]
        )
        return s

    def create_particles(self):
        """Create the circular patch of fluid."""
        dx = self.dx
        hdx = self.hdx
        co = self.co
        ro = self.ro
        name = 'fluid'
        x, y = mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        m = ones_like(x)*dx*dx
        h = ones_like(x)*hdx*dx
        rho = ones_like(x) * ro

        p = ones_like(x) * 1./7.0 * co**2
        cs = ones_like(x) * co

        u = -100*x
        v = 100*y

        # remove particles outside the circle
        indices = []
        for i in range(len(x)):
            if sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
                indices.append(i)

        pa = get_particle_array(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v,
                                cs=cs, name=name)
        pa.remove_particles(indices)

        print("Elliptical drop :: %d particles"%(pa.get_number_of_particles()))
        mu = ro*self.alpha*hdx*dx*co/8.0
        print("Effective viscosity: rho*alpha*h*c/8 = %s"%mu)

        self.scheme.setup_properties([pa])
        return [pa]

    def _make_final_plot(self):
        from matplotlib import pyplot as plt
        last_output = self.output_files[-1]
        from pysph.solver.utils import load
        data = load(last_output)
        pa = data['arrays']['fluid']
        tf = data['solver_data']['t']
        a, A, po, xe, ye = exact_solution(tf)
        print("At tf=%s"%tf)
        print("Semi-major axis length (exact, computed) = %s, %s"
                %(1.0/a, max(pa.y)))
        plt.plot(xe, ye)
        plt.scatter(pa.x, pa.y, marker='.')
        plt.ylim(-2, 2)
        plt.xlim(plt.ylim())
        plt.title("Particles at %s secs"%tf)
        plt.xlabel('x'); plt.ylabel('y')
        fig = os.path.join(self.output_dir, "comparison.png")
        plt.savefig(fig, dpi=300)
        print("Figure written to %s."%fig)

    def _compute_results(self):
        from pysph.solver.utils import iter_output
        from collections import defaultdict
        data = defaultdict(list)
        for sd, array in iter_output(self.output_files, 'fluid'):
            _t = sd['t']
            data['t'].append(_t)
            m, u, v, x, y = array.get('m', 'u', 'v', 'x', 'y')
            vmag2 = u**2 + v**2
            data['ke'].append(0.5*np.sum(m*vmag2))
            data['xmax'].append(x.max())
            data['ymax'].append(y.max())
            a, A, po, _xe, _ye = exact_solution(_t, n=0)
            data['minor'].append(a)
            data['major'].append(1.0/a)

        for key in data:
            data[key] = np.asarray(data[key])
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, **data)

    def post_process(self, info_file_or_dir):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        if self.rank > 0:
            return
        info = self.read_info(info_file_or_dir)
        if len(self.output_files) == 0:
            return
        self._compute_results()
        self._make_final_plot()


if __name__ == '__main__':
    app = EllipticalDrop()
    app.run()
    app.post_process(app.info_filename)
