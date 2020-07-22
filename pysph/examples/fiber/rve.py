"""Example for Mini RVE.

################################################################################
Mini RVE
################################################################################

Reference
---------
N. Meyer et. al "Parameter Identification of Fiber Orientation Models Based on Direct
Fiber Simulation with Smoothed Particle Hydrodynamics",
Journal of Composites Science, 2020, 4, 77; doi:10.3390/jcs4020077
"""
import itertools
# general imports
import os
import random

import numpy as np

from pysph.base.nnps import DomainManager
from pysph.base.utils import (get_particle_array_beadchain_fiber,
                              get_particle_array_beadchain_fluid)
from pysph.solver.application import Application
from pysph.solver.utils import load, remove_irrelevant_files
from pysph.sph.scheme import BeadChainScheme

# from pysph.solver.tools import FiberIntegrator


class RVE(Application):
    """Generate a mini RVE and evaluate its fiber orientation tensor."""

    def create_scheme(self):
        """Use the BeadChainScheme for this model."""
        return BeadChainScheme(["fluid"], [], ["fibers"], dim=3)

    def add_user_options(self, group):
        """Set these parameters in the command line."""
        group.add_argument(
            "--dx",
            action="store",
            type=float,
            dest="dx",
            default=0.0001,
            help="Particle spacing",
        )
        group.add_argument(
            "--lf",
            action="store",
            type=int,
            dest="lf",
            default=5,
            help="Fiber length in multiples of dx",
        )
        group.add_argument(
            "--mu",
            action="store",
            type=float,
            dest="mu",
            default=1.0,
            help="Absolute viscosity",
        )
        group.add_argument(
            "--S",
            action="store",
            type=float,
            dest="S",
            default=100,
            help="Dimensionless fiber stiffness",
        )
        group.add_argument(
            "--G",
            action="store",
            type=float,
            dest="G",
            default=1.0,
            help="Shear rate applied to the cube",
        )
        group.add_argument(
            "--Re",
            action="store",
            type=float,
            dest="Re",
            default=0.5,
            help="Desired Reynolds number.",
        )
        group.add_argument(
            "--volfrac",
            action="store",
            type=float,
            dest="vol_frac",
            default=0.0014,
            help="Volume fraction of fibers in suspension.",
        )
        group.add_argument(
            "--rot",
            action="store",
            type=float,
            dest="rot",
            default=2.0,
            help="Number of half rotations.",
        )
        group.add_argument(
            "--C",
            action="store",
            type=float,
            dest="C",
            default=15.0,
            help="Cube size as multiples of particle spacing.",
        )
        group.add_argument(
            "--continue",
            action="store",
            type=str,
            dest="continuation",
            default=None,
            help="Set a file for continuation of run.",
        )

    def consume_user_options(self):
        """Initialize geometry, properties and time stepping."""
        # Initial spacing of particles
        self.dx = self.options.dx
        self.h0 = self.dx

        # The fiber length is the aspect ratio times fiber diameter
        self.L = self.options.lf * self.dx

        # Cube size
        self.C = self.options.C * self.dx

        # Density from Reynolds number
        self.Vmax = self.options.G * self.C / 2.0
        self.rho0 = (self.options.mu * self.options.Re) / (self.Vmax * self.C)

        # The kinematic viscosity
        self.nu = self.options.mu / self.rho0

        # Fiber aspect ratio (assuming cylindrical shape)
        R = self.dx / (np.sqrt(np.pi))
        self.d = 2.0 * R
        self.ar = self.L / self.d
        print("Aspect ratio is %f" % self.ar)

        # cross section properties
        self.A = np.pi * R ** 2.0
        self.Ip = np.pi * R ** 4.0 / 4.0

        # inertia properties
        mass = 3.0 * self.rho0 * self.dx * self.A
        self.J = (
            1.0 / 4.0 * mass * R ** 2.0 + 1.0 / 12.0 * mass * (3.0 * self.dx) ** 2.0
        )

        # stiffness from dimensionless stiffness
        self.E = (
            4.0 / np.pi * (self.options.S * self.options.mu * self.options.G * self.ar)
        )

        # The speed of sound c0 is computed as 10 times the maximum velocity.
        # This should keep the density change within 1%
        self.c0 = 10.0 * self.Vmax
        self.p0 = self.c0 ** 2 * self.rho0

        # Background pressure in Adami's transport velocity formulation
        self.pb = self.p0

        # Simulation time
        self.t = self.options.rot * np.pi * (self.ar + 1.0 / self.ar) / self.options.G
        print("Simulated time is %g s" % self.t)

        # Determine number of fibers to be generated
        vol_fiber = self.L * self.dx * self.dx
        vol = self.C ** 3
        self.n = int(round(self.options.vol_frac * vol / vol_fiber))

    def configure_scheme(self):
        """Set up scheme and solver.

        The flag 'direct' means that elastic equations and contact equations are solved
        together with all other equations. If the fiber is very stiff, one may use a
        subcycle to integrate fiber positions. Therefore, set 'direct=False' and
        uncomment the FiberIntegrator tool.
        """
        self.scheme.configure(
            rho0=self.rho0,
            c0=self.c0,
            nu=self.nu,
            p0=self.p0,
            pb=self.pb,
            h0=self.h0,
            dx=self.dx,
            A=self.A,
            Ip=self.Ip,
            J=self.J,
            E=self.E,
            d=self.d,
            direct=True,
        )
        if self.n < 1:
            self.scheme.configure(fibers=[])
        self.scheme.configure_solver(
            tf=self.t,
            # pfreq=1,
            N=self.options.rot * 100,
        )

    def create_particles(self):
        """Create or load particle arrays."""
        if self.options.continuation:
            data = load(self.options.continuation)
            fluid = data["arrays"]["fluid"]
            fibers = data["arrays"]["fibers"]
            fibers.phifrac[:] = 2.0
            fibers.phi0[:] = np.pi
            self.solver.t = data["solver_data"]["t"]
            self.solver.count = data["solver_data"]["count"]
            return [fluid, fibers]
        else:
            return self.create_suspension_particles()

    def create_suspension_particles(self):
        """Create particle arrays."""
        fdx = self.dx
        dx2 = fdx / 2

        # Computation of each particles initial volume.
        volume = fdx ** 3

        # Mass is set to get the reference density of rho0.
        mass = volume * self.rho0

        # Initial inverse volume (necessary for transport velocity equations)
        V = 1.0 / volume

        # Create grid points for particles
        _x = np.arange(dx2, self.C, fdx)
        _y = np.arange(dx2, self.C, fdx)
        _z = np.arange(dx2, self.C, fdx)
        fx, fy, fz = self.get_meshgrid(_x, _y, _z)

        # Remove particles at fiber position and create fiber particles.
        indices = []
        fibers = []
        fibx = tuple()
        fiby = tuple()
        fibz = tuple()

        positions = list(itertools.product(_x, _y, _z))
        random.shuffle(positions)
        N = 0
        while N < self.n:
            xx, yy, zz = positions.pop()
            idx_list = []
            for i in range(len(fx)):
                # periodic extending above
                if xx + self.L / 2 > self.C:
                    if (
                        (fx[i] < (xx + self.L / 2 - self.C) or fx[i] > xx - self.L / 2)
                        and fy[i] < yy + self.dx / 2
                        and fy[i] > yy - self.dx / 2
                        and fz[i] < zz + self.dx / 2
                        and fz[i] > zz - self.dx / 2
                    ):
                        idx_list.append(i)
                # periodic extending below
                elif xx - self.L / 2 < 0:
                    if (
                        (fx[i] < xx + self.L / 2 or fx[i] > (xx - self.L / 2 + self.C))
                        and fy[i] < yy + self.dx / 2
                        and fy[i] > yy - self.dx / 2
                        and fz[i] < zz + self.dx / 2
                        and fz[i] > zz - self.dx / 2
                    ):
                        idx_list.append(i)
                # standard case
                else:
                    if (
                        fx[i] < xx + self.L / 2
                        and fx[i] > xx - self.L / 2
                        and fy[i] < yy + self.dx / 2
                        and fy[i] > yy - self.dx / 2
                        and fz[i] < zz + self.dx / 2
                        and fz[i] > zz - self.dx / 2
                    ):
                        idx_list.append(i)

            idx_set = set(idx_list)
            if len(idx_set.intersection(set(indices))) == 0:
                N = N + 1
                indices = indices + idx_list

                # Generate fiber particles
                if self.options.lf % 2 == 1:
                    _fibx = np.linspace(
                        xx - self.options.lf // 2 * self.dx,
                        xx + self.options.lf // 2 * self.dx,
                        self.options.lf,
                    )
                else:
                    _fibx = np.arange(
                        xx - self.L / 2, xx + self.L / 2 - self.dx / 4, self.dx
                    )
                _fiby = np.array([yy])
                _fibz = np.array([zz])
                _fibx, _fiby, _fibz = self.get_meshgrid(_fibx, _fiby, _fibz)
                fibx = fibx + (_fibx,)
                fiby = fiby + (_fiby,)
                fibz = fibz + (_fibz,)

        print("Created %d fibers." % N)

        # Finally create all particle arrays. Note that fluid particles are
        # removed in the area, where the fiber is placed.
        fluid = get_particle_array_beadchain_fluid(
            name="fluid", x=fx, y=fy, z=fz, m=mass, rho=self.rho0, h=self.h0, V=V
        )
        fluid.remove_particles(indices)

        if self.n > 0:
            fibers = get_particle_array_beadchain_fiber(
                name="fibers",
                x=np.concatenate(fibx),
                y=np.concatenate(fiby),
                z=np.concatenate(fibz),
                m=mass,
                rho=self.rho0,
                h=self.h0,
                lprev=self.dx,
                lnext=self.dx,
                phi0=np.pi,
                phifrac=2.0,
                fidx=range(self.options.lf * self.n),
                V=V,
            )
            # 'Break' fibers in segments
            endpoints = [i * self.options.lf - 1 for i in range(1, self.n)]
            fibers.fractag[endpoints] = 1

        # Setting the initial velocities for a shear flow.
        fluid.v[:] = self.options.G * (fluid.x[:] - self.C / 2)

        fibers.v[:] = self.options.G * (fibers.x[:] - self.C / 2)
        return [fluid, fibers]

    def create_domain(self):
        """Create periodic boundary conditions in all directions.

        Additionally, gamma values are set to enforce Lee-Edwards BCs.
        """
        return DomainManager(
            xmin=0,
            xmax=self.C,
            periodic_in_x=True,
            ymin=0,
            ymax=self.C,
            periodic_in_y=True,
            zmin=0,
            zmax=self.C,
            periodic_in_z=True,
            gamma_yx=self.options.G,
            n_layers=1,
            dt=self.solver.dt,
            calls_per_step=2,
        )

    # def create_tools(self):
    #     """Set up fiber integrator."""
    #     return [FiberIntegrator(self.particles, self.scheme, self.domain,
    #                             D=0.002*self.options.lf)]

    def get_meshgrid(self, xx, yy, zz):
        """Create meshgrids."""
        x, y, z = np.meshgrid(xx, yy, zz)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        return [x, y, z]

    def post_process(self, info_fname):
        """Save fiber orientation tensor to csv file."""
        if len(self.output_files) == 0:
            return

        from pysph.tools.pprocess import get_ke_history
        from matplotlib import pyplot as plt

        t, ke = get_ke_history(self.output_files, "fluid")
        plt.clf()
        plt.plot(t, ke)
        plt.xlabel("t")
        plt.ylabel("Kinetic energy")
        fig = os.path.join(self.output_dir, "ke_history.png")
        plt.savefig(fig, dpi=300)

        # empty list for time
        t = []

        # empty lists for fiber orientation tensors
        A = []

        # iteration over all output files
        output_files = remove_irrelevant_files(self.output_files)
        for fname in output_files:
            data = load(fname)

            # extracting time
            t.append(data["solver_data"]["t"])

            if self.n > 0:
                # extrating all arrays.
                directions = []
                fiber = data["arrays"]["fibers"]
                startpoints = [i * (self.options.lf - 1) for i in range(0, self.n)]
                endpoints = [
                    i * (self.options.lf - 1) - 1 for i in range(1, self.n + 1)
                ]
                for start, end in zip(startpoints, endpoints):
                    px = np.mean(fiber.rxnext[start:end])
                    py = np.mean(fiber.rynext[start:end])
                    pz = np.mean(fiber.rznext[start:end])

                    n = np.array([px, py, pz])
                    norm = np.linalg.norm(n)
                    if norm == 0:
                        p = np.array([1, 0, 0])
                    else:
                        p = n / norm
                    directions.append(p)

                N = len(directions)
                a = np.zeros([3, 3])
                for p in directions:
                    for i in range(3):
                        for j in range(3):
                            a[i, j] += 1.0 / N * (p[i] * p[j])
                A.append(a.ravel())

        csv_file = os.path.join(self.output_dir, "A.csv")
        data = np.hstack((np.matrix(t).T, np.vstack(A)))
        np.savetxt(csv_file, data, delimiter=",")

        # plot results
        data = np.loadtxt(csv_file, delimiter=',')
        t = data[:, 0]
        plt.figure(figsize=(12, 3))
        for j, i in enumerate([0, 4, 8, 1]):
            plt.subplot("14"+str(j+1))
            p = plt.plot(self.options.G*t, data[:, i+1])
            plt.xlabel("Strains")
            plt.ylabel("Component %d" % j)
            if i % 2 == 0:
                plt.ylim([0, 1])
            else:
                plt.ylim([-1, 1])
        plt.tight_layout()
        plt.savefig(csv_file.replace('.csv', '.pdf'))
        try:
            from matplotlib2tikz import save as tikz_save
            tikz_save(csv_file.replace('.csv', '.tex'))
        except ImportError:
            print("Did not write tikz figure.")


if __name__ == "__main__":
    app = RVE()
    app.run()
    app.post_process(app.info_filename)
