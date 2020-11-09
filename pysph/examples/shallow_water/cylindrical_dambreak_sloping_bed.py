"""Cylindrical dam break over a sloping parabolic dry bed. (24 mins)

The case is described in "A corrected smooth particle hydrodynamics formulation
of the shallow-water equations", Miguel Rodriguez-Paz and Javier Bonet,
Computers & Structures, Vol 83, pp 1396-1410 (2005).
DOI:10.1016/j.compstruc.2004.11.025

"""
# Numpy
from numpy import (ones_like, zeros, pi, arange, concatenate, sin, cos)

# PySPH base
from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array_swe as gpa_swe

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# PySPH equations
from pysph.sph.equation import Group
from pysph.sph.swe.basic import (
        InitialGuessDensity, SummationDensity, DensityResidual,
        DensityNewtonRaphsonIteration, CheckConvergence, UpdateSmoothingLength,
        SWEOS, SWEIntegrator, SWEStep, CorrectionFactorVariableSmoothingLength,
        ParticleAcceleration
        )

# PySPH Evaluator
from pysph.tools.sph_evaluator import SPHEvaluator


# Constants
rho_w = 1000.0
g = 9.81
dim = 2


class ParticleAccelerationAnalytBottom(ParticleAcceleration):
    def post_loop(self, d_y, d_idx, d_u, d_v, d_tu, d_tv, d_au, d_av, d_Sfx,
                  d_Sfy):
        # Gradient of fluid bottom
        bx = -0.839
        by = 0.909 * d_y[d_idx]

        # Curvature of fluid bottom
        bxx = 0
        bxy = 0
        byy = 0.909

        vikivi = (d_u[d_idx]*d_u[d_idx]*bxx
                  + 2*d_u[d_idx]*d_v[d_idx]*bxy
                  + d_v[d_idx]*d_v[d_idx]*byy)
        tidotgradbi = d_tu[d_idx]*bx + d_tv[d_idx]*by
        gradbidotgradbi = bx**2 + by**2
        temp3 = self.g + vikivi - tidotgradbi
        temp4 = 1 + gradbidotgradbi
        if not self.v_only:
            d_au[d_idx] = -(temp3/temp4)*bx - d_tu[d_idx] - d_Sfx[d_idx]
        if not self.u_only:
            d_av[d_idx] = -(temp3/temp4)*by - d_tv[d_idx] - d_Sfy[d_idx]


class CylindricalDamBreakSlopingBed(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=2.0,
            help="h/dx value used in SPH to change the smoothness")
        group.add_argument(
            "--dw0", action="store", type=float, dest="dw0", default=0.25,
            help="Initial depth of the fluid column (m)")
        group.add_argument(
            "--r", action="store", type=float, dest="r", default=0.25,
            help="Initial radius of the fluid column (m)")
        group.add_argument(
            "--n", action="store", type=float, dest="n", default=50,
            help="Number of concentric fluid particle circles (Determines\
            spacing btw particles, dr = r/n)")
        group.add_argument(
            "--R", action="store", type=float, dest="R", default=1.1,
            help="Radius of curvature of bed surface (m)")
        group.add_argument(
            "--theta", action="store", type=float, dest="theta", default=40,
            help="Bed slope measured along clockwise direction from horizontal\
            (degrees)")
        group.add_argument(
            "--xcen", action="store", type=float, dest="xcen", default=0.25,
            help="x-coordinate of the center of the base of the fluid column\
            (m)")
        group.add_argument(
            "--ycen", action="store", type=float, dest="ycen", default=0.5,
            help="y-coordinate of the center of the base of the fluid column\
            (m)")

    def consume_user_options(self):
        self.hdx = self.options.hdx
        self.dw0 = self.options.dw0
        self.r = self.options.r
        self.n = self.options.n
        self.R = self.options.R
        self.theta = self.options.theta
        self.xcen = self.options.xcen
        self.ycen = self.options.ycen

    def create_particles(self):
        n = self.n
        r = self.r
        dr = r / n

        d = self.dw0
        hdx = self.hdx

        x = zeros(0)
        y = zeros(0)

        # Create circular patch of fluid in a radial grid
        rad = 0.0
        for j in range(1, n+1):
                npnts = 4 * j
                dtheta = (2*pi) / npnts

                theta = arange(0, 2*pi-1e-10, dtheta)
                rad = rad + dr

                _x = rad * cos(theta)
                _y = rad * sin(theta)

                x = concatenate((x, _x))
                y = concatenate((y, _y))

        arr_ones = ones_like(x)

        m = arr_ones * (1.56*dr*dr) * rho_w * d

        rho = arr_ones * rho_w * d
        rho0 = arr_ones * rho_w * d

        h = arr_ones * hdx * dr
        h0 = arr_ones * hdx * dr

        # Analytially set fluid bottom height and gradients
        R = self.R
        theta = self.theta
        b = y**2/R + (theta*(pi/180.))*x
        bx = arr_ones * -0.839
        by = 0.909 * y
        byy = 0.909

        # Eccentricity in fluid column from mid section
        x += self.xcen
        y += self.ycen

        fluid = gpa_swe(x=x, y=y, m=m, rho=rho, rho0=rho0, h=h, h0=h0,
                        b=b, bx=bx, by=by, byy=byy, name='fluid')

        compute_initial_props([fluid])
        return [fluid]

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        integrator = SWEIntegrator(fluid=SWEStep())
        tf = 1.0
        solver = Solver(
            kernel=kernel,
            dim=2,
            integrator=integrator,
            cfl=0.2,
            adaptive_timestep=True,
            output_at_times=(0.2, 0.5, 1.0),
            tf=tf
            )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    InitialGuessDensity(dim=dim, dest='fluid',
                                        sources=['fluid']),
                    UpdateSmoothingLength(dim=dim, dest='fluid')
                ], update_nnps=True
            ),

            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(
                            dest='fluid', sources=['fluid']),
                    SummationDensity(dest='fluid', sources=['fluid']),
                    DensityResidual('fluid')
                ]
            ),

            Group(
                equations=[
                    Group(
                        equations=[
                            DensityNewtonRaphsonIteration(dim=dim, 
                                                          dest='fluid'),
                            UpdateSmoothingLength(dim=dim, dest='fluid')
                        ], update_nnps=True
                    ),

                    Group(
                        equations=[
                            CorrectionFactorVariableSmoothingLength(
                                    dest='fluid', sources=['fluid']),
                            SummationDensity(dest='fluid', sources=['fluid']),
                            DensityResidual(dest='fluid'),
                            CheckConvergence(dest='fluid')
                        ],
                    )
                ], iterate=True, max_iterations=100
            ),

            Group(
                equations=[
                    CorrectionFactorVariableSmoothingLength(dest='fluid',
                                                            sources=['fluid']),
                    SWEOS(dest='fluid')
                ]
            ),

            Group(
                equations=[
                    ParticleAccelerationAnalytBottom(dim=dim, dest='fluid', 
                                                     sources=['fluid'])
                ]
            )
        ]
        return equations

def compute_initial_props(particles):
    one_time_equations = [
        Group(
            equations=[
                CorrectionFactorVariableSmoothingLength(dest='fluid', 
                                                        sources=['fluid']),
                SWEOS(dest='fluid')
            ]
        ),
    ]
    kernel = CubicSpline(dim=2)
    sph_eval = SPHEvaluator(particles, one_time_equations, dim=2,
                            kernel=kernel)
    sph_eval.evaluate()


if __name__ == '__main__':
    app = CylindricalDamBreakSlopingBed()
    app.run()
