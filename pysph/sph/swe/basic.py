"""
Basic Equations for solving shallow water problems
#####################
"""

from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.integrator import Integrator
from compyle.api import declare
from pysph.sph.wc.linalg import gj_solve, augmented_matrix

from numpy import sqrt, cos, sin, zeros, pi, exp
import numpy as np
import numpy
M_PI = pi


class CheckForParticlesToSplit(Equation):
    r"""Particles are tagged for splitting if the following condition is
    satisfied:

    .. math::
        (A_i > A_max) and (h_i < h_max) and (x_min < x_i < x_max) and (y_min <
        y_i < y_max)

    References
    ----------
    .. [Vacondio2010] R. Vacondio, B.D. Rodgers and P.K. Stansby, "Accurate
    particle splitting for smoothed particle hydrodynamics in shallow water
    with shock capturing", Int. J. Numer. Meth. Fluids, 69 (2012), pp.
    1377-1410

    """
    def __init__(self, dest, h_max=1e9, A_max=1e9, x_min=-1e9, x_max=1e9,
                 y_min=-1e9, y_max=1e9):
        r"""
        Parameters
        ----------
        h_max : float
            maximum smoothing length beyond which splitting is deactivated
        A_max : float
            maximum area beyond which splitting is activated
        x_min : float
            minimum distance along x-direction beyond which splitting is
            activated
        x_max : float
            maximum distance along x-direction beyond which splitting is
            deactivated
        y_min : float
            minimum distance along y-direction beyond which splitting is
            activated
        y_max : float
            maximum distance along y-direction beyond which splitting is
            deactivated

        """
        self.A_max = A_max
        self.h_max = h_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        super(CheckForParticlesToSplit, self).__init__(dest, None)

    def initialize(self, d_idx, d_A, d_h, d_x, d_y, d_pa_to_split):
        if (d_A[d_idx] > self.A_max and d_h[d_idx] < self.h_max
           and (self.x_min < d_x[d_idx] < self.x_max)
           and (self.y_min < d_y[d_idx] < self.y_max)):
            d_pa_to_split[d_idx] = 1
        else:
            d_pa_to_split[d_idx] = 0


class ParticleSplit(object):
    r"""**Hexagonal particle splitting algorithm**

    References
    ----------
    .. [Vacondio2010] R. Vacondio, B.D. Rodgers and P.K. Stansby, "Accurate
    particle splitting for smoothed particle hydrodynamics in shallow water
    with shock capturing", Int. J. Numer. Meth. Fluids, 69 (2012), pp.
    1377-1410

    """
    def __init__(self, pa_arr):
        r"""
        Parameters
        ----------
        pa_arr : pysph.base.particle_array.ParticleArray
            particle array of fluid

        """
        self.pa_arr = pa_arr

        # Ratio of mass of daughter particle located at center of hexagon to
        # that of its parents mass
        self.center_pa_mass_frac = 0.178705766141917

        # Ratio of mass of daughter particle located at vertex of hexagon to
        # that of its parents mass
        self.vertex_pa_mass_frac = 0.136882287617319

        # Ratio of smoothing length of daughter particle to that of its parents
        # smoothing length
        self.pa_h_ratio = 0.9

        # Ratio of distance between center daughter particle and vertex
        # daughter particle to that of its parents smoothing length
        self.center_and_vertex_pa_separation_frac = 0.4

        # Get index of the parent particles to split
        self.idx_pa_to_split = self._get_idx_of_particles_to_split()

        # Number of daughter particles located at the vertices of hexagon after
        # splitting
        self.num_vertex_pa_after_single_split = 6

    def do_particle_split(self, solver=None):
        if not self.idx_pa_to_split.size:
            # If no particles to split then return
            return
        else:
            # Properties of parent particles to split
            h_parent = self.pa_arr.h[self.idx_pa_to_split]
            h0_parent = self.pa_arr.h0[self.idx_pa_to_split]
            m_parent = self.pa_arr.m[self.idx_pa_to_split]
            x_parent = self.pa_arr.x[self.idx_pa_to_split]
            y_parent = self.pa_arr.y[self.idx_pa_to_split]
            u_parent = self.pa_arr.u[self.idx_pa_to_split]
            v_parent = self.pa_arr.v[self.idx_pa_to_split]
            u_prev_step_parent = self.pa_arr.u_prev_step[self.idx_pa_to_split]
            v_prev_step_parent = self.pa_arr.v_prev_step[self.idx_pa_to_split]
            rho_parent = self.pa_arr.rho[self.idx_pa_to_split]
            rho0_parent = self.pa_arr.rho0[self.idx_pa_to_split]
            alpha_parent = self.pa_arr.alpha[self.idx_pa_to_split]

            # Vertex daughter particle properties update
            n = self.num_vertex_pa_after_single_split
            h_vertex_pa = self.pa_h_ratio * np.repeat(h_parent, n)
            h0_vertex_pa = self.pa_h_ratio * np.repeat(h0_parent, n)
            u_prev_step_vertex_pa = np.repeat(u_prev_step_parent, n)
            v_prev_step_vertex_pa = np.repeat(v_prev_step_parent, n)
            m_vertex_pa = self.vertex_pa_mass_frac * np.repeat(m_parent, n)
            vertex_pa_pos = self._get_vertex_pa_positions(h_parent, u_parent,
                                                          v_parent)
            x_vertex_pa = vertex_pa_pos[0] + np.repeat(x_parent, n)
            y_vertex_pa = vertex_pa_pos[1] + np.repeat(y_parent, n)

            rho0_vertex_pa = np.repeat(rho0_parent, n)
            rho_vertex_pa = np.repeat(rho_parent, n)
            alpha_vertex_pa = np.repeat(alpha_parent, n)
            parent_idx_vertex_pa = np.repeat(self.idx_pa_to_split, n)

            # Note:
            # The center daughter particle properties are set at index of
            # parent. The properties of parent needed for further calculations
            # are not changed for now

            # Center daughter particle properties update
            for idx in self.idx_pa_to_split:
                self.pa_arr.m[idx] *= self.center_pa_mass_frac
                self.pa_arr.h[idx] *= self.pa_h_ratio
                self.pa_arr.h0[idx] *= self.pa_h_ratio
                self.pa_arr.parent_idx[idx] = int(idx)

            # Update particle array to include vertex daughter particles
            self._add_vertex_pa_prop(
                h0_vertex_pa, h_vertex_pa, m_vertex_pa, x_vertex_pa,
                y_vertex_pa, rho0_vertex_pa, rho_vertex_pa,
                u_prev_step_vertex_pa, v_prev_step_vertex_pa, alpha_vertex_pa,
                parent_idx_vertex_pa)

    def _get_idx_of_particles_to_split(self):
        idx_pa_to_split = []
        for idx, val in enumerate(self.pa_arr.pa_to_split):
            if val:
                idx_pa_to_split.append(idx)
        return np.array(idx_pa_to_split)

    def _get_vertex_pa_positions(self, h_parent, u_parent, v_parent):
        # Number of particles to split
        num_of_pa_to_split = len(self.idx_pa_to_split)

        n = self.num_vertex_pa_after_single_split
        theta_vertex_pa = zeros(n)
        r = self.center_and_vertex_pa_separation_frac

        for i, theta in enumerate(range(0, 360, 60)):
            theta_vertex_pa[i] = (pi/180)*theta

        # Angle of velocity vector with horizontal
        angle_vel = np.where(
            (np.abs(u_parent) > 1e-3) | (np.abs(v_parent) > 1e-3),
            np.arctan2(v_parent, u_parent), 0
            )

        # Rotates the hexagon such that its horizontal axis aligns with the
        # direction of velocity vector
        angle_actual = (np.tile(theta_vertex_pa, num_of_pa_to_split)
                        + np.repeat(angle_vel, n))

        x = r * cos(angle_actual) * np.repeat(h_parent, n)
        y = r * sin(angle_actual) * np.repeat(h_parent, n)
        return x.copy(), y.copy()

    def _add_vertex_pa_prop(self, h0_vertex_pa, h_vertex_pa, m_vertex_pa,
                            x_vertex_pa, y_vertex_pa, rho0_vertex_pa,
                            rho_vertex_pa, u_prev_step_vertex_pa,
                            v_prev_step_vertex_pa, alpha_vertex_pa,
                            parent_idx_vertex_pa):
        vertex_pa_props = {
            'm': m_vertex_pa,
            'h': h_vertex_pa,
            'h0': h0_vertex_pa,
            'x': x_vertex_pa,
            'y': y_vertex_pa,
            'u_prev_step': u_prev_step_vertex_pa,
            'v_prev_step': v_prev_step_vertex_pa,
            'rho0': rho0_vertex_pa,
            'rho': rho_vertex_pa,
            'alpha': alpha_vertex_pa,
            'parent_idx': parent_idx_vertex_pa.astype(int)
            }

        # Add vertex daughter particles to particle array
        self.pa_arr.add_particles(**vertex_pa_props)


class DaughterVelocityEval(Equation):
    r"""**Evaluation of the daughter particle velocity after splitting
    procedure**

    .. math::

        \boldsymbol{v_k} = c_v\frac{d_N}{d_k}\boldsymbol{v_N}

    where,

    .. math::

        c_v = \dfrac{A_N}{\sum_{k=1}^{M}A_k}

    References
    ----------
    .. [Vacondio2010] R. Vacondio, B.D. Rodgers and P.K. Stansby, "Accurate
    particle splitting for smoothed particle hydrodynamics in shallow water
    with shock capturing", Int. J. Numer. Meth. Fluids, 69 (2012), pp.
    1377-1410

    """
    def __init__(self, dest, sources, rhow=1000.0):
        r""""
        Parameters
        ----------
        rhow : float
            constant 3-D density of water (kg/m3)

        Notes
        -----
        This equation should be called before the equation SWEOS, as the parent
        particle area is required for calculating velocities. On calling the
        SWEOS equation, the parent properties are changed to the center
        daughter particle properties.

        """
        self.rhow = rhow
        super(DaughterVelocityEval, self).__init__(dest, sources)

    def initialize(self, d_sum_Ak, d_idx, d_m, d_rho, d_u, d_v, d_uh,
                   d_vh, d_u_parent, d_v_parent, d_uh_parent, d_vh_parent,
                   d_parent_idx):
        # Stores sum of areas of daughter particles
        d_sum_Ak[d_idx] = 0.0
        d_u_parent[d_idx] = d_u[d_parent_idx[d_idx]]
        d_uh_parent[d_idx] = d_uh[d_parent_idx[d_idx]]
        d_v_parent[d_idx] = d_v[d_parent_idx[d_idx]]
        d_vh_parent[d_idx] = d_vh[d_parent_idx[d_idx]]

    def loop_all(self, d_sum_Ak, d_pa_to_split, d_parent_idx, d_idx, s_m,
                 s_rho, s_parent_idx, NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('long')
        if d_pa_to_split[d_idx]:
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                if s_parent_idx[s_idx] == d_parent_idx[d_idx]:
                    # Sums the area of daughter particles who have same parent
                    # idx
                    d_sum_Ak[d_idx] += s_m[s_idx] / s_rho[s_idx]

    def post_loop(self, d_idx, d_parent_idx, d_A, d_sum_Ak, d_dw, d_rho, d_u,
                  d_uh, d_vh, d_v, d_u_parent, d_v_parent, d_uh_parent,
                  d_vh_parent, t):
        # True only for daughter particles
        if d_parent_idx[d_idx]:

            # Ratio of parent area (before split) to sum of areas of its
            # daughters (after split)
            cv = d_A[d_parent_idx[d_idx]] / d_sum_Ak[d_parent_idx[d_idx]]

            # The denominator (d_rho[d_idx]/self.rhow) represents the water
            # depth of daughter particle. d_dw[d_idx] cannot be used as
            # equation of state is called after this equation (Refer Notes in
            # the constructor)
            dw_ratio = d_dw[d_parent_idx[d_idx]] / (d_rho[d_idx]/self.rhow)

            d_u[d_idx] = cv * dw_ratio * d_u_parent[d_idx]
            d_uh[d_idx] = cv * dw_ratio * d_uh_parent[d_idx]
            d_v[d_idx] = cv * dw_ratio * d_v_parent[d_idx]
            d_vh[d_idx] = cv * dw_ratio * d_vh_parent[d_idx]
            d_parent_idx[d_idx] = 0


class FindMergeable(Equation):
    r"""**Particle merging algorithm**

    Particles are tagged for merging if the following condition is
    satisfied:

    .. math::

        (A_i < A_min) and (x_min < x_i < x_max) and (y_min < y_i < y_max)

    References
    ----------
    .. [Vacondio2013] R. Vacondio et al., "Shallow water SPH for flooding with
    dynamic particle coalescing and splitting", Advances in Water Resources,
    58 (2013), pp. 10-23

    """
    def __init__(self, dest, sources, A_min, x_min=-1e9, x_max=1e9, y_min=-1e9,
                 y_max=1e9):
        r"""
        Parameters
        ----------
        A_min : float
            minimum area below which merging is activated
        x_min : float
            minimum distance along x-direction beyond which merging is
            activated
        x_max : float
            maximum distance along x-direction beyond which merging is
            deactivated
        y_min : float
            minimum distance along y-direction beyond which merging is
            activated
        y_max : float
            maximum distance along y-direction beyond which merging is
            deactivated

        Notes
        -----
        The merging algorithm merges two particles 'a' and 'b' if the following
        conditions are satisfied:

        #. Both particles have area less than A_min
        #. Both particles lies within :math:`x_min < x_i < x_max` and
        :math:`y_min < y_i < y_max`
        #. if 'a' is the closest neighbor of 'b' and vice versa

        The merging algorithm is run every timestep

        """
        self.A_min = A_min
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        super(FindMergeable, self).__init__(dest, sources)

    def loop_all(self, d_idx, d_merge, d_closest_idx, d_x, d_y, d_h, d_A,
                 d_is_merged_pa, s_x, s_y, s_A, NBRS, N_NBRS):
        # Finds the closest neighbor of a particle and stores the index of that
        # neighbor in d_closest_idx[d_idx]
        i, closest = declare('int', 2)
        s_idx = declare('unsigned int')
        d_merge[d_idx] = 0
        d_is_merged_pa[d_idx] = 0
        xi = d_x[d_idx]
        yi = d_y[d_idx]
        rmin = d_h[d_idx] * 10.0
        closest = -1
        if (d_A[d_idx] < self.A_min and ((self.x_min < d_x[d_idx] < self.x_max)
           and (self.y_min < d_y[d_idx] < self.y_max))):
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                if s_idx == d_idx:
                    continue
                xij = xi - s_x[s_idx]
                yij = yi - s_y[s_idx]
                rij = sqrt(xij*xij + yij*yij)
                if rij < rmin:
                    closest = s_idx
                    rmin = rij
        d_closest_idx[d_idx] = closest

    def post_loop(self, d_idx, d_m, d_u, d_v, d_h, d_uh, d_vh, d_closest_idx,
                  d_is_merged_pa, d_merge, d_x, d_y, SPH_KERNEL):
        idx = declare('int')
        xma = declare('matrix(3)')
        xmb = declare('matrix(3)')
        idx = d_closest_idx[d_idx]
        if idx > -1:
            # If particle 'a' is closest neighbor of 'b' and vice versa
            if d_idx == d_closest_idx[idx]:
                if d_idx < idx:
                    # The newly merged particle properties are set at index of
                    # particle 'a'
                    m_merged = d_m[d_idx] + d_m[idx]
                    x_merged = ((d_m[d_idx]*d_x[d_idx] + d_m[idx]*d_x[idx])
                                / m_merged)
                    y_merged = ((d_m[d_idx]*d_y[d_idx] + d_m[idx]*d_y[idx])
                                / m_merged)
                    xma[0] = x_merged - d_x[d_idx]
                    xma[1] = y_merged - d_y[d_idx]
                    xmb[0] = x_merged - d_x[idx]
                    xmb[1] = y_merged - d_y[idx]
                    rma = sqrt(xma[0]*xma[0] + xma[1]*xma[1])
                    rmb = sqrt(xmb[0]*xmb[0] + xmb[1]*xmb[1])
                    d_u[d_idx] = ((d_m[d_idx]*d_u[d_idx] + d_m[idx]*d_u[idx])
                                  / m_merged)
                    d_uh[d_idx] = (d_m[d_idx]*d_uh[d_idx]
                                   + d_m[idx]*d_uh[idx]) / m_merged
                    d_v[d_idx] = ((d_m[d_idx]*d_v[d_idx] + d_m[idx]*d_v[idx])
                                  / m_merged)
                    d_vh[d_idx] = (d_m[d_idx]*d_vh[d_idx]
                                   + d_m[idx]*d_vh[idx]) / m_merged
                    const1 = d_m[d_idx] * SPH_KERNEL.kernel(xma, rma,
                                                            d_h[d_idx])
                    const2 = d_m[idx] * SPH_KERNEL.kernel(xmb, rmb, d_h[idx])
                    d_h[d_idx] = sqrt((7*M_PI/10.) * (m_merged/(const1+const2)))
                    d_m[d_idx] = m_merged
                    # Tags the newly formed particle after merging
                    d_is_merged_pa[d_idx] = 1
                else:
                    # Tags particle 'b' for removal after merging
                    d_merge[d_idx] = 1

    def reduce(self, dst, t, dt):
        # The indices of particle 'b' are removed from particle array after
        # merging is done
        indices = declare('object')
        indices = numpy.where(dst.merge > 0)[0]
        if len(indices) > 0:
            dst.remove_particles(indices)


class InitialDensityEvalAfterMerge(Equation):
    r"""**Initial density of the newly formed particle after merging**

    .. math ::

        \rho_M = \sum_{j}^{}m_jW_{M,j}

    References
    ----------
    .. [Vacondio2013] R. Vacondio et al., "Shallow water SPH for flooding with
    dynamic particle coalescing and splitting", Advances in Water Resources,
    58 (2013), pp. 10-23

    """
    def loop_all(self, d_rho, d_idx, d_is_merged_pa, d_x, d_y, s_h, s_m, s_x,
                 d_merge, d_closest_idx, s_y, SPH_KERNEL, NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('long')
        xij = declare('matrix(3)')
        # Evaluates the initial density of the newly formed particle after
        # merging
        if d_is_merged_pa[d_idx] == 1:
            d_rho[d_idx] = 0.0
            rij = 0.0
            rho_sum = 0.0
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                xij[0] = d_x[d_idx] - s_x[s_idx]
                xij[1] = d_y[d_idx] - s_y[s_idx]
                rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1])
                rho_sum += s_m[s_idx] * SPH_KERNEL.kernel(xij, rij, s_h[s_idx])
            d_rho[d_idx] += rho_sum


class EulerStep(IntegratorStep):
    """Fast but inaccurate integrator. Use this for testing"""
    def initialize(self, d_u, d_v, d_u_prev_step, d_v_prev_step, d_idx):
        d_u_prev_step[d_idx] = d_u[d_idx]
        d_v_prev_step[d_idx] = d_v[d_idx]

    def stage1(self, d_idx, d_u, d_v, d_au, d_av, d_x, d_y, dt):
        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]


class SWEStep(IntegratorStep):
    """Leap frog time integration scheme"""
    def initialize(self, t, d_u, d_v, d_uh, d_vh, d_u_prev_step, d_v_prev_step,
                   d_idx):
        # Stores the velocities at previous time step
        d_u_prev_step[d_idx] = d_u[d_idx]
        d_v_prev_step[d_idx] = d_v[d_idx]

    def stage1(self, d_uh, d_vh, d_idx, d_au, d_av, dt):
        # Velocities at half time step
        d_uh[d_idx] += dt * d_au[d_idx]
        d_vh[d_idx] += dt * d_av[d_idx]

    def stage2(self, d_u, d_v, d_uh, d_vh, d_idx, d_au, d_av, d_x, d_y, dt):
        d_x[d_idx] += dt * d_uh[d_idx]
        d_y[d_idx] += dt * d_vh[d_idx]
        d_u[d_idx] = d_uh[d_idx] + dt/2.*d_au[d_idx]
        d_v[d_idx] = d_vh[d_idx] + dt/2.*d_av[d_idx]


class SWEIntegrator(Integrator):
    """Integrator for shallow water problems"""
    def one_timestep(self, t, dt):
        self.compute_accelerations()

        self.initialize()

        # Predict
        self.stage1()

        # Call any post-stage functions.
        self.do_post_stage(0.5*dt, 1)

        # Correct
        self.stage2()

        # Call any post-stage functions.
        self.do_post_stage(dt, 2)


class GatherDensityEvalNextIteration(Equation):
    r"""**Gather formulation for evaluating the density of a particle**

    .. math::

        \rho_i = \sum_{j}{m_jW(\textbf{x}_i - \textbf{x}_j, h_i)}

    References
    ----------
    .. [Hernquist and Katz, 1988] L. Hernquist and N. Katz, "TREESPH: A
    unification of SPH with the hierarcgical tree method", The Astrophysical
    Journal Supplement Series, 70 (1989), pp 419-446.

    """
    def initialize(self, d_rho, d_idx, d_rho_prev_iter):
        # Stores density of particle i of the previous iteration
        d_rho_prev_iter[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = 0.0

    def loop(self, d_rho, d_idx, s_m, s_idx, WI):
        d_rho[d_idx] += s_m[s_idx] * WI


class ScatterDensityEvalNextIteration(Equation):
    r"""**Scatter formulation for evaluating the density of a particle**

    .. math::

        \rho_i = \sum_{J}{m_JW(\textbf{x}_i - \textbf{x}_j, h_j)}

    References
    ----------
    .. [Hernquist and Katz, 1988] L. Hernquist and N. Katz, "TREESPH: A
    unification of SPH with the hierarcgical tree method", The Astrophysical
    Journal Supplement Series, 70 (1989), pp 419-446.

    """
    def initialize(self, t, d_rho, d_idx, d_rho_prev_iter):
        # Stores density of particle i of the previous iteration
        d_rho_prev_iter[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = 0.0

    def loop(self, d_rho, d_idx, s_m, s_idx, WJ):
        d_rho[d_idx] += s_m[s_idx] * WJ


class NonDimensionalDensityResidual(Equation):
    r"""**Non-dimensional density residual**

    .. math::

        \psi^{k+1} = \dfrac{|\rho_i^{k+1} - \rho_i^k|}{\rho_i^k}

    References
    ----------
    .. [Vacondio2010] R. Vacondio, B.D. Rodgers and P.K. Stansby, "Accurate
    particle splitting for smoothed particle hydrodynamics in shallow water
    with shock capturing", Int. J. Numer. Meth. Fluids, 69 (2012), pp.
    1377-1410

    """
    def __init__(self, dest, sources=None):
        super(NonDimensionalDensityResidual, self).__init__(dest, sources)

    def post_loop(self, d_psi, d_rho, d_rho_prev_iter, d_idx):
        # Non-dimensional residual
        d_psi[d_idx] = abs(d_rho[d_idx] - d_rho_prev_iter[d_idx]) \
                       / d_rho_prev_iter[d_idx]


class CheckConvergenceDensityResidual(Equation):
    r"""The iterative process is stopped once the following condition is met

    .. math::

        \psi^{k+1} < \epsilon_{\psi}

    where,

        \epsilon_{\psi} = 1e-3


    References
    ----------
    .. [Vacondio2010] R. Vacondio, B.D. Rodgers and P.K. Stansby, "Accurate
    particle splitting for smoothed particle hydrodynamics in shallow water
    with shock capturing", Int. J. Numer. Meth. Fluids, 69 (2012), pp.
    1377-1410

    Notes
    -----
    If particle splitting is activated, better to use this convergence
    criteria. It can be used even if particle splitting is not activated.

    """
    def __init__(self, dest, sources=None):
        super(CheckConvergenceDensityResidual, self).__init__(dest, sources)
        self.eqn_has_converged = 0

    def initialize(self):
        self.eqn_has_converged = 0

    def reduce(self, dst, t, dt):
        epsilon = max(dst.psi)
        if epsilon <= 1e-3:
            self.eqn_has_converged = 1

    def converged(self):
        return self.eqn_has_converged


class CorrectionFactorVariableSmoothingLength(Equation):
    r"""**Correction factor in internal force due to variable smoothing
    length**

    .. math::

        \alpha_i = -\sum_j m_j r_{ij}\frac{dW_i}{dr_{ij}}

    References
    ----------
    .. [Rodriguez and Bonet, 2005] M. Rodriguez and J. Bonet, "A corrected
    smooth particle hydrodynamics formulation of the shallow-water equations",
    Computers and Structures, 83 (2005), pp. 1396-1410

    """
    def initialize(self, d_idx, d_alpha):
        d_alpha[d_idx] = 0.0

    def loop(self, d_alpha, d_idx, DWIJ, XIJ, s_idx, s_m):
        d_alpha[d_idx] += -s_m[s_idx] * (DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1])


class RemoveParticlesWithZeroAlpha(Equation):
    r"""Removes particles if correction factor (alpha) in internal force due to
    variable smoothing length is zero

    """
    def __init__(self, dest):
        super(RemoveParticlesWithZeroAlpha, self).__init__(dest, None)

    def post_loop(self, d_alpha, d_pa_alpha_zero, d_idx):
        if d_alpha[d_idx] == 0:
            d_pa_alpha_zero[d_idx] = 1

    def reduce(self, dst, t, dt):
        indices = declare('object')
        indices = numpy.where(dst.pa_alpha_zero > 0)[0]
        if len(indices) > 0:
            dst.remove_particles(indices)


class SummationDensity(Equation):
    r"""**Summation Density**

    .. math::

        \rho_i = \sum_{j}{m_jW(\textbf{x}_i - \textbf{x}_j, h_i)}

    """
    def initialize(self, d_summation_rho, d_idx):
        d_summation_rho[d_idx] = 0.0

    def loop(self, d_summation_rho, d_idx, s_m, s_idx, WI):
        d_summation_rho[d_idx] += s_m[s_idx] * WI


class InitialGuessDensityVacondio(Equation):
    r"""**Initial guess density to start the iterative evaluation of density
    for time step n+1**

    .. math::

        \rho_{i(0)}^{n+1} = \rho_i^n + dt\dfrac{d\rho_i}{dt}\\
        h_{i(0)}^{n+1} = h_i^n + -\dfrac{h_i^n}{\rho_i^n}\dfrac{dt}{dm}
        \dfrac{d\rho_i}{dt}

    where,

    .. math::

        \frac{d\rho_i}{dt} = \rho_i^n\sum_j\dfrac{m_j}{\rho_j}
        (\textbf{v}_i-\textbf{v}_j).\nabla W_i

    References
    ----------
    .. [VacondioSWE-SPHysics, 2013] R. Vacondio et al., SWE-SPHysics source
    code, File: SWE_SPHYsics/SWE-SPHysics_2D_v1.0.00/source/SPHYSICS_SWE_2D/
    ac_dw_var_hj_2D.f

    Note:
    If particle splitting is activated, better to use this method. It can be
    used even if particle splitting is not activated.

    """
    def __init__(self, dest, sources, dim=2):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)

        """
        self.dim = dim
        super(InitialGuessDensityVacondio, self).__init__(dest, sources)

    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0

    def loop(self, d_arho, d_rho, d_idx, s_m, s_rho, s_idx, d_u_prev_step,
             d_v_prev_step, s_u_prev_step, s_v_prev_step, DWI):
        tmp1 = (d_u_prev_step[d_idx]-s_u_prev_step[s_idx]) * DWI[0]
        tmp2 = (d_v_prev_step[d_idx]-s_v_prev_step[s_idx]) * DWI[1]
        d_arho[d_idx] += d_rho[d_idx] * ((s_m[s_idx]/s_rho[s_idx])*(tmp1+tmp2))

    def post_loop(self, d_rho, d_h, dt, d_arho, d_idx):
        d_rho[d_idx] += dt * d_arho[d_idx]
        d_h[d_idx] += -(dt/self.dim)*d_h[d_idx]*(d_arho[d_idx]/d_rho[d_idx])


class InitialGuessDensity(Equation):
    r"""**Initial guess density to start the iterative evaluation of density
    for time step n+1 based on properties of time step n**

    .. math::

        \rho_{I, n+1}^{(0)} = \rho_{I,n}e^{\lambda_n}

    where,

        \lambda = \dfrac{\rho_Id_m}{\alpha_I}\sum_{J}^{}m_J
        (\textbf{v}_J - \textbf{v}_I).\nabla W_I(\textbf{x}_I
        - \textbf{x}_J, h_I)

    References
    ----------
    .. [Rodriguez and Bonet, 2005] M. Rodriguez and J. Bonet, "A corrected
    smooth particle hydrodynamics formulation of the shallow-water equations",
    Computers and Structures, 83 (2005), pp. 1396-1410

    """
    def __init__(self, dest, sources, dim=2):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)

        """
        self.dim = dim
        super(InitialGuessDensity, self).__init__(dest, sources)

    def initialize(self, d_exp_lambda, d_idx):
        d_exp_lambda[d_idx] = 0.0

    def loop(self, d_exp_lambda, d_u_prev_step, d_v_prev_step, d_alpha, d_idx,
             s_m, s_u_prev_step, s_v_prev_step, s_idx, DWI, dt, t):
        a1 = (d_u_prev_step[d_idx]-s_u_prev_step[s_idx]) * DWI[0]
        a2 = (d_v_prev_step[d_idx]-s_v_prev_step[s_idx]) * DWI[1]
        const = (self.dim*dt) / d_alpha[d_idx]
        d_exp_lambda[d_idx] += const * (s_m[s_idx]*(a1+a2))

    def post_loop(self, t, d_rho, d_exp_lambda, d_idx):
        d_rho[d_idx] = d_rho[d_idx] * exp(d_exp_lambda[d_idx])


class UpdateSmoothingLength(Equation):
    r"""**Update smoothing length based on density**

    .. math::

        h_I^{(k)} = h_I^{0}\biggl(\dfrac{\rho_I^0}{\rho_I^{(k)}}
        \biggl)^\frac{1}{d_m}


    References
    ----------
    .. [Rodriguez and Bonet, 2005] M. Rodriguez and J. Bonet, "A corrected
    smooth particle hydrodynamics formulation of the shallow-water equations",
    Computers and Structures, 83 (2005), pp. 1396-1410

    """
    def __init__(self, dest, dim=2):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)

        """
        self.dim = dim
        super(UpdateSmoothingLength, self).__init__(dest, None)

    def post_loop(self, d_h, d_h0, d_rho0, d_rho, d_idx):
        d_h[d_idx] = d_h0[d_idx] * (d_rho0[d_idx]/d_rho[d_idx])**(1./self.dim)


class DensityResidual(Equation):
    r"""**Residual of density**

    .. math::

        R(\rho^{(k)}) = \rho_I^{(k)} - \sum_{J}^{}m_J
        W_I(\textbf{x}_I - \textbf{x}_J, h_I^{(k)})

    References
    ----------
    .. [Rodriguez and Bonet, 2005] M. Rodriguez and J. Bonet, "A corrected
    smooth particle hydrodynamics formulation of the shallow-water equations",
    Computers and Structures, 83 (2005), pp. 1396-1410

    """
    def __init__(self, dest, sources=None):
        super(DensityResidual, self).__init__(dest, sources)

    def post_loop(self, d_rho, d_idx, d_rho_residual, d_summation_rho, t):
        d_rho_residual[d_idx] = d_rho[d_idx] - d_summation_rho[d_idx]


class DensityNewtonRaphsonIteration(Equation):
    r"""**Newton-Raphson approximate solution for the density equation at
    iteration k+1**

    .. math::

        \rho^{(k+1)} = \rho_I^{(k)}\biggl[1 - \dfrac{R_I^{(k)}d_m}{(
        R_I^{(k)} d_m + \alpha_I^k)}\biggr]

    References
    ----------
    .. [Rodriguez and Bonet, 2005] M. Rodriguez and J. Bonet, "A corrected
    smooth particle hydrodynamics formulation of the shallow-water equations",
    Computers and Structures, 83 (2005), pp. 1396-1410

    """
    def __init__(self, dest, sources=None, dim=2):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)

        """
        self.dim = dim
        super(DensityNewtonRaphsonIteration, self).__init__(dest, sources)

    def initialize(self, d_rho, d_rho_prev_iter, d_idx):
        d_rho_prev_iter[d_idx] = d_rho[d_idx]

    def post_loop(self, d_rho, d_idx, d_alpha, d_rho_residual):
        a1 = d_rho_residual[d_idx] * self.dim
        a2 = a1 + d_alpha[d_idx]
        const = 1 - (a1/a2)
        d_rho[d_idx] = d_rho[d_idx] * const


class CheckConvergence(Equation):
    r"""Stops the Newton-Raphson iterative procedure if the following
    convergence criteria is satisfied:

    .. math::

        \dfrac{|R_I^{(k+1)}|}{\rho_I^{(k)}} \leq \epsilon

    where,

        \epsilon = 1e-15

    References
    ----------
    .. [Rodriguez and Bonet, 2005] M. Rodriguez and J. Bonet, "A corrected
    smooth particle hydrodynamics formulation of the shallow-water equations",
    Computers and Structures, 83 (2005), pp. 1396-1410

    Notes
    -----
    Use this convergence criteria when using the Newton-Raphson iterative
    procedure.

    """
    def __init__(self, dest, sources=None):
        super(CheckConvergence, self).__init__(dest, sources)
        self.eqn_has_converged = 0

    def initialize(self):
        self.eqn_has_converged = 0

    def post_loop(self, d_positive_rho_residual, d_rho_residual,
                  d_rho_prev_iter,  d_idx, t):
        d_positive_rho_residual[d_idx] = abs(d_rho_residual[d_idx])

    def reduce(self, dst, t, dt):
        max_epsilon = max(dst.positive_rho_residual / dst.rho_prev_iter)
        if max_epsilon <= 1e-15:
            self.eqn_has_converged = 1

    def converged(self):
        return self.eqn_has_converged


class SWEOS(Equation):
    r"""**Update fluid properties based on density**

    References
    ----------
    .. [Rodriguez and Bonet, 2005] M. Rodriguez and J. Bonet, "A corrected
    smooth particle hydrodynamics formulation of the shallow-water equations",
    Computers and Structures, 83 (2005), pp. 1396-1410

    """
    def __init__(self, dest, sources=None, g=9.81, rhow=1000.0):
        r"""
        Parameters
        ----------
        g : float
            acceleration due to gravity
        rhow : float
            constant 3-D density of water

        """
        self.rhow = rhow
        self.g = g
        self.fac = 0.5 * (g/rhow)
        super(SWEOS, self).__init__(dest, sources)

    def post_loop(self, d_rho, d_cs, d_u, d_v, d_idx, d_p, d_dw, d_dt_cfl,
                  d_m, d_A, d_alpha):
        # Pressure
        d_p[d_idx] = self.fac * (d_rho[d_idx])**2

        # Wave speed
        d_cs[d_idx] = sqrt(self.g * d_rho[d_idx]/self.rhow)

        # Area
        d_A[d_idx] = d_m[d_idx] / d_rho[d_idx]

        # Depth of water
        d_dw[d_idx] = d_rho[d_idx] / self.rhow

        # dt = CFL * (h_min / max(dt_cfl))
        d_dt_cfl[d_idx] = d_cs[d_idx] + (d_u[d_idx]**2 + d_v[d_idx]**2)**0.5


def mu_calc(hi=1.0, hj=1.0, velij_dot_rij=1.0, rij2=1.0):
    r"""Term present in the artificial viscosity formulation (Monaghan)

    .. math::

      \mu_{ij} = \dfrac{\bar h_{ij}\textbf{v}_{ij}.\textbf{x}_{ij}}
      {|\textbf{x}_{ij}|^2 + \zeta^2}

    References
    ----------
    .. [Monaghan2005] J. Monaghan, "Smoothed particle hydrodynamics",
        Reports on Progress in Physics, 68 (2005), pp. 1703-1759.

    """
    h_bar = (hi+hj) / 2.0
    eta2 = 0.01 * hi**2
    muij = (h_bar*velij_dot_rij) / (rij2+eta2)
    return muij


def artificial_visc(alpha=1.0, rij2=1.0,  hi=1.0, hj=1.0, rhoi=1.0, rhoj=1.0,
                    csi=1.0, csj=1.0, muij=1.0):
    r"""**Artificial viscosity based stabilization term (Monaghan)**

    Activated when :math:`\textbf{v}_{ij}.\textbf{x}_{ij} < 0`

    Given by

    .. math::

    \Pi_{ij} = \dfrac{-a\bar c_{ij}\mu_{ij}+b\bar c_{ij}\mu_{ij}^2}{\rho_{ij}}

    References
    ----------
    .. [Monaghan2005] J. Monaghan, "Smoothed particle hydrodynamics",
        Reports on Progress in Physics, 68 (2005), pp. 1703-1759.

    """
    cs_bar = (csi+csj) / 2.0
    rho_bar = (rhoi+rhoj) / 2.0
    pi_visc = -(alpha*cs_bar*muij) / rho_bar
    return pi_visc


def viscosity_LF(alpha=1.0, rij2=1.0, hi=1.0, hj=1.0, rhoi=1.0, rhoj=1.0,
                 csi=1.0, csj=1.0, muij=1.0):
    r"""**Lax-Friedrichs flux based stabilization term (Ata and Soulaimani)**

    .. math::

    \Pi_{ij} = \dfrac{\bar c_{ij}\textbf{v}_{ij}.\textbf{x}_{ij}}
    {\bar\rho_{ij}\sqrt{|x_{ij}|^2 + \zeta^2}}

    References
    ----------
    .. [Ata and Soulaimani, 2004] R. Ata and A. Soulaimani, "A stabilized SPH
    method for inviscid shallow water", Int. J. Numer. Meth. Fluids, 47 (2005),
    pp. 139-159.

    Notes
    -----
    The advantage of this term is that it automatically sets the required level
    of numerical viscosity based on the Lax-Friedrichs flux. This is the
    default stabilization method.

    """
    cs_bar = (csi+csj) / 2.0
    rho_bar = (rhoi+rhoj) / 2.0
    eta2 = 0.01 * hi**2
    h_bar = (hi+hj) / 2.0
    tmp = (muij*(rij2+eta2)**0.5) / h_bar
    pi_visc = -(cs_bar*tmp) / rho_bar
    return pi_visc


class ParticleAcceleration(Equation):
    r"""**Acceleration of a particle**

    .. math::

        \textbf{a}_i = -\frac{g+\textbf{v}_i.\textbf{k}_i\textbf{v}_i
                       -\textbf{t}_i.\nabla H_i}{1+\nabla H_i.\nabla H_i}
                       \nabla H_i - \textbf{t}_i - \textbf{S}_{fi}

    where,

    .. math::

        \textbf{t}_i &= \sum_j m_j\ \biggl[\biggl(\frac{p_j}{
        \alpha_j \rho_j^2}+0.5\Pi_{ij}\biggr)\nabla W_j(\textbf{x}_i, h_j) -
        \biggl(\frac{p_i}{\alpha_i \rho_i^2}+0.5\Pi_{ij}\biggr)\nabla
        W_i(\textbf{x}_j, h_i)\biggr]

    .. math::

        \textbf{S}_f = \textbf{v}\dfrac{gn^2|\textbf{v}|}{d^{\frac{4}{3}}}

    with,

    .. math::

        \alpha_i = -\sum_j m_j r_{ij}\frac{dW_i}{dr_{ij}}

    .. math::

        n_i  = \sum_jn_j^b\overline W_i(x_i - x_j^b, h^b)V_j

    References
    ----------
    .. [Vacondio2010] R. Vacondio, B.D. Rodgers and P.K. Stansby, "Accurate
    particle splitting for smoothed particle hydrodynamics in shallow water
    with shock capturing", Int. J. Numer. Meth. Fluids, 69 (2012), pp.
    1377-1410

    Notes
    -----
    The acceleration term given in [Vacondio2010] has incorrect sign.

    """
    def __init__(self, dest, sources, dim=2, u_only=False, v_only=False,
                 alpha=0.0, visc_option=2, rhow=1000.0):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)
        u_only : bool
            motion of fluid column in x-direction only evaluated
            (Default: False)
        v_only : bool
            motion of fluid column in y-direction only evaluated
            (Default: False)
        alpha : float
            coefficient to control amount of artificial viscosity (Monaghan)
            (Default: 0.0)
        visc_option : int
            artifical viscosity (1) or Lax-Friedrichs flux (2) based
            stabilization term (Default: 2)
        rhow : float
            constant 3-D density of water

        """
        self.g = 9.81
        self.rhow = rhow
        self.ct = self.g / (2*self.rhow)
        self.dim = dim
        self.u_only = u_only
        self.v_only = v_only
        self.alpha = alpha
        if visc_option == 1:
            self.viscous_func = artificial_visc
        else:
            self.viscous_func = viscosity_LF
        super(ParticleAcceleration, self).__init__(dest, sources)

    def initialize(self, d_idx, d_tu, d_tv):
        d_tu[d_idx] = 0.0
        d_tv[d_idx] = 0.0

    def loop(self, d_x, d_y, s_x, s_y, d_rho, d_idx, s_m, s_idx, s_rho, d_m,
             DWI, DWJ, d_au, d_av, s_alpha, d_alpha, s_p, d_p, d_tu, s_dw,
             d_dw, t, s_is_wall_boun_pa, s_tu, d_tv, s_tv, d_h, s_h, d_u, s_u,
             d_v, s_v, d_cs, s_cs):
        # True if neighbor is wall boundary particle
        if s_is_wall_boun_pa[s_idx] == 1:

            # Setting artificial viscosity to zero when a particle interacts
            # with wall boundary particles
            pi_visc = 0.0

            # Setting water depth of wall boundary particles same as particle
            # interacting with it (For sufficient pressure to prevent wall
            # penetration)
            s_dw[s_idx] = d_dw[d_idx]
        else:
            uij = d_u[d_idx] - s_u[s_idx]
            vij = d_v[d_idx] - s_v[s_idx]
            xij = d_x[d_idx] - s_x[s_idx]
            yij = d_y[d_idx] - s_y[s_idx]
            rij2 = xij**2 + yij**2
            uij_dot_xij = uij * xij
            vij_dot_yij = vij * yij
            velij_dot_rij = uij_dot_xij + vij_dot_yij

            muij = mu_calc(d_h[d_idx], s_h[s_idx], velij_dot_rij, rij2)

            if velij_dot_rij < 0:
                # Stabilization term
                pi_visc = self.viscous_func(self.alpha, rij2, d_h[d_idx],
                                            s_h[s_idx], d_rho[d_idx],
                                            s_rho[s_idx], d_cs[d_idx],
                                            s_cs[s_idx], muij)
            else:
                pi_visc = 0

        tmp1 = (s_dw[s_idx]*self.rhow*self.dim) / s_alpha[s_idx]
        tmp2 = (d_dw[d_idx]*self.rhow*self.dim) / d_alpha[d_idx]

        # Internal force per unit mass
        d_tu[d_idx] += s_m[s_idx] * ((self.ct*tmp1 + 0.5*pi_visc)*DWJ[0] +
                                     (self.ct*tmp2 + 0.5*pi_visc)*DWI[0])

        d_tv[d_idx] += s_m[s_idx] * ((self.ct*tmp1 + 0.5*pi_visc)*DWJ[1] +
                                     (self.ct*tmp2 + 0.5*pi_visc)*DWI[1])

    def _get_helpers_(self):
        return [mu_calc, artificial_visc, viscosity_LF]

    def post_loop(self, d_idx, d_u, d_v, d_tu, d_tv, d_au, d_av, d_Sfx, d_Sfy,
                  d_bx, d_by, d_bxx, d_bxy, d_byy):
        vikivi = d_u[d_idx]*d_u[d_idx]*d_bxx[d_idx] \
                 + 2*d_u[d_idx]*d_v[d_idx]*d_bxy[d_idx] \
                 + d_v[d_idx]*d_v[d_idx]*d_byy[d_idx]

        tidotgradbi = d_tu[d_idx]*d_bx[d_idx] + d_tv[d_idx]*d_by[d_idx]
        gradbidotgradbi = d_bx[d_idx]**2 + d_by[d_idx]**2

        temp3 = self.g + vikivi - tidotgradbi
        temp4 = 1 + gradbidotgradbi

        if not self.v_only:
            # Acceleration along x-direction
            d_au[d_idx] = -(temp3/temp4)*d_bx[d_idx] - d_tu[d_idx] \
                          - d_Sfx[d_idx]
        if not self.u_only:
            # Acceleration along y-direction
            d_av[d_idx] = -(temp3/temp4)*d_by[d_idx] - d_tv[d_idx] \
                          - d_Sfy[d_idx]


class FluidBottomElevation(Equation):
    r"""**Bottom elevation of fluid**

    .. math::

        b_i = \sum_jb_j^b\overline{W_i}(\textbf{x}_i - \textbf{x}_j^b, h^b)V_j

    References
    ----------
    .. [Vacondio2010] R. Vacondio, B.D. Rodgers and P.K. Stansby, "Accurate
    particle splitting for smoothed particle hydrodynamics in shallow water
    with shock capturing", Int. J. Numer. Meth. Fluids, 69 (2012), pp.
    1377-1410

    """
    def initialize(self, d_b, d_idx):
        d_b[d_idx] = 0.0

    def loop_all(self, d_shep_corr, d_x, d_y, d_idx, s_x, s_y, s_V, s_idx, s_h,
                 SPH_KERNEL, NBRS, N_NBRS):
        # Shepard filter
        i = declare('int')
        xij = declare('matrix(3)')
        rij = 0.0
        corr_sum = 0.0
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1])
            corr_sum += s_V[s_idx] * SPH_KERNEL.kernel(xij, rij, s_h[s_idx])
        d_shep_corr[d_idx] = corr_sum

    def loop(self, d_b, d_idx, s_b, s_idx, WJ, s_V, RIJ):
        d_b[d_idx] += s_b[s_idx] * WJ * s_V[s_idx]

    def post_loop(self, d_b, d_shep_corr, d_idx):
        if d_shep_corr[d_idx] > 1e-14:
            d_b[d_idx] /= d_shep_corr[d_idx]


class FluidBottomGradient(Equation):
    r"""**Bottom gradient of fluid**

    .. math::

        \nabla b_i &=& \sum_j\nabla b_j^b W_i(\textbf{x}_i - \textbf{x}_j^b,
        h^b)V_j

    Notes:
    It is obtained from a simple SPH interpolation from the gradient of bed
    particles

    """
    def initialize(self, d_idx, d_bx, d_by):
        d_bx[d_idx] = 0.0
        d_by[d_idx] = 0.0

    def loop(self, d_idx, d_bx, d_by, WJ, s_idx, s_bx, s_by, s_V):
        # Bottom gradient of fluid
        d_bx[d_idx] += s_bx[s_idx] * WJ * s_V[s_idx]
        d_by[d_idx] += s_by[s_idx] * WJ * s_V[s_idx]


class FluidBottomCurvature(Equation):
    r"""Bottom curvature of fluid**

    .. math::

        \nabla^2 b_i = \sum_j\nabla^2 b_j^b W_i(\textbf{x}_i - \textbf{x}_j^b,
        h^b)V_j

    Notes:
    It is obtained from a simple SPH interpolation from the curvature of bed
    particles

    """
    def initialize(self, d_idx, d_bx, d_by, d_bxx, d_bxy, d_byy):
        d_bxx[d_idx] = 0.0
        d_bxy[d_idx] = 0.0
        d_byy[d_idx] = 0.0

    def loop(self, d_idx, d_bxx, d_bxy, d_byy, WJ, s_idx, s_bxx, s_bxy, s_byy,
             s_V):
        # Bottom curvature of fluid
        d_bxx[d_idx] += s_bxx[s_idx] * WJ * s_V[s_idx]
        d_bxy[d_idx] += s_bxy[s_idx] * WJ * s_V[s_idx]
        d_byy[d_idx] += s_byy[s_idx] * WJ * s_V[s_idx]


class BedGradient(Equation):
    r"""**Gradient of bed**

    .. math::

        \nabla b_i = \sum_jb_j^b\tilde{\nabla}W_i(\textbf{x}_i -
        \textbf{x}_j^b, h^b)V_j

    References
    ----------
    .. [Vacondio2010] R. Vacondio, B.D. Rodgers and P.K. Stansby, "Accurate
    particle splitting for smoothed particle hydrodynamics in shallow water
    with shock capturing", Int. J. Numer. Meth. Fluids, 69 (2012), pp.
    1377-1410

    """
    def initialize(self, d_bx, d_by, d_idx):
        d_bx[d_idx] = 0.0
        d_by[d_idx] = 0.0

    def loop(self, d_bx, d_by, d_idx, s_b, s_idx, DWJ, s_V, RIJ):
        if RIJ > 1e-6:
            # Gradient of bed
            d_bx[d_idx] += s_b[s_idx] * DWJ[0] * s_V[s_idx]
            d_by[d_idx] += s_b[s_idx] * DWJ[1] * s_V[s_idx]


class BedCurvature(Equation):
    r"""**Curvature of bed**

    .. math::

        \biggl(\dfrac{\partial^2b}{\partial x^\alpha \partial x^\beta}
        \biggr)_i = \sum_{j}^{}\biggl(4\dfrac{x_{ij}^\alphax_{ij}^\beta}
        {r_{ij}^2}-\delta^{\alpha\beta}\biggr)\dfrac{b_i - b_j^b}{
        \textbf{r}_{ij}\textbf{r}_{ij} + \eta^2}\textbf{r}_{ij}.\tilde{\nabla}
        W_i(\textbf{x}_i - \textbf{x}_j^b, h^b)V_j

    References
    ----------
    .. [Vacondio2010] R. Vacondio, B.D. Rodgers and P.K. Stansby, "Accurate
    particle splitting for smoothed particle hydrodynamics in shallow water
    with shock capturing", Int. J. Numer. Meth. Fluids, 69 (2012), pp.
    1377-1410

    """
    def initialize(self, d_bxx, d_bxy, d_byy, d_idx):
        d_bxx[d_idx] = 0.0
        d_bxy[d_idx] = 0.0
        d_byy[d_idx] = 0.0

    def loop(self, d_bxx, d_bxy, d_byy, d_b, d_idx, s_h, s_b, s_idx, XIJ, RIJ,
             DWJ, s_V):
        if RIJ > 1e-6:
            eta = 0.01 * s_h[s_idx]
            temp1 = (d_b[d_idx]-s_b[s_idx]) / (RIJ**2+eta**2)
            temp2 = XIJ[0]*DWJ[0] + XIJ[1]*DWJ[1]
            temp_bxx = ((4*XIJ[0]**2/RIJ**2)-1) * temp1
            temp_bxy = (4*XIJ[0]*XIJ[1]/RIJ**2) * temp1
            temp_byy = ((4*XIJ[1]**2/RIJ**2)-1) * temp1
            # Curvature of bed
            d_bxx[d_idx] += temp_bxx * temp2 * s_V[s_idx]
            d_bxy[d_idx] += temp_bxy * temp2 * s_V[s_idx]
            d_byy[d_idx] += temp_byy * temp2 * s_V[s_idx]


class BedFrictionSourceEval(Equation):
    r"""**Friction source term**

    .. math::

        \textbf{S}_f = \textbf{v}\dfrac{gn^2|\textbf{v}|}{d^{\frac{4}{3}}}

    where,

    .. math::

        n_i  = \sum_jn_j^b\overline W_i(x_i - x_j^b, h^b)V_j

    """
    def __init__(self, dest, sources):
        self.g = 9.8
        super(BedFrictionSourceEval, self).__init__(dest, sources)

    def initialize(self, d_n, d_idx):
        d_n[d_idx] = 0.0

    def loop(self, d_n, d_idx, s_n, s_idx, WJ, s_V, RIJ):
        if RIJ > 1e-6:
            # Manning coefficient
            d_n[d_idx] += s_n[s_idx] * WJ * s_V[s_idx]

    def post_loop(self, d_idx, d_Sfx, d_Sfy, d_u, d_v, d_n, d_dw):
        vmag = sqrt(d_u[d_idx]**2 + d_v[d_idx]**2)
        temp = (self.g*d_n[d_idx]**2*vmag) / d_dw[d_idx]**(4.0/3.0)
        # Friction source term
        d_Sfx[d_idx] = d_u[d_idx] * temp
        d_Sfy[d_idx] = d_v[d_idx] * temp


class BoundaryInnerReimannStateEval(Equation):
    r"""Evaluates the inner Riemann state of velocity and depth

    .. math::

        \textbf{v}_i^o = \sum_j\dfrac{m_j^f}{\rho_j^f}\textbf{v}_j^f\bar
        W_i(\textbf{x}_i^o - \textbf{x}_j^f, h_o)\\

        {d}_i^o = \sum_j\dfrac{m_j^f}{\rho_j^f}d_j^f\bar W_i(\textbf{x}_i^o -
        \textbf{x}_j^f, h_o)

    References
    ----------
    .. [Vacondio2012] R. Vacondio et al., "SPH modeling of shallow flow with
    open boundaries for practical flood simulation", J. Hydraul. Eng., 2012,
    138(6), pp. 530-541.

    """
    def initialize(self, d_u_inner_reimann, d_v_inner_reimann,
                   d_dw_inner_reimann, d_idx):
        d_u_inner_reimann[d_idx] = 0.0
        d_v_inner_reimann[d_idx] = 0.0
        d_dw_inner_reimann[d_idx] = 0.0

    def loop_all(self, d_shep_corr, d_x, d_y, d_idx, s_x, s_y, s_m, s_rho,
                 s_idx, d_h, SPH_KERNEL, NBRS, N_NBRS):
        # Shepard filter
        i = declare('int')
        xij = declare('matrix(3)')
        rij = 0.0
        corr_sum = 0.0
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1])
            corr_sum += ((s_m[s_idx]/s_rho[s_idx])
                         * SPH_KERNEL.kernel(xij, rij, d_h[d_idx]))
        d_shep_corr[d_idx] = corr_sum

    def loop(self, d_u_inner_reimann, d_v_inner_reimann, d_dw_inner_reimann,
             d_idx, WI, s_m, s_u, s_v, s_rho, s_dw, s_idx):
        tmp = WI * (s_m[s_idx]/s_rho[s_idx])
        # Riemann invariants at open boundaries
        d_u_inner_reimann[d_idx] += s_u[s_idx] * tmp
        d_v_inner_reimann[d_idx] += s_v[s_idx] * tmp
        d_dw_inner_reimann[d_idx] += s_dw[s_idx] * tmp

    def post_loop(self, d_u_inner_reimann, d_v_inner_reimann,
                  d_dw_inner_reimann, d_shep_corr, d_idx):
        if d_shep_corr[d_idx] > 1e-14:
            d_u_inner_reimann[d_idx] /= d_shep_corr[d_idx]
            d_v_inner_reimann[d_idx] /= d_shep_corr[d_idx]
            d_dw_inner_reimann[d_idx] /= d_shep_corr[d_idx]


class SubCriticalInFlow(Equation):
    r"""**Subcritical inflow condition**

    ..math ::

        d_B = \biggl[\frac{1}{2\sqrt{g}}(v_{B,n}-v_{I,n}) + \sqrt{d_I}\biggr]^2

    References
    ----------
    .. [Vacondio2012] R. Vacondio et al., "SPH modeling of shallow flow with
    open boundaries for practical flood simulation", J. Hydraul. Eng., 2012,
    138(6), pp. 530-541.

    Notes
    -----
    The velocity is imposed at the open boundary.

    """
    def __init__(self, dest, dim=2, rhow=1000.0):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)
        rhow : float
            constant 3-D density of water

        """
        self.g = 9.8
        self.dim = dim
        self.rhow = rhow
        super(SubCriticalInFlow, self).__init__(dest, None)

    def post_loop(self, d_dw, d_dw_inner_reimann, d_u, d_u_inner_reimann,
                  d_rho, d_alpha, d_cs, d_idx):
        const = 1. / (2.*sqrt(self.g))
        # Properties of open boundary particles
        d_dw[d_idx] = (const*(d_u_inner_reimann[d_idx] - d_u[d_idx])
                       + sqrt(d_dw_inner_reimann[d_idx]))**2
        d_rho[d_idx] = d_dw[d_idx] * self.rhow
        d_alpha[d_idx] = self.dim * d_rho[d_idx]
        d_cs[d_idx] = sqrt(self.g * d_dw[d_idx])


class SubCriticalOutFlow(Equation):
    r"""**Subcritical outflow condition**

    ..math ::

        v_{B,n} = v_{I,n} + 2\sqrt{g}(\sqrt{d_I} - \sqrt{d_B}), v_{B,t} =
        v_{I,t}

    References
    ----------
    .. [Vacondio2012] R. Vacondio et al., "SPH modeling of shallow flow with
    open boundaries for practical flood simulation", J. Hydraul. Eng., 2012,
    138(6), pp. 530-541.

    Notes:
    -----
    The constant water depth is imposed at the open boundary.

    """
    def __init__(self, dest, dim=2, rhow=1000.0):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)
        rhow : float
            constant 3-D density of water

        """
        self.g = 9.8
        self.dim = dim
        self.rhow = rhow
        super(SubCriticalOutFlow, self).__init__(dest, None)

    def post_loop(self, d_dw, d_dw_inner_reimann, d_u, d_u_inner_reimann,
                  d_rho, d_cs, d_alpha, d_v, d_v_inner_reimann, d_idx):
        const = 2. * sqrt(self.g)
        # Velocities of open boundary particles
        d_u[d_idx] = (d_u_inner_reimann[d_idx]
                      + const*(sqrt(d_dw_inner_reimann[d_idx])
                               - sqrt(d_dw[d_idx])))
        d_v[d_idx] = d_v_inner_reimann[d_idx]


class SubCriticalTimeVaryingOutFlow(Equation):
    r"""**Subcritical outflow condition**

    ..math ::

        v_{B,n} = v_{I,n} + 2\sqrt{g}(\sqrt{d_I} - \sqrt{d_B}), v_{B,t} =
        v_{I,t}

    References
    ----------
    .. [Vacondio2012] R. Vacondio et al., "SPH modeling of shallow flow with
    open boundaries for practical flood simulation", J. Hydraul. Eng., 2012,
    138(6), pp. 530-541.

    Notes:
    -----
    The time varying water depth is imposed at the open boundary.

    """
    def __init__(self, dest, dim=2, rhow=1000.0):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)
        rhow : float
            constant 3-D density of water

        """
        self.g = 9.8
        self.dim = dim
        self.rhow = rhow
        super(SubCriticalTimeVaryingOutFlow, self).__init__(dest, None)

    def post_loop(self, d_dw, d_dw_inner_reimann, d_u, d_u_inner_reimann,
                  d_rho, d_cs, d_alpha, d_v, d_v_inner_reimann, d_idx,
                  d_dw_at_t):
        # Properties of open boundary particles
        # Time varying water depth imposed
        d_dw[d_idx] = d_dw_at_t[d_idx]
        d_rho[d_idx] = d_dw[d_idx] * self.rhow
        d_cs[d_idx] = sqrt(d_dw[d_idx] * self.g)
        d_alpha[d_idx] = d_rho[d_idx] * self.dim

        const = 2. * sqrt(self.g)
        d_u[d_idx] = (d_u_inner_reimann[d_idx]
                      + const*(sqrt(d_dw_inner_reimann[d_idx])
                               - sqrt(d_dw[d_idx])))
        d_v[d_idx] = d_v_inner_reimann[d_idx]


class SuperCriticalOutFlow(Equation):
    r"""**Supercritical outflow condition**

    .. math::

        v_{B,n} = v_{I,n}, v_{B,t} = v_{I,t}, d_B = d_I

    References
    ----------
    .. [Vacondio2012] R. Vacondio et al., "SPH modeling of shallow flow with
    open boundaries for practical flood simulation", J. Hydraul. Eng., 2012,
    138(6), pp. 530-541.

    Notes:
    -----
    For supercritical outflow condition, the velocity and water depth at the
    open boundary equals the respective inner Riemann state values. For
    supercritical inflow condition, both the velocity and water depth at the
    open boundary have to be imposed.

    """
    def __init__(self, dest, dim=2, rhow=1000.0):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)
        rhow : float
            constant 3-D density of water

        """
        self.g = 9.8
        self.dim = dim
        self.rhow = rhow
        super(SuperCriticalOutFlow, self).__init__(dest, None)

    def post_loop(self, d_dw, d_rho, d_dw_inner_reimann, d_u_inner_reimann,
                  d_u, d_v, d_v_inner_reimann, d_alpha, d_cs, d_idx):
        # Properties of open boundary particles
        d_u[d_idx] = d_u_inner_reimann[d_idx]
        d_v[d_idx] = d_v_inner_reimann[d_idx]
        d_dw[d_idx] = d_dw_inner_reimann[d_idx]
        d_rho[d_idx] = d_dw[d_idx] * self.rhow
        d_alpha[d_idx] = self.dim * d_rho[d_idx]
        d_cs[d_idx] = sqrt(self.g * d_dw[d_idx])


class GradientCorrectionPreStep(Equation):
    def __init__(self, dest, sources, dim=2):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)

        """
        self.dim = dim
        super(GradientCorrectionPreStep, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m_mat):
        i = declare('int')
        for i in range(9):
            d_m_mat[9*d_idx + i] = 0.0

    def loop_all(self, d_idx, d_m_mat, s_V, d_x, d_y, d_z, d_h, s_x,
                 s_y, s_z, s_h, SPH_KERNEL, NBRS, N_NBRS):
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        h = d_h[d_idx]
        i, j, s_idx, n = declare('int', 4)
        xij = declare('matrix(3)')
        dwij = declare('matrix(3)')
        n = self.dim
        for k in range(N_NBRS):
            s_idx = NBRS[k]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            hij = (h + s_h[s_idx]) * 0.5
            r = sqrt(xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2])
            SPH_KERNEL.gradient(xij, r, hij, dwij)
            dw = sqrt(dwij[0]*dwij[0] + dwij[1]*dwij[1]
                      + dwij[2]*dwij[2])
            V = s_V[s_idx]
            if r >= 1.0e-12:
                for i in range(n):
                    xi = xij[i]
                    for j in range(n):
                        xj = xij[j]
                        d_m_mat[9*d_idx + 3*i + j] += (dw*V*xi*xj) / r


class GradientCorrection(Equation):
    r"""**Kernel Gradient Correction**

    .. math::
            \nabla \tilde{W}_{ab} = L_{a}\nabla W_{ab}

    .. math::
            L_{a} = \left(\sum \frac{m_{b}}{\rho_{b}}\nabla W_{ab}
            \mathbf{\times}x_{ab} \right)^{-1}

    References
    ----------
    .. [Bonet and Lok, 1999]  J. Bonet and T.-S.L. Lok, "Variational and
    Momentum Preservation Aspects of Smoothed Particle Hydrodynamic
    Formulations", Comput. Methods Appl. Mech. Engrg., 180 (1999), pp. 97-115

    """
    def _get_helpers_(self):
        return [gj_solve, augmented_matrix]

    def __init__(self, dest, sources, dim=2, tol=0.5):
        r"""
        Parameters
        ----------
        dim : int
            number of space dimensions (Default: 2)
        tol : float
            tolerance for gradient correction (Default: 0.5)

        """
        self.dim = dim
        self.tol = tol
        super(GradientCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_m_mat, DWJ, s_h, s_idx):
        i, j, n = declare('int', 3)
        n = self.dim
        temp = declare('matrix(9)')
        aug_m = declare('matrix(12)')
        res = declare('matrix(3)')
        eps = 1.0e-04 * s_h[s_idx]
        for i in range(n):
            for j in range(n):
                temp[n*i + j] = d_m_mat[9*d_idx + 3*i + j]
        augmented_matrix(temp, DWJ, n, 1, n, aug_m)
        gj_solve(aug_m, n, 1, res)
        change = 0.0
        for i in range(n):
            change += abs(DWJ[i]-res[i]) / (abs(DWJ[i])+eps)
        if change <= self.tol:
            for i in range(n):
                DWJ[i] = res[i]


class RemoveOutofDomainParticles(Equation):
    r"""Removes particles if the following condition is satisfied:

    .. math::

        (x_i < x_min) or (x_i > x_max) or (y_i < y_min) or (y_i > y_max)

    """
    def __init__(self, dest, x_min=-1e9, x_max=1e9,
                 y_min=-1e9, y_max=1e9):
        r"""
        Parameters
        ----------
        x_min : float
            minimum distance along x-direction below which particles are
            removed
        x_max : float
            maximum distance along x-direction above which particles are
            removed
        y_min : float
            minimum distance along y-direction below which particles are
            removed
        y_max : float
            maximum distance along x-direction above which particles are
            removed

        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        super(RemoveOutofDomainParticles, self).__init__(dest, None)

    def initialize(self, d_pa_out_of_domain, d_x, d_y, d_idx):
        if (
            (d_x[d_idx] < self.x_min or d_x[d_idx] > self.x_max)
            or (d_y[d_idx] < self.y_min or d_y[d_idx] > self.y_max)
        ):
            d_pa_out_of_domain[d_idx] = 1
        else:
            d_pa_out_of_domain[d_idx] = 0

    def reduce(self, dst, t, dt):
        indices = declare('object')
        indices = numpy.where(dst.pa_out_of_domain > 0)[0]
        # Removes the out of domain particles
        if len(indices) > 0:
            dst.remove_particles(indices)


class RemoveCloseParticlesAtOpenBoundary(Equation):
    r"""Removes the newly created open boundary particle if the distance
    between this particle and any of its neighbor is less than min_dist_ob

    The following cases creates new open boundary particles

    * Particles which are moved back to the inlet after exiting the inlet.
    * Particles which have moved from another domain into the open boundary and
    have been converted to open boundary particles.

    References
    ----------
    .. [VacondioSWE-SPHysics, 2013] R. Vacondio et al., SWE-SPHysics source
    code, File: SWE_SPHYsics/SWE-SPHysics_2D_v1.0.00/source/SPHYSICS_SWE_2D/
    check_limits_2D.f

    """
    def __init__(self, dest, sources, min_dist_ob=0.0):
        """
        Parameters
        ----------
        min_dist_ob : float
            minimum distance of a newly created open boundary particle and its
            neighbor below which the particle is removed
        """
        self.min_dist_ob = min_dist_ob
        super(RemoveCloseParticlesAtOpenBoundary, self).__init__(dest, sources)

    def loop_all(self, d_idx, d_ob_pa_to_tag, d_ob_pa_to_remove, d_x, d_y, s_x,
                 s_y, NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('unsigned int')
        # ob_pa_to_tag is 1 for newly created open boundary particles
        if d_ob_pa_to_tag[d_idx]:
            xi = d_x[d_idx]
            yi = d_y[d_idx]
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                if s_idx == d_idx:
                    continue
                xij = xi - s_x[s_idx]
                yij = yi - s_y[s_idx]
                rij = sqrt(xij*xij + yij*yij)
                if rij < self.min_dist_ob:
                    d_ob_pa_to_remove[d_idx] = 1

    def reduce(self, dst, t, dt):
        indices = declare('object')
        indices = numpy.where(dst.ob_pa_to_remove > 0)[0]
        if len(indices) > 0:
            dst.remove_particles(indices)
        dst.ob_pa_to_tag = numpy.zeros_like(dst.ob_pa_to_tag)


class RemoveFluidParticlesWithNoNeighbors(Equation):
    r"""Removes fluid particles if there exists no neighboring particles within
    its kernel radius (2*smoothing length)

    """
    def loop_all(self, d_idx, d_ob_pa_to_tag, d_fluid_pa_to_remove, d_x, d_y,
                 s_x, s_y, d_h, NBRS, N_NBRS):
        i, n_nbrs_outside_ker = declare('int', 2)
        s_idx = declare('unsigned int')
        xi = d_x[d_idx]
        yi = d_y[d_idx]
        # Number of neighbors outside the particles kernel radius
        n_nbrs_outside_ker = 0
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            if s_idx == d_idx:
                continue
            xij = xi - s_x[s_idx]
            yij = yi - s_y[s_idx]
            rij = sqrt(xij*xij + yij*yij)
            if rij > 2*d_h[d_idx]:
                n_nbrs_outside_ker += 1
        # If all neighbors outside its kernel then tag particle for removal
        if n_nbrs_outside_ker == N_NBRS-1:
            d_fluid_pa_to_remove[d_idx] = 1
        else:
            d_fluid_pa_to_remove[d_idx] = 0

    def reduce(self, dst, t, dt):
        indices = declare('object')
        indices = numpy.where(dst.fluid_pa_to_remove > 0)[0]
        if len(indices) > 0:
            dst.remove_particles(indices)


class SWEInletOutletStep(IntegratorStep):
    r"""Stepper for both inlet and outlet particles for the cases dealing with
    shallow water flows

    """
    def initialize(self):
        pass

    def stage1(self, d_idx, d_x, d_y, d_uh, d_vh, d_u, d_v, dt):
        dtb2 = 0.5*dt
        d_uh[d_idx] = d_u[d_idx]
        d_vh[d_idx] = d_v[d_idx]
        d_x[d_idx] += dtb2 * d_u[d_idx]
        d_y[d_idx] += dtb2 * d_v[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_u, d_v, dt):
        dtb2 = 0.5*dt
        d_x[d_idx] += dtb2 * d_u[d_idx]
        d_y[d_idx] += dtb2 * d_v[d_idx]


class SWEInlet(object):
    """This inlet is used for shallow water flows. It has particles
    stacked along a particular axis (defaults to 'x'). These particles can
    move along any direction and as they flow out of the domain they are copied
    into the destination particle array at each timestep.

    Inlet particles are stacked by subtracting the spacing amount from the
    existing inlet array. These are copied when the inlet is created. The
    particles that cross the inlet domain are copied over to the destination
    particle array and moved back to the other side of the inlet.

    The particles from the source particle array which have moved to the inlet
    domain are removed from the source and added to the inlet particle array.

    The motion of the particles can be along any direction required.  One
    can set the 'u' velocity to have a parabolic profile in the 'y' direction
    if so desired.

    """
    def __init__(self, inlet_pa, dest_pa, source_pa, spacing, n=5, axis='x',
                 xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, callback=None):
        """Constructor.

        Note that the inlet must be defined such that the spacing times the
        number of stacks of particles is equal to the length of the domain in
        the stacked direction.  For example, if particles are stacked along
        the 'x' axis and n=5 with spacing 0.1, then xmax - xmin should be 0.5.

        Parameters
        ----------

        inlet_pa: ParticleArray
           Particle array for the inlet particles.

        dest_pa: ParticleArray
           Particle array for the destination into which inlet flows.

        source_pa : ParticleArray
            Particle array from which the particles flow in.

        spacing: float
           Spacing of particles in the inlet domain.

        n: int
           Total number of copies of the initial particles.

        axis: str
           Axis along which to stack particles, one of 'x', 'y'.

        xmin, xmax, ymin, ymax : float
           Domain of the outlet.

        """
        self.inlet_pa = inlet_pa
        self.dest_pa = dest_pa
        self.spacing = spacing
        self.source_pa = source_pa
        self.callback = callback
        assert axis in ('x', 'y')
        self.axis = axis
        self.n = n
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self._create_inlet_particles()

    def _create_inlet_particles(self):
        props = self.inlet_pa.get_property_arrays()
        inlet_props = {}
        for prop, array in props.items():
            new_array = np.array([], dtype=array.dtype)
            for i in range(1, self.n):
                if prop == self.axis:
                    new_array = np.append(new_array, array - i*self.spacing)
                else:
                    new_array = np.append(new_array, array)
            inlet_props[prop] = new_array
        self.inlet_pa.add_particles(**inlet_props)

    def update(self, t, dt, stage):
        """This is called by the solver after each timestep and is passed
        the solver instance.
        """
        pa_add = {}
        inlet_pa = self.inlet_pa
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        lx, ly = xmax - xmin, ymax - ymin
        x, y = inlet_pa.x, inlet_pa.y

        xcond = (x > xmax)
        ycond = (y > ymax)
        # All the indices of particles which have left.
        all_idx = np.where(xcond | ycond)[0]
        # The indices which need to be wrapped around.
        x_idx = np.where(xcond)[0]
        y_idx = np.where(ycond)[0]

        # Adding particles to the destination array.
        props = inlet_pa.get_property_arrays()
        for prop, array in props.items():
            pa_add[prop] = np.array(array[all_idx])
        self.dest_pa.add_particles(**pa_add)

        # Moving the moved particles back to the array beginning.
        inlet_pa.x[x_idx] -= np.sign(inlet_pa.x[x_idx] - xmax)*lx
        inlet_pa.y[y_idx] -= np.sign(inlet_pa.y[y_idx] - ymax)*ly

        # Tags the particles which have been moved back to inlet. These tagged
        # particles are then used for checking minimum spacing condition
        # with other open boundary particles.
        inlet_pa.ob_pa_to_tag[all_idx] = 1

        source_pa = self.source_pa
        x, y = source_pa.x, source_pa.y
        idx = np.where((x <= xmax) & (x >= xmin) & (y <= ymax) & (y >=
                       ymin))[0]

        # Adding particles to the destination array.
        pa_add = {}
        props = source_pa.get_property_arrays()
        for prop, array in props.items():
            pa_add[prop] = np.array(array[idx])

        # Tags the particles which have been added to the destination array
        # from the source array. These tagged particles are then used for
        # checking minimum spacing condition with other open boundary
        # particles.
        pa_add['ob_pa_to_tag'] = np.ones_like(pa_add['ob_pa_to_tag'])

        if self.callback is not None:
            self.callback(inlet_pa, pa_add)

        inlet_pa.add_particles(**pa_add)

        source_pa.remove_particles(idx)

        # Removing the particles that moved out of inlet
        x, y = inlet_pa.x, inlet_pa.y
        idx = np.where((x > xmax) | (x < xmin) | (y > ymax) | (y < ymin))[0]
        inlet_pa.remove_particles(idx)
